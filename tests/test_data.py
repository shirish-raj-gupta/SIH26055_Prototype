"""Dataset builder, loader, credential handling and the external bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from smartscan.config import load_config
from smartscan.credentials import (
    CredentialStatus,
    credential_status,
    fingerprint,
    load_dotenv,
    require,
)
from smartscan.data.schema import (
    DATASET_VERSION,
    SIZE_BUDGET_BYTES,
    SPLITS,
    TIER_COUNTS,
    episode_id,
    pack_occupancy,
    split_for_seed,
    unpack_occupancy,
)

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """Build a small corpus once for the whole module."""
    from smartscan.data.dataset_builder import build_dataset

    root = tmp_path_factory.mktemp("corpus")
    build_dataset(root, counts={"easy": 2, "medium": 2, "hard": 1}, verbose=False)
    return root


# --------------------------------------------------------------------------- #
# Schema and splits
# --------------------------------------------------------------------------- #
def test_episode_id_is_content_derived():
    """Ids must not be positional, or adding an episode renumbers the corpus."""
    assert episode_id("medium", 20260902) == "medium_20260902"
    assert episode_id("easy", 7) != episode_id("hard", 7)


def test_split_assignment_is_deterministic_and_stable():
    seeds = range(20260902, 20260902 + 400)
    once = [split_for_seed(s) for s in seeds]
    assert once == [split_for_seed(s) for s in seeds]
    assert set(once) == set(SPLITS)


def test_split_proportions_are_roughly_as_configured():
    seeds = range(100_000, 110_000)
    counts: dict[str, int] = {}
    for s in seeds:
        counts[split_for_seed(s)] = counts.get(split_for_seed(s), 0) + 1
    for name, want in SPLITS.items():
        assert counts[name] / 10_000 == pytest.approx(want, abs=0.02)


def test_a_seed_never_appears_in_two_splits():
    """The leakage guard. Splitting by seed is what makes the held-out set real."""
    for seed in range(20260902, 20260902 + 2000):
        assert len({split_for_seed(seed) for _ in range(5)}) == 1


def test_occupancy_packing_round_trips():
    rng = np.random.default_rng(0)
    occ = rng.random((128, 997)) < 0.1
    packed = pack_occupancy(occ)
    assert packed.dtype == np.uint8
    assert packed.nbytes < occ.nbytes / 7  # 1 bit vs 1 byte per cell
    assert np.array_equal(unpack_occupancy(packed, occ.shape), occ)


def test_tier_counts_match_the_brief():
    assert TIER_COUNTS == {"easy": 1000, "medium": 1200, "hard": 800}
    assert sum(TIER_COUNTS.values()) == 3000


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def test_build_writes_every_required_artefact(corpus):
    import pandas as pd

    assert (corpus / "index.parquet").is_file()
    assert (corpus / "dataset_card.md").is_file()
    assert (corpus / "build_report.json").is_file()

    index = pd.read_parquet(corpus / "index.parquet")
    assert len(index) == 5
    assert set(index["tier"]) == {"easy", "medium", "hard"}
    for _, row in index.iterrows():
        ep = corpus / row["path"]
        for name in ("truth_occupancy.npz", "emitter_manifest.parquet", "observations.parquet"):
            assert (ep / name).is_file(), f"{row['episode_id']} missing {name}"


def test_index_split_matches_the_hash_function(corpus):
    import pandas as pd

    index = pd.read_parquet(corpus / "index.parquet")
    for _, row in index.iterrows():
        assert row["split"] == split_for_seed(int(row["seed"]))


def test_tiers_use_disjoint_seed_blocks(corpus):
    """A shared seed would put the same scenario in two tiers under two labels."""
    import pandas as pd

    index = pd.read_parquet(corpus / "index.parquet")
    by_tier = index.groupby("tier")["seed"].apply(set)
    tiers = list(by_tier.index)
    for i, a in enumerate(tiers):
        for b in tiers[i + 1 :]:
            assert not (by_tier[a] & by_tier[b]), f"{a} and {b} share seeds"


def test_dataset_card_documents_schema_and_limitations(corpus):
    card = (corpus / "dataset_card.md").read_text(encoding="utf-8")
    for required in (
        "Known limitations", "Licence", "CC BY-SA 4.0", "Splits",
        "truth_occupancy.npz", "emitter_manifest.parquet", "observations.parquet",
        "Simulator commit", "Source digest", "Units",
    ):
        assert required in card, f"dataset card is missing {required!r}"
    # It must be explicit about not mirroring the gated dataset.
    assert "not" in card.lower() and "mirror" in card.lower()


def test_build_report_carries_provenance(corpus):
    report = json.loads((corpus / "build_report.json").read_text(encoding="utf-8"))
    assert report["dataset_version"] == DATASET_VERSION
    assert len(report["source_digest"]) == 32
    assert report["n_episodes"] == 5
    assert report["total_bytes"] > 0


def test_size_budget_is_enforced(tmp_path):
    """The builder must refuse to exceed Kaggle's ceiling, not discover it mid-upload."""
    from smartscan.data.dataset_builder import build_dataset

    with pytest.raises(RuntimeError, match="size budget exceeded"):
        build_dataset(
            tmp_path / "toobig", counts={"easy": 3}, size_budget_bytes=1024, verbose=False
        )


def test_size_budget_matches_the_documented_kaggle_limits():
    """The two Kaggle limits are different numbers and get confused constantly.

    100 GB is the per-user *dataset storage* quota, which is what the builder
    must respect. 20 GB is a *notebook's writable disk*; attached datasets are
    mounted read-only and do not count against it.
    """
    from smartscan.data.schema import NOTEBOOK_WORKING_BYTES

    assert SIZE_BUDGET_BYTES == 100 * 1024**3
    assert NOTEBOOK_WORKING_BYTES == 20 * 1024**3
    assert SIZE_BUDGET_BYTES > NOTEBOOK_WORKING_BYTES


def test_duty_is_serialised(corpus):
    """duty was documented but not written before schema 1.1.0.

    A frequency-agile emitter hopping inside one slot, or a 1 us pulse in a
    1 ms slot, is representable ONLY in duty -- not in the binary occupancy.
    """
    from smartscan.data.kaggle_io import load_dataset

    rec = load_dataset("all", root=corpus).load(0)
    assert rec.duty is not None, "duty missing from the serialised tensors"
    assert rec.duty.shape == rec.occupancy.shape
    assert ((rec.duty > 0) == rec.occupancy).all()
    assert (rec.duty[rec.occupancy] <= 1.0).all()


def test_observations_carry_every_replayed_agent(corpus):
    import pandas as pd

    from smartscan.data.dataset_builder import DEFAULT_AGENTS

    index = pd.read_parquet(corpus / "index.parquet")
    obs = pd.read_parquet(corpus / index.iloc[0]["path"] / "observations.parquet")
    assert set(obs["agent"]) == set(DEFAULT_AGENTS)
    assert (obs["step"] >= 0).all()
    assert obs["hit_mask"].dtype == np.uint8


def test_observation_snr_is_the_reported_estimate_not_ground_truth(corpus):
    """A model trained on true SNR would not be deployable.

    The receiver reports SNR with estimation noise; the trace must carry that
    noisy value, so the recorded numbers should differ from the ground-truth
    tensor at the same cells.
    """

    from smartscan.data.kaggle_io import load_dataset

    ds = load_dataset("all", root=corpus)
    rec = ds.load(0)
    obs = rec.observations[rec.observations["agent"] == "sequential"]

    diffs = []
    for lo, t, mask, snr_row in zip(
        obs["window_lo"].to_numpy(), obs["t"].to_numpy(),
        obs["hit_mask"].to_numpy(), obs["snr_est_db"].to_numpy(), strict=True,
    ):
        bits = np.unpackbits(np.uint8(mask), bitorder="little")[: len(snr_row)]
        for i, bit in enumerate(bits):
            if bit and np.isfinite(snr_row[i]):
                truth = rec.snr_db[int(lo) + i, int(t)]
                if truth > -100:
                    diffs.append(abs(float(snr_row[i]) - float(truth)))
    assert diffs, "no reported SNR values found to compare"
    # Estimation noise is ~2 dB by default, so the values must not be identical.
    assert np.mean(diffs) > 0.1, "reported SNR looks like ground truth"


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_load_from_local_root(corpus):
    from smartscan.data.kaggle_io import load_dataset

    ds = load_dataset("all", root=corpus)
    assert ds.source == "local"
    assert len(ds) == 5
    rec = ds.load(0)
    assert rec.occupancy.dtype == bool
    assert rec.occupancy.shape == rec.snr_db.shape == rec.emitter_id.shape
    assert len(rec.manifest) > 0


def test_tier_and_split_filtering(corpus):
    from smartscan.data.kaggle_io import load_dataset

    everything = load_dataset("all", root=corpus)
    easy = load_dataset("all", tier="easy", root=corpus)
    assert 0 < len(easy) <= len(everything)
    assert set(easy.index["tier"]) == {"easy"}


def test_unknown_split_or_tier_is_rejected(corpus):
    from smartscan.data.kaggle_io import load_dataset

    with pytest.raises(ValueError, match="unknown split"):
        load_dataset("nonsense", root=corpus)
    with pytest.raises(ValueError, match="unknown tier"):
        load_dataset("all", tier="impossible", root=corpus)


def test_verify_dataset_passes_on_a_good_corpus(corpus):
    from smartscan.data.kaggle_io import verify_dataset

    report = verify_dataset(corpus, sample=5)
    assert report["ok"], report["problems"]
    assert report["n_episodes"] == 5


def test_verify_detects_a_missing_file(corpus, tmp_path):
    """Integrity checking must actually catch corruption."""
    import shutil

    from smartscan.data.kaggle_io import verify_dataset

    copy = tmp_path / "damaged"
    shutil.copytree(corpus, copy)
    import pandas as pd

    victim = pd.read_parquet(copy / "index.parquet").iloc[0]["path"]
    (copy / victim / "observations.parquet").unlink()

    report = verify_dataset(copy, sample=5)
    assert not report["ok"]
    assert any("observations.parquet" in p for p in report["problems"])


@pytest.mark.slow
def test_offline_fallback_regenerates_identical_episodes(corpus):
    """The demo must not fail because of wifi.

    A regenerated episode has to be byte-identical to the published one, or the
    fallback is quietly producing different data under the same episode id.
    """
    from smartscan.data.kaggle_io import _regenerate_episode, load_dataset

    ds = load_dataset("all", root=corpus)
    on_disk = ds.load(0, with_observations=False)
    regenerated = _regenerate_episode(on_disk.tier, on_disk.seed, with_observations=False)

    assert np.array_equal(on_disk.occupancy, regenerated.occupancy)
    assert np.array_equal(on_disk.emitter_id, regenerated.emitter_id)
    # SNR round-trips through float16 on disk, so compare within that precision.
    assert np.allclose(on_disk.snr_db, regenerated.snr_db, atol=0.5)


def test_fallback_engages_when_nothing_is_reachable(tmp_path):
    """An explicit-but-missing root must regenerate, not silently load another.

    Falling through to whatever corpus happens to sit in the working directory
    would hand back data the caller never asked for, under the episode ids of
    the one they did.
    """
    from smartscan.data.kaggle_io import load_dataset

    with pytest.warns(UserWarning, match="not substituting"):
        ds = load_dataset(
            "train", tier="easy", root=tmp_path / "missing",
            cache_dir=tmp_path / "empty-cache", allow_download=False, n_episodes=2,
        )
    assert ds.source == "regenerated"
    assert ds.root is None
    assert len(ds) == 2


def test_implicit_discovery_still_works_without_an_explicit_root(corpus, monkeypatch):
    """With no root given, the usual local candidates are still searched."""
    from smartscan.data.kaggle_io import resolve_dataset_root

    monkeypatch.setenv("SMARTSCAN_DATA", str(corpus))
    found, source = resolve_dataset_root(None, allow_download=False)
    assert found == corpus
    assert source == "local"


@pytest.mark.slow
def test_window_dataset_shapes_and_masking(corpus):
    pytest.importorskip("torch")
    from smartscan.data.kaggle_io import OccupancyWindowDataset, load_dataset

    ds = load_dataset("all", root=corpus)
    wd = OccupancyWindowDataset(ds, window=64, stride=128, max_windows_per_episode=4)
    assert len(wd) > 0
    x, y, mask, y_true = wd[0]
    cfg = load_config("easy.yaml")
    assert tuple(x.shape) == (4, cfg.n_channels, 64)
    assert tuple(y.shape) == tuple(mask.shape) == tuple(y_true.shape) == (cfg.n_channels,)
    # The label mask must cover exactly the K channels the receiver was tuned to.
    assert int(mask.sum()) == cfg.receiver.ibw_channels


@pytest.mark.slow
def test_class_balanced_sampler_is_produced_only_when_requested(corpus):
    pytest.importorskip("torch")
    from smartscan.data.kaggle_io import OccupancyWindowDataset, load_dataset

    ds = load_dataset("all", root=corpus)
    plain = OccupancyWindowDataset(ds, window=64, stride=256, max_windows_per_episode=3)
    balanced = OccupancyWindowDataset(
        ds, window=64, stride=256, max_windows_per_episode=3, class_balanced=True
    )
    assert plain.sampler() is None
    assert balanced.sampler(0) is not None


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def test_fingerprint_hides_the_value():
    secret = "KGAT_supersecrettokenvalue"
    fp = fingerprint(secret)
    assert len(fp) == 8
    assert secret not in fp
    assert fingerprint(secret) == fp          # stable
    assert fingerprint(secret + "x") != fp    # sensitive
    assert fingerprint(None) == "-"


def test_dotenv_skips_placeholders(tmp_path, monkeypatch):
    """A half-filled template must not authenticate as 'your-kaggle-username'."""
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "KAGGLE_USERNAME=your-kaggle-username\n"
        "export SMARTSCAN_DATASET_SLUG='real-value'\n"
        "\n"
        "MALFORMED\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SMARTSCAN_DATASET_SLUG", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    applied = load_dotenv(env)
    assert "SMARTSCAN_DATASET_SLUG" in applied
    assert "KAGGLE_USERNAME" not in applied
    assert os.environ["SMARTSCAN_DATASET_SLUG"] == "real-value"


def test_dotenv_does_not_override_a_real_export(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SMARTSCAN_CACHE=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SMARTSCAN_CACHE", "from-shell")
    load_dotenv(env)
    assert os.environ["SMARTSCAN_CACHE"] == "from-shell"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_credential_report_never_contains_a_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "TOPSECRETKEYVALUE")
    status = credential_status(dotenv=None)
    report = status.report()
    assert "TOPSECRETKEYVALUE" not in report
    assert status.kaggle_key_fingerprint in report
    assert isinstance(status, CredentialStatus)


def test_require_names_what_is_missing(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require("DEFINITELY_NOT_SET", dotenv=None)
    message = str(exc.value)
    assert "DEFINITELY_NOT_SET" in message
    assert ".env" in message


# --------------------------------------------------------------------------- #
# External bridge (gated dataset)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_external_report_is_labelled_and_never_crashes():
    """Whether or not access is granted, the report must be usable and tagged.

    Without a grant it must carry actionable guidance rather than raising; with
    one it must still be tagged ``external`` so a downstream table cannot pool
    it with synthetic results by accident.
    """
    from smartscan.data.tsrd_bridge import TSRD_LICENCE, external_validation_report

    report = external_validation_report(load_config("easy.yaml"), max_records=1)
    assert report["external"] is True, "external results must be tagged"
    assert "turing-synthetic-radar-dataset" in report["source"]
    assert "Apache-2.0" in report["licence_note"]
    assert "not" in report["licence_note"] and "mirror" in report["licence_note"]
    assert TSRD_LICENCE.startswith("Apache-2.0")

    if report["available"]:
        assert report["n_records"] >= 1
        assert report["rows"], "available report must carry rows"
        for row in report["rows"]:
            assert {"agent", "record", "n_pdws", "twir_rate"} <= set(row)
    else:
        assert "GATED" in report["reason"]
        assert "huggingface.co" in report["reason"]


@pytest.mark.slow
def test_live_tsrd_access_if_granted():
    """Exercise the real gated dataset when a token and grant are present.

    Skipped otherwise, so the suite is green for anyone without access -- but it
    must not be silently absent for those who do, or the bridge rots.
    """
    pytest.importorskip("h5py")
    pytest.importorskip("huggingface_hub")
    from smartscan.data.tsrd_bridge import (
        TSRDUnavailableError,
        bin_pdws_to_tensors,
        load_tsrd_split,
        token_available,
    )

    if not token_available():
        pytest.skip("no Hugging Face token configured")
    try:
        streams = load_tsrd_split("test", subset="archive", max_records=1)
    except TSRDUnavailableError as exc:
        pytest.skip(f"TSRD not accessible: {str(exc).splitlines()[0]}")

    stream = streams[0]
    assert len(stream) > 100
    # Units must be SI after loading: TSRD stores microseconds and MHz.
    assert 1.0 < stream.duration_s < 60.0, "ToA was not converted from microseconds"
    lo, hi = stream.band_hz
    assert 1e8 < lo < 2e10 and 1e8 < hi < 2e10, "RF was not converted from MHz"
    assert (stream.pw_s > 0).all() and stream.pw_s.max() < 1e-3

    cfg = load_config("medium.yaml")
    ep = bin_pdws_to_tensors(stream, cfg)
    assert ep.seed == -1
    assert (ep.occupancy > 0).any(), "no pulse landed in band"
    snr = ep.snr_db[ep.occupancy > 0]
    # The physical mapping should put the bulk of pulses in a plausible
    # intercept range, not at an arbitrary offset.
    assert -60 < float(np.median(snr)) < 60


def test_pdw_binning_produces_valid_episode_tensors():
    """The adapter must yield tensors the receiver model can consume unchanged."""
    from smartscan.data.tsrd_bridge import PDWStream, bin_pdws_to_tensors

    cfg = load_config("easy.yaml")
    grid = cfg.grid()
    rng = np.random.default_rng(0)
    n = 4000
    stream = PDWStream(
        toa_s=np.sort(rng.uniform(0, 5.0, n)),
        rf_hz=rng.choice(grid.centers_hz, n),
        pw_s=np.full(n, 1e-6),
        amplitude_db=rng.uniform(5, 30, n),
        emitter_id=rng.integers(1, 6, n).astype(np.int32),
        source="synthetic-pdw-test",
    )
    ep = bin_pdws_to_tensors(stream, cfg, treat_pa_as_dbm=False)

    assert ep.occupancy.shape == ep.snr_db.shape == ep.emitter_id.shape
    assert ep.occupancy.dtype == np.uint8
    assert ep.n_slots <= cfg.n_slots
    assert ep.seed == -1, "external episodes must not claim to be seed-derived"
    assert len(ep.truth) > 0
    assert all(t.emitter_class == "ExternalPDW" for t in ep.truth)
    # Every occupied cell must carry a real SNR, and empty cells the floor.
    assert (ep.snr_db[ep.occupancy > 0] > -200).all()


def test_pdw_binning_drops_out_of_band_pulses():
    """Clamping to the edge channels would manufacture emitters that are not there."""
    from smartscan.data.tsrd_bridge import PDWStream, bin_pdws_to_tensors

    cfg = load_config("easy.yaml")
    grid = cfg.grid()
    stream = PDWStream(
        toa_s=np.array([0.0, 0.001, 0.002, 0.003]),
        rf_hz=np.array([grid.f_start_hz - 5e9, 1e9, grid.f_stop_hz + 5e9, 2e9]),
        pw_s=np.full(4, 1e-6),
        amplitude_db=np.full(4, 20.0),
        emitter_id=np.array([1, 2, 3, 4], dtype=np.int32),
    )
    ep = bin_pdws_to_tensors(stream, cfg, treat_pa_as_dbm=False)
    # Only the two in-band pulses survive.
    assert int(ep.n_pulses.sum()) == 2


def test_empty_pdw_stream_is_rejected():
    from smartscan.data.tsrd_bridge import PDWStream, bin_pdws_to_tensors

    empty = PDWStream(*(np.zeros(0) for _ in range(4)))
    with pytest.raises(ValueError, match="empty"):
        bin_pdws_to_tensors(empty, load_config("easy.yaml"))
