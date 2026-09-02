"""Build the publishable RF-environment corpus from the simulator.

Produces, for each of 3000 episodes (1000 EASY / 1200 MEDIUM / 800 HARD):

* ``truth_occupancy.npz`` -- bit-packed occupancy plus SNR/emitter-id at
  occupied cells (see :mod:`smartscan.data.schema` for why that layout);
* ``emitter_manifest.parquet`` -- the ground-truth order of battle;
* ``observations.parquet`` -- replayed dwell traces from seven schedulers. The
  three open-loop baselines are what the supervised predictor should train on;
  the four closed-loop policies are there for offline policy evaluation.

Plus a root ``index.parquet``, a ``dataset_card.md`` and Kaggle metadata.

Two properties are load-bearing:

**Splits are assigned by scenario seed, never by time slice.** Cutting a single
episode into a train half and a test half would let a model memorise the very
emitters it is about to be scored on, and the leakage would be invisible in
every aggregate metric. Enforced by :func:`smartscan.data.schema.split_for_seed`
and asserted in ``tests/test_data.py``.

**The size budget is asserted, not hoped for.** Kaggle's per-user dataset quota
is 100 GB (a *notebook's* writable disk is the separate 20 GB figure -- see
:mod:`smartscan.data.schema`); the builder measures as it goes and fails with a
specific number rather than part-way through an upload.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from smartscan.agents import build_agent
from smartscan.config import Config, load_config
from smartscan.data.schema import (
    DATASET_VERSION,
    SIZE_BUDGET_BYTES,
    TIER_COUNTS,
    episode_id,
    pack_occupancy,
    split_for_seed,
)
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.runner import run_episode

__all__ = [
    "BASELINE_AGENTS",
    "DEFAULT_AGENTS",
    "BuildReport",
    "build_dataset",
    "build_one_episode",
    "seed_block",
    "source_fingerprint",
    "write_dataset_card",
]

#: Schedulers whose traces are replayed into ``observations.parquet``.
#:
#: The three open-loop baselines come first and are what the predictor should
#: train on: their coverage is close to uniform, so the training distribution is
#: not biased toward the behaviour of the policy a model is later meant to
#: improve on.
#:
#: The closed-loop policies are included as well, because at 100 GB the space is
#: free and their traces are what an *offline* policy-evaluation study needs --
#: you cannot estimate the value of a policy you have no data from. Consumers
#: training a predictor should filter to :data:`BASELINE_AGENTS`; the
#: ``agent`` column makes that a one-line selection.
BASELINE_AGENTS: tuple[str, ...] = ("sequential", "random", "priority_rr")

DEFAULT_AGENTS: tuple[str, ...] = (
    *BASELINE_AGENTS, "ucb1", "thompson", "whittle", "coprime_sweep",
)


def _require_pandas() -> Any:
    """Import pandas, or explain how to get it."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas and pyarrow are required to build the dataset; "
            'install `pip install "smartscan[viz]"`.'
        ) from exc
    return pd


#: Seed offsets for the canonical tiers. These are **frozen**: ``episode_id``
#: embeds the seed, so changing an offset would silently repoint every published
#: id at different data. New tiers get a hashed block above this range instead.
_CANONICAL_BLOCKS: dict[str, int] = {"easy": 0, "medium": 1_000_000, "hard": 2_000_000}


def seed_block(tier: str) -> int:
    """Return the disjoint seed offset for a tier.

    The three canonical tiers keep their original offsets, because ``episode_id``
    embeds the seed and a published dataset must stay addressable. Any other
    tier gets a hashed block starting well above them, so an added tier can
    neither collide with an existing one nor perturb it -- the previous
    ``blocks.get(tier, 0)`` silently gave every new tier the same seeds as EASY.

    Args:
        tier: Tier name.

    Returns:
        Seed offset for that tier.
    """
    if tier in _CANONICAL_BLOCKS:
        return _CANONICAL_BLOCKS[tier]
    from smartscan.seeding import stable_hash

    return 10_000_000 + (stable_hash(f"tier:{tier}") % 900) * 1_000_000


def source_fingerprint() -> dict[str, str]:
    """Identify the exact simulator that produced a dataset.

    Prefers the git commit; falls back to a blake2b digest over every ``.py`` in
    the package. The repository may legitimately not be a git checkout (a Kaggle
    kernel, a zip download), and a dataset with no provenance at all is worse
    than one fingerprinted by content.

    Returns:
        Dict with ``git_commit`` (or ``'unavailable'``), ``source_digest``,
        ``dataset_version`` and ``built_utc``.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        git = commit.stdout.strip() if commit.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        git = "unavailable"

    root = Path(__file__).resolve().parent.parent
    h = hashlib.blake2b(digest_size=16)
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return {
        "git_commit": git,
        "source_digest": h.hexdigest(),
        "dataset_version": DATASET_VERSION,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@dataclass
class BuildReport:
    """Outcome of a dataset build.

    Attributes:
        root: Dataset root directory.
        n_episodes: Episodes written.
        total_bytes: On-disk size.
        per_tier: Episode count per tier.
        per_split: Episode count per split.
        wall_time_s: Wall-clock build time.
        fingerprint: Simulator provenance.
    """

    root: Path
    n_episodes: int = 0
    total_bytes: int = 0
    per_tier: dict[str, int] = field(default_factory=dict)
    per_split: dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    fingerprint: dict[str, str] = field(default_factory=dict)

    @property
    def gigabytes(self) -> float:
        """On-disk size in GB."""
        return self.total_bytes / 1024**3

    def summary(self) -> str:
        """Return a one-screen human summary."""
        used = 100 * self.total_bytes / SIZE_BUDGET_BYTES
        quota = SIZE_BUDGET_BYTES // 1024**3
        return "\n".join(
            [
                f"{self.n_episodes} episodes, {self.gigabytes:.2f} GB "
                f"({used:.2f}% of the {quota} GB budget)",
                f"  tiers  {self.per_tier}",
                f"  splits {self.per_split}",
                f"  built in {self.wall_time_s:.1f}s -> {self.root}",
            ]
        )


def _manifest_rows(eid: str, episode: Any) -> list[dict[str, Any]]:
    """Flatten an episode's ground-truth emitters into manifest rows."""
    rows = []
    for t in episode.truth:
        p = t.params
        hop = p.get("hop_set", ())
        rows.append({
            "episode_id": eid,
            "emitter_id": np.int16(t.emitter_id),
            "emitter_class": t.emitter_class,
            "f_centre_hz": float(t.f_center_hz),
            "home_channel": np.int16(t.home_channel),
            "scan_period_s": float(t.scan_period_s),
            "beamwidth_deg": float(p.get("beamwidth_deg", np.nan)),
            "pri_s": float(p.get("pri_s", np.nan)),
            "pulse_width_s": float(p.get("pulse_width_s", np.nan)),
            "hop_set": ",".join(str(int(c)) for c in hop) if hop else "",
            "hop_rate_hz": float(p.get("hop_rate_hz", np.nan)),
            "threat_priority": np.float32(t.threat_priority),
            "t_first_active": np.int32(t.t_first_active),
            "is_novel": bool(t.is_novel),
            "is_interferer": bool(t.is_interferer),
            "detection_mode": t.detection_mode,
            "eirp_dbm": np.float32(getattr(t, "eirp_dbm", np.nan)),
            "range_km": np.float32(getattr(t, "range_km", np.nan)),
        })
    return rows


def _observation_frame(eid: str, results: dict[str, Any], k: int, pd: Any) -> Any:
    """Build the wide observation table for one episode across schedulers."""
    frames = []
    for agent, res in results.items():
        n = res.n_steps
        hit_mask = np.zeros(n, dtype=np.uint8)
        true_mask = np.zeros(n, dtype=np.uint8)

        # Replayed from the recorded masks rather than re-simulated, so the
        # parquet cannot disagree with the metrics computed alongside it.
        for i, t in enumerate(res.dwell_slots):
            lo = int(res.window_lo[i])
            hit_mask[i] = np.packbits(res.hit_mask[lo : lo + k, t], bitorder="little")[0]
            true_mask[i] = np.packbits(res.true_hit_mask[lo : lo + k, t], bitorder="little")[0]

        # slots_elapsed is exactly the gap between consecutive dwell slots.
        elapsed = np.clip(
            np.diff(np.concatenate([[-1], res.dwell_slots])), 1, 127
        ).astype(np.int8)
        snr = res.snr_est_db.astype(np.float16)

        frames.append(pd.DataFrame({
            "episode_id": eid,
            "agent": agent,
            "step": np.arange(n, dtype=np.int32),
            "t": res.dwell_slots.astype(np.int32),
            "action": res.actions.astype(np.int16),
            "window_lo": res.window_lo.astype(np.int16),
            "slots_elapsed": elapsed,
            "hit_mask": hit_mask,
            "true_hit_mask": true_mask,
            "snr_est_db": list(snr),
            "reward": res.rewards.astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True)


def build_one_episode(
    tier: str,
    seed: int,
    config: Config,
    out_root: Path,
    agents: Sequence[str] = DEFAULT_AGENTS,
) -> dict[str, Any]:
    """Generate, replay and serialise one episode.

    Args:
        tier: Difficulty tier.
        seed: Scenario seed.
        config: Resolved configuration for that tier.
        out_root: Dataset root directory.
        agents: Schedulers whose traces are replayed.

    Returns:
        The ``index.parquet`` row for this episode.
    """
    pd = _require_pandas()
    eid = episode_id(tier, seed)
    rel = Path("episodes") / tier / eid
    ep_dir = out_root / rel
    ep_dir.mkdir(parents=True, exist_ok=True)

    scenario = generate_scenario(seed, config=config)
    episode = build_episode(scenario)
    occupied = episode.occupancy > 0

    np.savez_compressed(
        ep_dir / "truth_occupancy.npz",
        occupancy_packed=pack_occupancy(occupied),
        snr_db=episode.snr_db[occupied].astype(np.float16),
        duty=episode.duty[occupied].astype(np.float16),
        emitter_id=episode.emitter_id[occupied].astype(np.int16),
        n_pulses=np.clip(episode.n_pulses[occupied], -32768, 32767).astype(np.int16),
        shape=np.asarray(episode.occupancy.shape, dtype=np.int32),
    )

    pd.DataFrame(_manifest_rows(eid, episode)).to_parquet(
        ep_dir / "emitter_manifest.parquet", index=False, compression="zstd"
    )

    results = {
        key: run_episode(
            config, seed, build_agent(key, config, seed, scenario),
            scenario=scenario, episode=episode,
        )
        for key in agents
    }
    _observation_frame(eid, results, config.receiver.ibw_channels, pd).to_parquet(
        ep_dir / "observations.parquet", index=False, compression="zstd"
    )

    size = sum(f.stat().st_size for f in ep_dir.iterdir() if f.is_file())
    return {
        "episode_id": eid,
        "tier": tier,
        "seed": int(seed),
        "split": split_for_seed(seed),
        "n_emitters": np.int32(len(episode.truth)),
        "n_channels": np.int32(episode.n_channels),
        "n_slots": np.int32(episode.n_slots),
        "dt_s": float(episode.dt_s),
        "occupancy_frac": np.float32(occupied.mean()),
        "n_popup": np.int32(sum(1 for t in episode.truth if t.t_first_active > 0)),
        "n_interferer": np.int32(sum(1 for t in episode.truth if t.is_interferer)),
        "config_hash": config.hash(),
        "path": rel.as_posix(),
        "bytes": np.int64(size),
    }


def build_dataset(
    out_dir: str | Path = "build/dataset",
    counts: dict[str, int] | None = None,
    agents: Sequence[str] = DEFAULT_AGENTS,
    root_seed: int = 20260902,
    n_jobs: int = 1,
    size_budget_bytes: int = SIZE_BUDGET_BYTES,
    verbose: bool = True,
) -> BuildReport:
    """Build the full corpus.

    Args:
        out_dir: Dataset root.
        counts: Episodes per tier; defaults to the brief's 1000/1200/800.
        agents: Schedulers whose traces are replayed.
        root_seed: First scenario seed; each tier gets a disjoint block so the
            same scenario never appears under two tiers.
        n_jobs: Parallel workers (joblib). ``1`` keeps everything in-process.
        size_budget_bytes: Hard ceiling; the build aborts if exceeded.
        verbose: Print progress.

    Returns:
        A :class:`BuildReport`.

    Raises:
        RuntimeError: If the corpus exceeds ``size_budget_bytes``.
    """
    pd = _require_pandas()
    counts = dict(counts or TIER_COUNTS)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    jobs: list[tuple[str, int, Config]] = []
    for tier, n in counts.items():
        cfg = load_config(f"{tier}.yaml")
        # Disjoint seed block per tier. A shared seed would put the *same*
        # scenario in two tiers under different labels, quietly corrupting any
        # tier-transfer experiment. Derived from the tier NAME rather than a
        # hardcoded table, so an extra tier (e.g. scan_on_scan) cannot silently
        # collide with an existing block.
        base = root_seed + seed_block(tier)
        jobs.extend((tier, base + i, cfg) for i in range(n))

    if verbose:
        print(f"[dataset] {len(jobs)} episodes, agents={list(agents)}, root={root}", flush=True)

    rows: list[dict[str, Any]] = []
    if n_jobs != 1:
        from joblib import Parallel, delayed

        rows = list(
            Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
                delayed(build_one_episode)(t, s, c, root, agents) for t, s, c in jobs
            )
        )
    else:
        for i, (tier, seed, cfg) in enumerate(jobs, 1):
            rows.append(build_one_episode(tier, seed, cfg, root, agents))
            if verbose and (i % 25 == 0 or i == len(jobs)):
                so_far = sum(int(r["bytes"]) for r in rows)
                print(
                    f"  {i}/{len(jobs)} episodes, {so_far / 1024**3:.2f} GB "
                    f"({100 * so_far / size_budget_bytes:.1f}% of budget)",
                    flush=True,
                )
            if sum(int(r["bytes"]) for r in rows) > size_budget_bytes:
                raise RuntimeError(
                    f"size budget exceeded after {i} episodes: "
                    f"{sum(int(r['bytes']) for r in rows) / 1024**3:.2f} GB > "
                    f"{size_budget_bytes / 1024**3:.0f} GB. Reduce `counts`, drop an agent "
                    f"from `agents`, or shorten `time.episode_s`."
                )

    index = pd.DataFrame(rows)
    index.to_parquet(root / "index.parquet", index=False, compression="zstd")

    total = int(index["bytes"].sum()) + (root / "index.parquet").stat().st_size
    if total > size_budget_bytes:
        raise RuntimeError(
            f"size budget exceeded: {total / 1024**3:.2f} GB > {size_budget_bytes / 1024**3:.0f} GB"
        )

    report = BuildReport(
        root=root,
        n_episodes=len(index),
        total_bytes=total,
        per_tier=index["tier"].value_counts().to_dict(),
        per_split=index["split"].value_counts().to_dict(),
        wall_time_s=time.perf_counter() - t0,
        fingerprint=source_fingerprint(),
    )
    write_dataset_card(root, index, report, agents)
    (root / "build_report.json").write_text(
        json.dumps(
            {
                "n_episodes": report.n_episodes,
                "total_bytes": report.total_bytes,
                "per_tier": report.per_tier,
                "per_split": report.per_split,
                "wall_time_s": report.wall_time_s,
                **report.fingerprint,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(report.summary(), flush=True)
    return report


def write_dataset_card(
    root: Path, index: Any, report: BuildReport, agents: Sequence[str]
) -> Path:
    """Write ``dataset_card.md``.

    Judges read this. It documents the schema, units, generation parameters,
    the simulator fingerprint and -- importantly -- the known limitations.

    Args:
        root: Dataset root.
        index: The index DataFrame.
        report: The build report.
        agents: Schedulers replayed into the observation traces.

    Returns:
        Path to the card.
    """
    from smartscan.data.schema import (
        INDEX_COLUMNS,
        MANIFEST_COLUMNS,
        OBSERVATION_COLUMNS,
        SPLITS,
        TRUTH_ARRAYS,
    )

    def table(columns: dict[str, str]) -> str:
        return "\n".join(f"| `{k}` | `{v}` |" for k, v in columns.items())

    tiers = ", ".join(f"{k}: {v}" for k, v in sorted(report.per_tier.items()))
    n_agents = len(agents)
    splits = ", ".join(f"{k}: {v}" for k, v in sorted(report.per_split.items()))
    fp = report.fingerprint

    card = f"""# EW Smart Scan — RF Environment Dataset

A reproducible corpus of simulated Electronic Support (ES) receiver scheduling
episodes: ground-truth spectrum occupancy over 0.5–18 GHz, the emitter order of
battle that produced it, and replayed receiver traces from {n_agents} schedulers
(three open-loop baselines plus four closed-loop policies).

Generated by **SmartScan** for **SIH 26055**
("Smart Scan Strategy for Electronic Warfare", DRDO / iDEX).

* **Dataset version** `{fp['dataset_version']}`
* **Simulator commit** `{fp['git_commit']}`
* **Source digest** `{fp['source_digest']}` (blake2b over every `.py` in the package)
* **Built** {fp['built_utc']}
* **Episodes** {report.n_episodes} ({tiers})
* **Splits** {splits}
* **Size** {report.gigabytes:.2f} GB
* **Licence** CC BY-SA 4.0

---

## What this is for

Two distinct uses:

1. **Sequence prediction.** Train a model to predict next-slot occupancy for
   *all* channels from an observation history that only ever covers 1/32 of the
   band. `observations.parquet` is the input; `truth_occupancy.npz` is the
   privileged label.
2. **Offline scheduler evaluation.** Replay any policy against fixed ground
   truth with a fixed detection realisation, so two policies face the same world
   *and* the same luck.

## What this is **not**

* It is **not** pulse-descriptor-word (PDW) data. Each cell is a channel-slot
  occupancy decision, not an individual pulse. For PDW-level work see the Turing
  Synthetic Radar Dataset (gated; we do not mirror it).
* It is **not** recorded off real hardware. See *Known limitations*.

---

## Layout

```
index.parquet                      one row per episode
dataset_card.md                    this file
build_report.json                  provenance and size accounting
episodes/<tier>/<episode_id>/
    truth_occupancy.npz            ground truth (see below)
    emitter_manifest.parquet       the order of battle
    observations.parquet           replayed dwell traces
```

`episode_id` is `<tier>_<seed>`, derived from content rather than a counter, so
adding episodes never renumbers existing ones.

---

## Schema

### `truth_occupancy.npz`

Occupancy is **bit-packed**; SNR and emitter id are stored **only at occupied
cells**, in raster order, because they are meaningless elsewhere. Unpack with
`smartscan.data.schema.unpack_occupancy`, or:

```python
import numpy as np
z = np.load("truth_occupancy.npz")
B, T = z["shape"]
occ = np.unpackbits(z["occupancy_packed"], count=B * T).astype(bool).reshape(B, T)
snr = np.full((B, T), -200.0, np.float32); snr[occ] = z["snr_db"]   # dB
eid = np.zeros((B, T), np.int16);          eid[occ] = z["emitter_id"]
```

| array | dtype |
|---|---|
{table(TRUTH_ARRAYS)}

### `index.parquet`

| column | dtype |
|---|---|
{table(INDEX_COLUMNS)}

### `emitter_manifest.parquet`

| column | dtype |
|---|---|
{table(MANIFEST_COLUMNS)}

### `observations.parquet`

One row per dwell, per replayed scheduler. `hit_mask` is a bitfield over the
`K` channels of the tuned window starting at `window_lo`; bit `i` corresponds to
channel `window_lo + i`.

| column | dtype |
|---|---|
{table(OBSERVATION_COLUMNS)}

`true_hit_mask` marks which declared hits were genuine rather than false alarms.
**It is ground truth and must not be used as a model input** — it is provided so
that offline evaluation can score a policy without re-simulating.

---

## Units

| quantity | unit |
|---|---|
| frequency | Hz |
| time | seconds, except `t`/`t_first_active`/`n_slots` which are **slots** |
| slot duration | `dt_s`, 1 ms by default |
| SNR, EIRP | dB, dBm |
| angles | degrees |
| threat priority | dimensionless, [0, 1] |

---

## Generation parameters

* Surveilled band 0.5–18 GHz, `B` = {int(index['n_channels'].iloc[0])} channels
* Receiver IBW `K` = 4 channels (K/B = 1/32)
* Slot `dt` = {float(index['dt_s'].iloc[0]) * 1e3:.0f} ms, episode = {int(index['n_slots'].iloc[0])} slots
* Eight emitter classes: FixedCW, PulsedRadar, CircularScanRadar, SectorScanRadar,
  FrequencyAgile, AgileBeamRadar, CommsBurst, Interferer
* Detection is probabilistic (square-law envelope detector, Swerling 0/I); `Pd`
  is never hard-coded to 1
* Replayed schedulers: {", ".join(agents)}

  Train a predictor on the **open-loop** traces (`sequential`, `random`,
  `priority_rr`): their coverage is close to uniform, so the training
  distribution is not biased toward the behaviour of the policy the model is
  meant to improve on. The closed-loop traces are for **offline policy
  evaluation** -- you cannot estimate the value of a policy you have no data
  from. The `agent` column makes that a one-line selection.

Full configuration is reproducible from `config_hash` in `index.parquet` against
the `configs/` directory of the source repository.

---

## Splits

Assigned **by scenario seed**, {", ".join(f"{k} {int(100 * v)}%" for k, v in SPLITS.items())},
by hashing the seed. Never by time slice: cutting one episode into a train half
and a test half would let a model memorise the emitters it is about to be scored
on, and the leakage would be invisible in every aggregate metric.

Because assignment is a hash of the seed, it is **stable as the corpus grows** —
adding episodes never moves an existing one between splits.

---

## Known limitations

1. **Synthetic.** Generated by a simulator, not recorded off a radio. Real
   captures carry non-Gaussian interference, spurs, LO drift and AGC transients
   that this corpus does not model. Results transfer as *relative* scheduler
   ordering more reliably than as absolute detection rates.
2. **Sea-level clear-air propagation only.** Atmospheric attenuation is a
   linear-in-frequency fit to ITU-R P.676; no rain, ducting, multipath or terrain.
3. **Scenarios are authored in SNR space.** Emitter range is back-solved from a
   target main-lobe SNR rather than sampled directly, so the SNR distribution is
   controlled by design. The link budget itself is physical, but the corpus is
   *not* a sample from any real engagement geometry.
4. **One receiver, one antenna, no AOA.** No direction finding, no multi-receiver
   fusion, no geolocation.
5. **No pulse-level detail.** Pulse trains are binned to per-slot counts and duty
   fractions; individual PDWs are not recoverable.
6. **Sector-scan periodicity is ambiguous.** A bidirectional sector scanner
   illuminates twice per frame at unequal spacing, so `scan_period_s` in the
   manifest is the *frame* period and a period estimator may legitimately
   recover half of it.
7. **Class balance is not uniform.** The tier mixes follow the problem brief, so
   FixedCW and PulsedRadar dominate by count. Use class-balanced sampling
   (provided in `smartscan.data.kaggle_io`) when this matters.

---

## Citation

```bibtex
@misc{{smartscan_rf_environment_2026,
  title  = {{EW Smart Scan: RF Environment Dataset for Receiver Scheduling}},
  note   = {{SIH 26055, DRDO/iDEX. Dataset version {fp['dataset_version']},
            simulator {fp['source_digest']}}},
  year   = {{2026}}
}}
```

## Licence

**CC BY-SA 4.0.** Share and adapt with attribution; derivatives under the same
licence.

The Turing Synthetic Radar Dataset referenced by the problem statement is a
**separate, gated** dataset with its own access conditions. It is **not**
mirrored here. Access it directly from Hugging Face under its own terms; this
repository provides only a runtime adapter
(`smartscan.data.tsrd_bridge`) that requires the user's own token.
"""
    path = root / "dataset_card.md"
    path.write_text(card, encoding="utf-8")
    return path
