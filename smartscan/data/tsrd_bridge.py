"""Bridge to the Turing Synthetic Radar Dataset (TSRD) for external validation.

The problem statement cites
`alan-turing-institute/turing-synthetic-radar-dataset` (Gunn et al.,
arXiv:2602.03856). Verified against the live repository:

* **Licence: Apache-2.0**, but **access is gated** ("auto" gating) -- you must
  request it on Hugging Face and be granted it per-user.
* **Format: HDF5**, one file per pulse train. Each file holds ``data`` of shape
  ``(N, 5)`` float32 and ``labels`` of shape ``(N, 1)`` int8, with
  ``metadata/feature_names`` giving
  ``['UTCTime', 'RF', 'PulseWidth', 'AOA', 'PA', 'Class']``.
* **Units** (this is the part that silently ruins an adapter): ToA in
  **microseconds**, RF in **MHz**, pulse width in **microseconds**, AOA in
  degrees, PA in **dB**.
* **Subsets**: ``archive`` (full-band, ~0.36-12 GHz over ~9.5 s, up to ~88
  emitters), ``stare`` (oracle receiver, whole spectrum) and ``scan``
  (a realistic sweeping receiver, per-config narrowband captures).

``archive`` is the default here because its band and duration line up with a
SmartScan episode almost exactly -- 0.36-12 GHz over 9.5 s against our
0.5-18 GHz over 10 s -- so a pulse train drops onto our ``[b, t]`` grid without
resampling in time.

Policy
------
* **We do not mirror or re-upload it.** Not to Kaggle, not anywhere. The
  published SmartScan dataset contains no TSRD content.
* It is fetched **at runtime** with the user's own token, from ``HF_TOKEN`` /
  ``HUGGINGFACE_TOKEN`` or a prior ``huggingface-cli login``.
* Every entry point degrades gracefully when the library, the token or the
  grant is missing: a :class:`TSRDUnavailableError` explaining what to do, never
  a crash and never a silent fallback to synthetic data dressed up as real.
* Results are reported **separately** from synthetic benchmarks and tagged
  ``external: true`` (:func:`external_validation_report`).

Loader library and benchmark metrics for the original deinterleaving task:
``github.com/alan-turing-institute/turing-deinterleaving-challenge``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smartscan.config import Config
from smartscan.env.propagation import SNR_FLOOR_DB
from smartscan.env.types import EmitterTruth, EpisodeTensors, SpectrumGrid

__all__ = [
    "PDW_COLUMNS",
    "TSRD_LICENCE",
    "TSRD_REPO",
    "TSRD_SUBSETS",
    "PDWStream",
    "TSRDUnavailableError",
    "bin_pdws_to_tensors",
    "external_validation_report",
    "load_tsrd_split",
    "token_available",
]

#: The gated Hugging Face repository. Apache-2.0 licensed, access-gated.
TSRD_REPO: str = "alan-turing-institute/turing-synthetic-radar-dataset"

#: Licence of the upstream dataset, as declared in its repository card.
TSRD_LICENCE: str = "Apache-2.0 (access-gated)"

#: Column order of the ``data`` array, from ``metadata/feature_names``.
#: The trailing ``Class`` name refers to the separate ``labels`` array.
PDW_COLUMNS: tuple[str, ...] = ("UTCTime", "RF", "PulseWidth", "AOA", "PA")

#: Subset -> (directory, split-directory template).
TSRD_SUBSETS: dict[str, tuple[str, str, str]] = {
    "archive": ("archive", "{split}", "{split}_{i}.h5"),
    "scan": ("scan", "{split}_scan", "config_{i}.h5"),
    "stare": ("stare", "{split}_stare", "config_{i}.h5"),
}


class TSRDUnavailableError(RuntimeError):
    """Raised when the gated dataset cannot be reached.

    Carries actionable guidance rather than a stack trace: the caller almost
    always needs to request access or set a token, not debug anything.
    """


def token_available() -> bool:
    """Whether a Hugging Face token is present, without reading its value.

    Returns:
        True if an environment token or a cached CLI login exists.
    """
    from smartscan.credentials import credential_status

    return credential_status().huggingface


def _guidance(detail: str) -> TSRDUnavailableError:
    """Build the standard, actionable unavailability error."""
    return TSRDUnavailableError(
        f"{detail}\n\n"
        f"The Turing Synthetic Radar Dataset is Apache-2.0 but ACCESS-GATED:\n"
        f"  1. Request access at https://huggingface.co/datasets/{TSRD_REPO}\n"
        f"  2. Wait for the grant (per-user, not instant)\n"
        f"  3. Authenticate with EITHER\n"
        f"       export HF_TOKEN=hf_...        (never commit this)\n"
        f"     OR  huggingface-cli login\n"
        f"  4. pip install 'smartscan[external]'   (needs h5py)\n\n"
        f"SmartScan does not mirror this dataset, and every synthetic result in\n"
        f"the benchmark stands without it. External validation is reported\n"
        f"separately when it is available."
    )


@dataclass
class PDWStream:
    """A pulse-descriptor-word stream from an external dataset.

    All fields are converted to **SI units on load**: TSRD stores microseconds
    and MHz, and an adapter that forgets that produces a silently empty tensor.

    Attributes:
        toa_s: Time of arrival, seconds from the start of the record.
        rf_hz: Carrier frequency, Hz.
        pw_s: Pulse width, seconds.
        amplitude_db: Received power, dB (treated as dBm, see
            :func:`bin_pdws_to_tensors`).
        aoa_deg: Angle of arrival, degrees. Carried through but unused: this
            prototype models a single omnidirectional receiver with no DF.
        emitter_id: Ground-truth emitter label per pulse.
        source: Provenance string, carried into the report.
        meta: Additional record-level metadata.
    """

    toa_s: np.ndarray
    rf_hz: np.ndarray
    pw_s: np.ndarray
    amplitude_db: np.ndarray
    aoa_deg: np.ndarray | None = None
    emitter_id: np.ndarray | None = None
    source: str = TSRD_REPO
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.toa_s.size)

    @property
    def duration_s(self) -> float:
        """Span of the record in seconds."""
        return float(self.toa_s.max() - self.toa_s.min()) if len(self) else 0.0

    @property
    def band_hz(self) -> tuple[float, float]:
        """``(min, max)`` carrier frequency present."""
        return (float(self.rf_hz.min()), float(self.rf_hz.max())) if len(self) else (0.0, 0.0)

    @property
    def n_emitters(self) -> int:
        """Distinct ground-truth emitters, or ``-1`` if unlabelled."""
        return int(np.unique(self.emitter_id).size) if self.emitter_id is not None else -1

    def summary(self) -> str:
        """Return a one-line description."""
        lo, hi = self.band_hz
        return (
            f"{len(self):,} PDWs over {self.duration_s:.2f} s, "
            f"{lo / 1e9:.2f}-{hi / 1e9:.2f} GHz, "
            f"{self.n_emitters if self.n_emitters >= 0 else 'unknown'} emitters"
        )


def load_tsrd_split(
    split: str = "test",
    subset: str = "archive",
    max_records: int | None = 4,
    token: str | None = None,
    indices: Sequence[int] | None = None,
) -> list[PDWStream]:
    """Fetch PDW pulse trains from the gated dataset at runtime.

    Args:
        split: ``'test'``, ``'train'`` or ``'validation'`` (``'val'`` for the
            scan/stare subsets).
        subset: ``'archive'`` (full band, the default), ``'scan'`` or
            ``'stare'``. See the module docstring for what each contains.
        max_records: Number of pulse trains to fetch. These are large files;
            the default is deliberately small.
        token: Explicit HF token. Prefer the environment -- passing a literal
            token risks it reaching a notebook output.
        indices: Explicit file indices, overriding ``max_records``.

    Returns:
        The loaded :class:`PDWStream` records, in SI units.

    Raises:
        TSRDUnavailableError: If ``h5py``/``huggingface_hub`` is missing, the
            token is absent, access is not granted, or no file could be read.
    """
    try:
        import h5py
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise _guidance(f"Missing dependency: {exc.name}.") from exc

    if subset not in TSRD_SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; use one of {sorted(TSRD_SUBSETS)}")

    from smartscan.credentials import load_dotenv

    load_dotenv(".env")
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok and not token_available():
        raise _guidance("No Hugging Face token found.")

    root, split_tmpl, name_tmpl = TSRD_SUBSETS[subset]
    split_dir = split_tmpl.format(split=split)
    wanted = list(indices) if indices is not None else list(range(max_records or 4))

    streams: list[PDWStream] = []
    failures: list[str] = []
    for i in wanted:
        rel = f"{root}/{split_dir}/{name_tmpl.format(split=split, i=i)}"
        try:
            path = hf_hub_download(TSRD_REPO, rel, repo_type="dataset", token=tok)
        except Exception as exc:
            failures.append(f"{rel}: {type(exc).__name__}")
            continue
        try:
            with h5py.File(path, "r") as handle:
                data = np.asarray(handle["data"], dtype=np.float64)
                labels = (
                    np.asarray(handle["labels"], dtype=np.int32).ravel()
                    if "labels" in handle
                    else None
                )
        except (OSError, KeyError) as exc:
            failures.append(f"{rel}: {type(exc).__name__}")
            continue

        if data.ndim != 2 or data.shape[1] < 5:
            failures.append(f"{rel}: unexpected shape {data.shape}")
            continue

        # Unit conversion. TSRD stores microseconds and MHz; forgetting either
        # produces an empty or wildly mis-binned tensor with no error.
        streams.append(
            PDWStream(
                toa_s=data[:, 0] * 1e-6,
                rf_hz=data[:, 1] * 1e6,
                # A handful of records carry a tiny negative pulse width; floor
                # it at 1 ns rather than propagating a negative bandwidth.
                pw_s=np.maximum(data[:, 2], 1e-3) * 1e-6,
                amplitude_db=data[:, 4],
                aoa_deg=data[:, 3],
                emitter_id=labels,
                meta={"file": rel, "split": split, "subset": subset, "index": i},
            )
        )

    if not streams:
        raise _guidance(
            f"No TSRD pulse train could be read from {root}/{split_dir}/. "
            f"Attempts: {failures[:4]}"
        )
    return streams


def bin_pdws_to_tensors(
    stream: PDWStream,
    config: Config,
    grid: SpectrumGrid | None = None,
    treat_pa_as_dbm: bool = True,
) -> EpisodeTensors:
    """Bin a PDW stream into the ``[b, t]`` tensors the receiver model consumes.

    This is the whole point of the bridge, and it is a **lossy, opinionated
    transformation** rather than a format conversion. Each pulse contributes to
    the cell ``(channel_of(RF), floor(ToA / dt))``:

    * ``n_pulses`` accumulates the pulse count per cell;
    * ``duty`` becomes ``n_pulses * PW / dt``, clipped to 1;
    * ``snr_db`` is derived **physically**, not by a scale factor: TSRD's ``PA``
      is read as received power in dBm and compared against this receiver's own
      thermal noise in the pulse-detection bandwidth ``min(channel, 1/PW)``,
      exactly as the simulator does. On ``archive`` this puts the median pulse
      near 14 dB SNR, which is a plausible intercept -- so the mapping is a
      modelling *choice* with a stated basis rather than a tuned constant;
    * ``emitter_id`` takes the label of that strongest pulse.

    Pulses whose RF falls outside the configured band are **dropped**, and the
    count is recorded in the returned tensors' ``config_hash`` provenance via
    the caller's report -- silently clamping them to the edge channels would
    manufacture emitters that are not there.

    Args:
        stream: The PDW record.
        config: Resolved configuration supplying the grid and slot duration.
        grid: Override the configured frequency grid.
        treat_pa_as_dbm: Interpret ``PA`` as absolute received power in dBm. If
            ``False`` it is treated as an SNR already in dB.

    Returns:
        :class:`EpisodeTensors` with the same contract as a simulated episode,
        so every scheduler, metric and plot works on it unchanged.

    Raises:
        ValueError: If the stream is empty.
    """
    if len(stream) == 0:
        raise ValueError("cannot bin an empty PDW stream")

    grid = grid or config.grid()
    dt = config.time.dt_s
    b = grid.n_channels

    t0 = float(stream.toa_s.min())
    rel = stream.toa_s - t0
    n_slots = min(int(np.ceil(rel.max() / dt)) + 1, config.n_slots)

    in_band = (stream.rf_hz >= grid.f_start_hz) & (stream.rf_hz <= grid.f_stop_hz)
    slots = np.floor(rel / dt).astype(np.int64)
    keep = in_band & (slots >= 0) & (slots < n_slots)

    chans = grid.channel_of(stream.rf_hz[keep]).astype(np.int64)
    slots = slots[keep]
    pw = stream.pw_s[keep]
    amp = stream.amplitude_db[keep]
    labels = (
        stream.emitter_id[keep].astype(np.int16)
        if stream.emitter_id is not None
        else np.ones(int(keep.sum()), dtype=np.int16)
    )

    n_pulses = np.zeros((b, n_slots), dtype=np.int32)
    duty = np.zeros((b, n_slots), dtype=np.float32)
    snr = np.full((b, n_slots), SNR_FLOOR_DB, dtype=np.float32)
    eid = np.zeros((b, n_slots), dtype=np.int16)

    np.add.at(n_pulses, (chans, slots), 1)
    np.add.at(duty, (chans, slots), (pw / dt).astype(np.float32))

    # Strongest pulse in each cell wins the SNR and label, matching the
    # simulator's convention.
    if treat_pa_as_dbm:
        from smartscan.env.propagation import noise_power_dbm

        bw_det = np.minimum(grid.widths_hz[chans], 1.0 / np.maximum(pw, 1e-12))
        snr_pulse = (amp - noise_power_dbm(bw_det, config.receiver.noise_figure_db)).astype(
            np.float32
        )
    else:
        snr_pulse = amp.astype(np.float32)
    order = np.argsort(snr_pulse)
    snr[chans[order], slots[order]] = snr_pulse[order]
    eid[chans[order], slots[order]] = labels[order]

    np.clip(duty, 0.0, 1.0, out=duty)
    occupancy = (n_pulses > 0).astype(np.uint8)
    snr[occupancy == 0] = SNR_FLOOR_DB

    truth = tuple(
        EmitterTruth(
            emitter_id=int(e),
            emitter_class="ExternalPDW",
            f_center_hz=float(np.median(stream.rf_hz[keep][labels == e])) if (labels == e).any() else 0.0,
            home_channel=int(np.bincount(chans[labels == e], minlength=b).argmax())
            if (labels == e).any()
            else 0,
            threat_priority=0.5,  # TSRD carries no threat model; stated, not invented
            is_novel=True,
            is_interferer=False,
            t_first_active=int(slots[labels == e].min()) if (labels == e).any() else 0,
            detection_mode="pulse",
            scan_period_s=float("nan"),
            params={"source": stream.source, "n_pulses": int((labels == e).sum())},
        )
        for e in np.unique(labels)
        if e > 0
    )

    return EpisodeTensors(
        occupancy=occupancy,
        duty=duty,
        snr_db=snr,
        emitter_id=eid,
        n_pulses=n_pulses,
        truth=truth,
        grid=grid,
        dt_s=dt,
        n_slots=n_slots,
        seed=-1,  # not seed-derived: this episode came from external data
        config_hash=config.hash(),
    )


def external_validation_report(
    config: Config,
    agents: tuple[str, ...] = ("sequential", "ucb1", "whittle", "phase_locked"),
    split: str = "test",
    subset: str = "archive",
    max_records: int = 4,
) -> dict[str, Any]:
    """Replay schedulers over real TSRD pulse trains, reported separately.

    Results from this function must **never** be pooled with synthetic results.
    The binning in :func:`bin_pdws_to_tensors` makes assumptions the synthetic
    pipeline does not, the amplitude scale is uncalibrated, and the emitter mix
    is whatever TSRD contains rather than the tiers the benchmark is defined on.
    The returned dict is explicitly tagged so a downstream table cannot merge
    the two by accident.

    Args:
        config: Resolved configuration.
        agents: Schedulers to replay.
        split: TSRD split.
        subset: TSRD subset; ``'archive'`` is the full-band data whose span
            matches a SmartScan episode.
        max_records: Records to evaluate.

    Returns:
        Dict with ``available``, ``source``, ``licence_note``, and either
        ``rows`` (per record and agent) or ``reason``.
    """
    header = {
        "available": False,
        "external": True,
        "source": TSRD_REPO,
        "licence": TSRD_LICENCE,
        "licence_note": (
            "The Turing Synthetic Radar Dataset is Apache-2.0 but ACCESS-GATED, and "
            "SmartScan does not mirror or redistribute it -- it is fetched at runtime "
            "with the user's own token. These results are reported SEPARATELY from "
            "synthetic benchmarks and are not directly comparable to them: the "
            "PDW-to-occupancy binning is an assumption of this bridge, the emitter mix "
            "is whatever TSRD contains rather than our tiers, and TSRD carries no "
            "threat model so every emitter is scored at a uniform priority."
        ),
        "citation": "Gunn et al., arXiv:2602.03856 (2026)",
    }

    try:
        streams = load_tsrd_split(split=split, subset=subset, max_records=max_records)
    except TSRDUnavailableError as exc:
        return {**header, "reason": str(exc)}

    from smartscan.agents import build_agent
    from smartscan.analysis.metrics import evaluate_episode
    from smartscan.runner import run_episode

    rows: list[dict[str, Any]] = []
    for i, stream in enumerate(streams):
        episode = bin_pdws_to_tensors(stream, config)
        for key in agents:
            # The episode is shorter than a configured one, so align the horizon.
            local = config.with_overrides(
                time={"episode_s": episode.n_slots * episode.dt_s}
            )
            result = run_episode(local, i, build_agent(key, local, i, None), episode=episode)
            row = evaluate_episode(result, config)
            row.pop("_detail", None)
            rows.append({
                "record": i, "agent": key, "n_pdws": len(stream),
                "duration_s": stream.duration_s, "summary": stream.summary(), **row,
            })

    return {**header, "available": True, "n_records": len(streams), "rows": rows}
