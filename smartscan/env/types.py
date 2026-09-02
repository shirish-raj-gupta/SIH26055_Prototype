"""Frozen data contracts shared across the whole package.

This module has **no internal dependencies**. Everything else may import it; it
imports nothing from ``smartscan``. That keeps the dependency graph acyclic and
lets ``agents/`` consume observations without ever being able to reach ground
truth (see ``docs/architecture.md`` §3).

Shapes and dtypes declared here are contracts, asserted in
``tests/test_contracts.py``. Silent ``float64`` promotion is a real source of
both slowness and non-determinism, so dtypes are pinned explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "Activity",
    "CaptureHandle",
    "Detection",
    "DetectionMode",
    "EmitterTruth",
    "EpisodeTensors",
    "Observation",
    "SpectrumGrid",
]

DetectionMode = Literal["energy", "pulse"]

# Emitter id 0 is reserved for "noise / no emitter" in the label tensor E[b, t].
NOISE_ID: int = 0


@dataclass(frozen=True, slots=True)
class SpectrumGrid:
    """A partition of the surveilled band into ``n_channels`` contiguous channels.

    Supports non-uniform partitioning (a PS requirement): the grid is defined by
    its edges, not by a constant width, so ``log`` and ``explicit`` partitions
    cost nothing extra downstream.

    Attributes:
        edges_hz: Shape ``(B + 1,)`` float64, strictly increasing band edges.
        centers_hz: Shape ``(B,)`` float64 channel centre frequencies.
        widths_hz: Shape ``(B,)`` float64 channel widths.
    """

    edges_hz: np.ndarray
    centers_hz: np.ndarray
    widths_hz: np.ndarray

    @property
    def n_channels(self) -> int:
        """Number of channels ``B``."""
        return int(self.centers_hz.size)

    @property
    def f_start_hz(self) -> float:
        """Lower edge of the surveilled band."""
        return float(self.edges_hz[0])

    @property
    def f_stop_hz(self) -> float:
        """Upper edge of the surveilled band."""
        return float(self.edges_hz[-1])

    def channel_of(self, f_hz: float | np.ndarray) -> np.ndarray:
        """Map frequency to channel index.

        Args:
            f_hz: Frequency or array of frequencies in Hz.

        Returns:
            Integer channel indices, clipped to ``[0, B - 1]``. Frequencies
            outside the band clamp to the nearest edge channel rather than
            raising, because emitters are always placed in-band by construction
            and clamping keeps the hot path branch-free.
        """
        idx = np.searchsorted(self.edges_hz, np.atleast_1d(f_hz), side="right") - 1
        return np.clip(idx, 0, self.n_channels - 1).astype(np.int32)

    def straddle_fraction(self, f_hz: float | np.ndarray) -> np.ndarray:
        """Normalised distance from the nearest channel edge, in ``[0, 1]``.

        ``0`` means the tone sits exactly on a channel edge (worst scalloping),
        ``1`` means it sits at the channel centre (no loss).

        Args:
            f_hz: Frequency or array of frequencies in Hz.

        Returns:
            Float32 array of the same shape as ``f_hz``.
        """
        f = np.atleast_1d(np.asarray(f_hz, dtype=np.float64))
        b = self.channel_of(f)
        lo, hi = self.edges_hz[b], self.edges_hz[b + 1]
        # Triangular weight: 0 at either edge, 1 at the centre.
        frac = 1.0 - 2.0 * np.abs((f - lo) / (hi - lo) - 0.5)
        return np.clip(frac, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Activity:
    """One emitter's instantaneous emission, vectorised over slots.

    Emitted by ``BaseEmitter.activity(t)``. The three arrays are parallel: entry
    ``i`` says that at slot ``slots[i]`` the emitter placed ``duty[i]`` of a slot
    of energy into channel ``channels[i]``, with antenna gain toward us of
    ``gain_db[i]``.

    A single emitter can contribute several entries for the same slot (a
    frequency-agile emitter hopping 10x inside one 1 ms slot), which is exactly
    why this is a sparse triplet rather than a dense ``(B, T)`` block.

    Attributes:
        slots: Shape ``(M,)`` int32 slot indices.
        channels: Shape ``(M,)`` int32 channel indices.
        duty: Shape ``(M,)`` float32 in ``[0, 1]``, fraction of the slot filled.
        gain_db: Shape ``(M,)`` float32, emitter antenna gain toward the receiver
            (main lobe, sidelobe or backlobe) relative to isotropic.
        n_pulses: Shape ``(M,)`` int32, pulses landing in the slot. ``0`` for
            ``energy``-mode emitters, which are integrated rather than counted.
    """

    slots: np.ndarray
    channels: np.ndarray
    duty: np.ndarray
    gain_db: np.ndarray
    n_pulses: np.ndarray

    @staticmethod
    def empty() -> Activity:
        """Return an activity with no emissions (used by inactive emitters)."""
        z_i = np.zeros(0, dtype=np.int32)
        z_f = np.zeros(0, dtype=np.float32)
        return Activity(z_i, z_i.copy(), z_f, z_f.copy(), z_i.copy())

    def __len__(self) -> int:
        return int(self.slots.size)


@dataclass(frozen=True, slots=True)
class EmitterTruth:
    """Ground-truth record for one emitter, used by evaluation only.

    Attributes:
        emitter_id: Unique id, ``>= 1`` (0 is reserved for noise).
        emitter_class: Class name, e.g. ``"CircularScanRadar"``.
        f_center_hz: Nominal centre frequency.
        home_channel: Channel index of ``f_center_hz``.
        threat_priority: Operator-assigned value in ``[0, 1]``.
        is_novel: Whether the emitter is unknown to pre-mission intelligence.
        is_interferer: Whether the emitter is a low-value, high-duty distractor.
        t_first_active: First slot at which the emitter may transmit (``> 0.6T``
            for pop-ups).
        detection_mode: ``"energy"`` or ``"pulse"``.
        scan_period_s: Mechanical/electronic scan period, or ``nan`` if the
            emitter is not periodic in scan.
        params: Free-form dict of the sampled class-specific parameters, carried
            into ``emitter_manifest.parquet`` by the dataset builder.
    """

    emitter_id: int
    emitter_class: str
    f_center_hz: float
    home_channel: int
    threat_priority: float
    is_novel: bool
    is_interferer: bool
    t_first_active: int
    detection_mode: DetectionMode
    scan_period_s: float
    params: dict[str, float | int | str | tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeTensors:
    """Hold the ground truth for one episode; never visible to a scheduler.

    Attributes:
        occupancy: ``X`` shape ``(B, T)`` uint8, 1 if any emitter energy present.
        duty: ``d`` shape ``(B, T)`` float32 in ``[0, 1]``, sub-slot occupied
            fraction summed over emitters and clipped.
        snr_db: ``SNR`` shape ``(B, T)`` float32, peak SNR of the strongest
            emitter in the cell (``-inf`` sentinel replaced by ``SNR_FLOOR_DB``).
        emitter_id: ``E`` shape ``(B, T)`` int16, id of the strongest emitter,
            ``0`` where only noise.
        n_pulses: shape ``(B, T)`` int32, pulses from the strongest emitter.
        truth: Per-emitter ground-truth records.
        grid: The frequency partition.
        dt_s: Slot duration in seconds.
        n_slots: ``T``.
        seed: Scenario seed that produced this episode.
        config_hash: blake2b of the resolved config that produced this episode.
    """

    occupancy: np.ndarray
    duty: np.ndarray
    snr_db: np.ndarray
    emitter_id: np.ndarray
    n_pulses: np.ndarray
    truth: tuple[EmitterTruth, ...]
    grid: SpectrumGrid
    dt_s: float
    n_slots: int
    seed: int
    config_hash: str

    @property
    def n_channels(self) -> int:
        """Number of channels ``B``."""
        return self.grid.n_channels

    def digest(self) -> str:
        """Return a blake2b digest over every ground-truth tensor.

        Used by ``tests/test_reproducibility.py`` to assert that a given seed
        reproduces byte-identical tensors (acceptance test 1). The arrays are
        hashed in a fixed order with their dtypes pinned, so a dtype change is
        also a digest change -- which is what we want.

        Returns:
            32-character hex digest.
        """
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.occupancy, self.duty, self.snr_db, self.emitter_id, self.n_pulses):
            h.update(np.ascontiguousarray(arr).tobytes())
            h.update(str(arr.dtype).encode())
        return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Observation:
    """What the receiver reports for one dwell. The **only** scheduler input.

    Attributes:
        window: ``[lo, hi)`` channel indices actually observed this dwell.
        hits: Shape ``(K,)`` bool, detection declared (may be a false alarm).
        snr_est_db: Shape ``(K,)`` float32, estimated SNR; ``nan`` where no hit.
        pfa_flags: Shape ``(K,)`` bool, hit arose from noise alone.
            **Evaluation only** -- schedulers must not read this, and
            ``BeliefState`` never does.
        slots_elapsed: Wall-clock slots consumed: ``1 + t_settle`` on a retune,
            else ``1``.
        t: Slot index at which the dwell *ended*.
        truth_ids: Shape ``(K,)`` int16 ground-truth emitter ids in the window.
            **Evaluation only.**
    """

    window: tuple[int, int]
    hits: np.ndarray
    snr_est_db: np.ndarray
    pfa_flags: np.ndarray
    slots_elapsed: int
    t: int
    truth_ids: np.ndarray

    @property
    def channels(self) -> np.ndarray:
        """Return the absolute channel indices covered by this dwell."""
        return np.arange(self.window[0], self.window[1], dtype=np.int32)


@dataclass(frozen=True, slots=True)
class Detection:
    """A single declared detection, in HAL units (Hz and seconds).

    This is the currency of ``ReceiverBackend.get_detections`` and is what a real
    SDR backend would emit from its CFAR stage. Keeping the HAL in physical
    units rather than channel indices is what lets a real radio drop in without
    touching scheduler code.

    Attributes:
        f_hz: Centre frequency of the detection.
        t_s: Time of detection, seconds from episode start.
        snr_db: Estimated SNR.
        bandwidth_hz: Resolution bandwidth of the detecting bin.
    """

    f_hz: float
    t_s: float
    snr_db: float
    bandwidth_hz: float


@dataclass(frozen=True, slots=True)
class CaptureHandle:
    """Opaque handle to one capture, returned by ``ReceiverBackend.capture``.

    For ``SimulatedBackend`` this is just a slot range. For a real SDR it would
    wrap an IQ buffer. Schedulers never see it.

    Attributes:
        t_start: First slot of the capture.
        t_stop: One past the last slot of the capture.
        center_hz: Tuned centre frequency.
        payload: Backend-private data (``None`` for the simulator).
    """

    t_start: int
    t_stop: int
    center_hz: float
    payload: object | None = None
