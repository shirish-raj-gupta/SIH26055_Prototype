"""Simulated receiver backend over precomputed ground truth.

**Common random numbers.** The entire detection realisation -- which true
signals would have been detected, and where noise alone would have raised a
false alarm -- is drawn *once per episode*, before any scheduler runs, from the
``("receiver",)`` RNG substream. Two consequences, both deliberate:

1. Replays are bit-identical regardless of the path the scheduler took.
2. Two schedulers on the same seed face the *same* luck. Paired comparisons
   (``eval.paired``) therefore measure policy quality, not detection noise,
   which is what makes a 25 % claim on 30 seeds defensible rather than lucky.

The precomputation is also what makes the simulator fast: stepping is an array
slice, not a call into ``scipy.stats``.
"""

from __future__ import annotations

import numpy as np

from smartscan.config import Config
from smartscan.env.propagation import SNR_FLOOR_DB, p_detect
from smartscan.env.types import CaptureHandle, Detection, EpisodeTensors
from smartscan.hal.backend import ReceiverBackend
from smartscan.seeding import SeedTree

__all__ = ["SimulatedBackend", "detection_probability_tensor"]


def detection_probability_tensor(episode: EpisodeTensors, config: Config) -> np.ndarray:
    """Compute ``Pd[b, t]`` for every ground-truth cell, vectorised.

    The detection regime is read off the ground truth rather than configured:
    a cell with ``n_pulses > 0`` came from a pulsed emitter and is detected
    per-pulse; a cell with ``n_pulses == 0`` came from a continuous emitter and
    is detected by energy integration. See the ``propagation`` module docstring.

    Args:
        episode: Ground-truth tensors.
        config: Resolved configuration.

    Returns:
        Float32 array of shape ``(B, T)`` with ``Pd`` in ``[0, 1]``; zero where
        no emitter is present.
    """
    det = config.receiver.detector
    occupied = episode.occupancy.astype(bool)
    pd = np.zeros(episode.occupancy.shape, dtype=np.float32)
    if not np.any(occupied):
        return pd

    snr = episode.snr_db[occupied].astype(np.float64)
    npul = episode.n_pulses[occupied]
    is_pulse = npul > 0

    # Pulse regime: independent single-pulse opportunities, 1-of-n combination.
    if np.any(is_pulse):
        pd_single = p_detect(snr[is_pulse], n_integrate=1, pfa=det.pfa, swerling=det.swerling)
        pd_cells = np.zeros(snr.size)
        pd_cells[is_pulse] = 1.0 - np.power(1.0 - pd_single, npul[is_pulse].astype(np.float64))
    else:
        pd_cells = np.zeros(snr.size)

    # Energy regime: N averaged periodograms over the dwell. N depends only on
    # channel width, so non-uniform grids are handled by grouping unique widths.
    if np.any(~is_pulse):
        widths = episode.grid.widths_hz
        chan_of_cell = np.nonzero(occupied)[0][~is_pulse]
        bin_bw = widths / det.fft_size
        n_avg = np.clip(
            np.round(config.time.dt_s * bin_bw).astype(int), 1, det.n_integrate_max
        )
        if det.n_integrate != "auto":
            n_avg = np.full_like(n_avg, int(det.n_integrate))
        snr_e = snr[~is_pulse]
        out = np.zeros(snr_e.size)
        for n_val in np.unique(n_avg[chan_of_cell]):
            sel = n_avg[chan_of_cell] == n_val
            out[sel] = p_detect(
                snr_e[sel], n_integrate=int(n_val), pfa=det.pfa, swerling=det.swerling
            )
        pd_cells[~is_pulse] = out

    pd[occupied] = pd_cells.astype(np.float32)
    # A cell below tangential sensitivity carries the SNR floor sentinel: the
    # emitter is physically present but the front end cannot register it at all.
    # Pd is zero there, not Pfa -- otherwise an undetectable emitter would appear
    # to be "detected" at the noise-only crossing rate and would corrupt both the
    # interception ratio and the false-alarm accounting.
    pd[episode.snr_db <= SNR_FLOOR_DB] = 0.0
    return pd


class SimulatedBackend(ReceiverBackend):
    """Backend that replays precomputed ground truth with realistic detection.

    Args:
        episode: Ground-truth tensors for the episode.
        config: Resolved configuration.
        seed: Seed for the detection realisation. Defaults to the episode seed,
            so the environment's luck is tied to the scenario, not the agent.
    """

    def __init__(self, episode: EpisodeTensors, config: Config, seed: int | None = None) -> None:
        self.episode = episode
        self.cfg = config
        self.grid = episode.grid
        self._center_hz = float(self.grid.centers_hz[0])
        self._t = 0

        rng = SeedTree(int(seed if seed is not None else episode.seed)).rng("receiver")
        det = config.receiver.detector

        #: Pd for every ground-truth cell.
        self.pd = detection_probability_tensor(episode, config)

        # -- common random numbers: draw the whole realisation up front ----- #
        shape = episode.occupancy.shape
        self.true_hit = rng.random(shape) < self.pd
        self.false_alarm = (rng.random(shape) < det.pfa) & (episode.occupancy == 0)
        #: Declared detections: the agent cannot tell these apart.
        self.declared = self.true_hit | self.false_alarm

        # Reported SNR carries estimation noise; false alarms report a plausible
        # just-above-threshold value rather than the truth sentinel.
        noise = rng.normal(0.0, det.snr_est_sigma_db, size=shape).astype(np.float32)
        fa_level = rng.uniform(0.0, 6.0, size=shape).astype(np.float32)
        self.snr_report = np.where(
            episode.occupancy.astype(bool), episode.snr_db + noise, fa_level
        ).astype(np.float32)

    # -- capabilities ------------------------------------------------------ #
    @property
    def ibw_hz(self) -> float:
        """Instantaneous bandwidth spanned by ``ibw_channels`` channels."""
        return float(self.grid.widths_hz[: self.cfg.receiver.ibw_channels].sum())

    @property
    def tune_range_hz(self) -> tuple[float, float]:
        """Tunable centre-frequency range."""
        return (float(self.grid.f_start_hz), float(self.grid.f_stop_hz))

    @property
    def settle_time_s(self) -> float:
        """Configured LO settling time."""
        return self.cfg.receiver.t_settle_slots * self.cfg.time.dt_s

    @property
    def noise_figure_db(self) -> float:
        """Configured noise figure."""
        return self.cfg.receiver.noise_figure_db

    # -- operation --------------------------------------------------------- #
    def tune(self, center_hz: float) -> None:
        """Retune the simulated front end.

        Args:
            center_hz: Requested centre frequency in Hz.

        Raises:
            ValueError: If outside :attr:`tune_range_hz`.
        """
        lo, hi = self.tune_range_hz
        if not (lo <= center_hz <= hi):
            raise ValueError(f"centre {center_hz / 1e9:.3f} GHz outside tune range")
        self._center_hz = float(center_hz)

    def capture(self, duration_s: float) -> CaptureHandle:
        """Advance simulated time and return a handle to the dwell.

        Args:
            duration_s: Dwell duration in seconds.

        Returns:
            A :class:`CaptureHandle` covering the dwell's slots.
        """
        n = max(int(round(duration_s / self.cfg.time.dt_s)), 1)
        start = self._t
        self._t = min(self._t + n, self.episode.n_slots)
        return CaptureHandle(t_start=start, t_stop=self._t, center_hz=self._center_hz)

    def get_detections(self, capture: CaptureHandle) -> list[Detection]:
        """Return declared detections inside a capture.

        Args:
            capture: Handle from :meth:`capture`.

        Returns:
            Declared detections, false alarms included and indistinguishable.
        """
        lo, hi = self.window_for(capture.center_hz)
        t = min(max(capture.t_stop - 1, 0), self.episode.n_slots - 1)
        out: list[Detection] = []
        for b in range(lo, hi):
            if self.declared[b, t]:
                out.append(
                    Detection(
                        f_hz=float(self.grid.centers_hz[b]),
                        t_s=t * self.cfg.time.dt_s,
                        snr_db=float(self.snr_report[b, t]),
                        bandwidth_hz=float(self.grid.widths_hz[b]),
                    )
                )
        return out

    # -- helpers ----------------------------------------------------------- #
    def window_for(self, center_hz: float) -> tuple[int, int]:
        """Return the ``[lo, hi)`` channel window covered by a centre frequency."""
        k = self.cfg.receiver.ibw_channels
        c = int(self.grid.channel_of(center_hz)[0])
        lo = c - (k - 1) // 2
        lo = int(np.clip(lo, 0, self.grid.n_channels - k))
        return lo, lo + k

    def seek(self, t: int) -> None:
        """Set the simulated clock (used by the environment, not by schedulers)."""
        self._t = int(np.clip(t, 0, self.episode.n_slots))

    @property
    def t(self) -> int:
        """Current simulated slot."""
        return self._t
