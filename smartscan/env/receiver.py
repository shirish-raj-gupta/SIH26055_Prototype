"""The ES receiver: a narrow window onto a wide band, with a cost to move it.

At slot ``t`` the receiver observes channels ``[lo, lo + K)`` and **nothing
else**. Everything outside is unobserved -- not zero, unknown. With the defaults
``K = 4`` of ``B = 128``, 31/32 of the surveilled band is dark at every instant.

The economics of the whole problem live in one line: changing the tuned centre
costs ``t_settle`` slots in which nothing at all is observed, while staying costs
none. A greedy hopper pays three slots per look where a parked receiver pays one.
"""

from __future__ import annotations

import numpy as np

from smartscan.config import Config
from smartscan.env.types import EpisodeTensors, Observation
from smartscan.hal.simulated import SimulatedBackend

__all__ = ["Receiver"]


class Receiver:
    """Channel-indexed receiver over a HAL backend.

    Args:
        episode: Ground-truth tensors for the episode.
        config: Resolved configuration.
        backend: HAL backend; a :class:`SimulatedBackend` is built if omitted.
        seed: Seed for the detection realisation, forwarded to the backend.

    Raises:
        NotImplementedError: If ``config.receiver.backend == "soapy"``.
    """

    def __init__(
        self,
        episode: EpisodeTensors,
        config: Config,
        backend: SimulatedBackend | None = None,
        seed: int | None = None,
    ) -> None:
        if backend is None and config.receiver.backend == "soapy":
            from smartscan.hal.soapy_stub import SoapySDRBackend

            SoapySDRBackend()  # raises NotImplementedError with a pointer to the roadmap

        self.cfg = config
        self.episode = episode
        self.backend = backend or SimulatedBackend(episode, config, seed=seed)
        self.k = config.receiver.ibw_channels
        self.n_channels = config.n_channels
        self.n_slots = episode.n_slots
        self.t_settle = config.receiver.t_settle_slots

        self._legal = self._compute_legal_mask()
        self.reset()

    # -- action geometry --------------------------------------------------- #
    def _compute_legal_mask(self) -> np.ndarray:
        """Return the boolean mask of legal actions.

        Windows that would fall off a band edge are **masked, not clipped**.
        Clipping would silently over-sample the two band edges and quietly
        distort every coverage metric.
        """
        mask = np.zeros(self.n_channels, dtype=bool)
        for a in range(self.n_channels):
            lo = self._window_start(a)
            if 0 <= lo <= self.n_channels - self.k:
                mask[a] = True
        return mask

    def _window_start(self, action: int) -> int:
        """Return the first channel of the window an action selects."""
        if self.cfg.receiver.action_space == "window_start":
            return int(action)
        return int(action) - (self.k - 1) // 2

    def legal_actions(self) -> np.ndarray:
        """Return the boolean legal-action mask, shape ``(B,)``.

        With ``K = 4`` and ``B = 128`` this leaves 125 of 128 actions legal.
        """
        return self._legal.copy()

    def window_of(self, action: int) -> tuple[int, int]:
        """Return the ``[lo, hi)`` channel window an action observes.

        Args:
            action: Action index.

        Returns:
            Half-open channel window.

        Raises:
            ValueError: If the action is illegal and masking is enabled.
        """
        lo = self._window_start(action)
        if not (0 <= lo <= self.n_channels - self.k):
            if self.cfg.receiver.mask_illegal_actions:
                raise ValueError(
                    f"action {action} is illegal (window [{lo}, {lo + self.k}) "
                    f"falls outside [0, {self.n_channels})); use legal_actions()"
                )
            lo = int(np.clip(lo, 0, self.n_channels - self.k))
        return lo, lo + self.k

    def center_hz_of(self, action: int) -> float:
        """Return the centre frequency an action tunes to, in Hz."""
        lo, hi = self.window_of(action)
        edges = self.episode.grid.edges_hz
        return float(0.5 * (edges[lo] + edges[hi]))

    # -- episode ----------------------------------------------------------- #
    def reset(self) -> None:
        """Restart the episode clock and forget the tuned frequency."""
        self.t = 0
        self.last_action: int | None = None
        self.n_retunes = 0
        self.settle_slots_lost = 0

    @property
    def done(self) -> bool:
        """Whether the episode horizon has been reached."""
        return self.t >= self.n_slots

    def step(self, action: int) -> Observation:
        """Tune, dwell for one slot, and report.

        Args:
            action: Centre-channel index (or window start, per ``action_space``).

        Returns:
            The :class:`Observation` for the dwell.

        Raises:
            ValueError: If the action is illegal and masking is enabled.
            RuntimeError: If the episode is already finished.
        """
        if self.done:
            raise RuntimeError("episode is finished; call reset()")

        lo, hi = self.window_of(action)
        retuned = self.last_action is None or self._window_start(action) != self._window_start(
            self.last_action
        )
        settle = self.t_settle if retuned else 0
        if retuned:
            self.n_retunes += 1
            self.settle_slots_lost += settle
            self.backend.tune(self.center_hz_of(action))

        # Settling burns slots in which nothing is observed; the dwell is the
        # slot that follows.
        t_dwell = min(self.t + settle, self.n_slots - 1)
        self.backend.seek(t_dwell)
        # Advances the backend clock; the simulated backend exposes its
        # detection realisation directly, so the handle itself is unused here.
        self.backend.capture(self.cfg.time.dt_s)

        hits = self.backend.declared[lo:hi, t_dwell].copy()
        pfa_flags = self.backend.false_alarm[lo:hi, t_dwell].copy()
        snr_est = np.where(
            hits,
            self.backend.snr_report[lo:hi, t_dwell],
            np.float32(np.nan) if not self.cfg.receiver.detector.report_snr_on_miss else np.float32(0.0),
        ).astype(np.float32)
        truth_ids = self.episode.emitter_id[lo:hi, t_dwell].copy()

        elapsed = settle + 1
        self.t = t_dwell + 1
        self.last_action = int(action)
        return Observation(
            window=(lo, hi),
            hits=hits,
            snr_est_db=snr_est,
            pfa_flags=pfa_flags,
            slots_elapsed=elapsed,
            t=t_dwell,
            truth_ids=truth_ids,
        )

    # -- introspection ----------------------------------------------------- #
    @property
    def ibw_hz(self) -> float:
        """Instantaneous bandwidth in Hz."""
        return self.backend.ibw_hz

    @property
    def ibw_ratio(self) -> float:
        """IBW as a fraction of the surveilled band -- the ``K/B`` ratio."""
        return self.k / self.n_channels
