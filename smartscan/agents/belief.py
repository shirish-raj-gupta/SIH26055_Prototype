"""Shared belief state over channel occupancy.

Because the receiver is blind outside its IBW, every scheduler -- from the
sweep baseline to the RL agent -- reasons over the same belief object. It is
built **only** from :class:`~smartscan.env.types.Observation`; it never touches
ground truth, and it deliberately does not read ``Observation.pfa_flags``, which
exists for evaluation alone.

Two design points carry weight:

**Decay toward the prior.** Unvisited channels relax back to ``Beta(a0, b0)``
with a configurable half-life. This is discounted-Bayesian tracking (Garivier &
Moulines, discounted UCB; Raj & Kalyani, discounted Thompson sampling) and is the
principled way to encode *"what I learned 4 s ago about a scanning radar is
nearly worthless"*. Without it every bandit policy converges to a stale answer
and stops revisiting.

**Phase as sin/cos.** The predicted phase to the next beam arrival enters the
feature vector as a ``(sin, cos)`` pair rather than a scalar, so a network sees
no discontinuity at the wrap. That is what makes phase-locking learnable rather
than only coverage.
"""

from __future__ import annotations

import numpy as np

from smartscan.config import Config
from smartscan.env.types import Observation

__all__ = ["FEATURE_NAMES", "N_CHANNEL_FEATURES", "N_GLOBAL_FEATURES", "BeliefState"]

#: Per-channel feature count. Surfaced in config only so a mismatch fails at load
#: rather than as a shape error 40 minutes into training.
N_CHANNEL_FEATURES: int = 12

#: Global (non-per-channel) feature count.
N_GLOBAL_FEATURES: int = 5

FEATURE_NAMES: tuple[str, ...] = (
    "p_occupied",
    "posterior_std",
    "log_alpha",
    "log_beta",
    "log_time_since_visit",
    "log_time_since_hit",
    "ewma_activity",
    "ewma_snr",
    "period_confidence",
    "phase_sin",
    "phase_cos",
    "interferer_score",
)

#: Maximum hit timestamps retained per channel for period estimation. Bounded so
#: memory and estimator cost stay O(1) per channel over arbitrarily long episodes.
_MAX_HIT_HISTORY: int = 96

#: Maximum visit timestamps retained per channel, used for spectral-window
#: deconvolution. Larger than the hit history because empty looks are the
#: majority and they are what pins down the sampling schedule.
_MAX_VISIT_HISTORY: int = 512

#: Period-grid resolution used for ONLINE re-estimation inside an episode. The
#: configured (much finer) grid is reserved for offline analysis: online we only
#: need enough resolution to phase-lock, and a 4000-bin periodogram per channel
#: per refresh would cost more than the entire rest of the episode.
_ONLINE_PERIOD_BINS: int = 512


class BeliefState:
    """Per-channel Beta posterior plus staleness and periodicity tracking.

    Args:
        config: Resolved configuration.
        n_slots: Episode length, used to normalise staleness features.
    """

    def __init__(self, config: Config, n_slots: int | None = None) -> None:
        self.cfg = config
        self.n_channels = config.n_channels
        self.n_slots = int(n_slots if n_slots is not None else config.n_slots)
        bc = config.belief

        self.alpha_prior = float(bc.alpha_prior)
        self.beta_prior = float(bc.beta_prior)
        #: Per-slot decay factor toward the prior.
        self.rho = float(0.5 ** (1.0 / bc.decay_half_life_slots))
        self._log_t = float(np.log(max(self.n_slots, 2)))

        b = self.n_channels
        self.alpha = np.full(b, self.alpha_prior, dtype=np.float64)
        self.beta = np.full(b, self.beta_prior, dtype=np.float64)
        self.time_since_visit = np.full(b, self.n_slots, dtype=np.float64)
        self.time_since_hit = np.full(b, self.n_slots, dtype=np.float64)
        self.n_visits = np.zeros(b, dtype=np.int64)
        self.n_hits = np.zeros(b, dtype=np.int64)
        self.ewma_activity = np.zeros(b, dtype=np.float64)
        self.ewma_snr_db = np.zeros(b, dtype=np.float64)

        # Periodicity, filled by analysis.estimators via refresh_periods().
        self.period_hat_slots = np.zeros(b, dtype=np.float64)
        self.period_confidence = np.zeros(b, dtype=np.float64)
        self.period_sigma_slots = np.zeros(b, dtype=np.float64)
        self.last_hit_slot = np.full(b, -1, dtype=np.int64)
        self._hit_times: list[list[int]] = [[] for _ in range(b)]
        # Visit times are retained as well as hit times: the Lomb-Scargle
        # estimator needs the sampling schedule to deconvolve its own spectral
        # window, without which it reports OUR sweep period (estimators.py).
        self._visit_times: list[list[int]] = [[] for _ in range(b)]
        self.track_visits = config.belief.period_estimator != "none"

        self.t = 0
        self.current_center: int | None = None
        self.slots_since_retune = 0
        self.total_dwells = 0
        self.total_hits = 0
        self._visited_mask = np.zeros(b, dtype=bool)

    # -- update ----------------------------------------------------------- #
    def decay(self, n_slots: int = 1) -> None:
        """Relax the posterior toward the prior over ``n_slots`` elapsed slots.

        ``alpha <- 1 + (alpha - a0) * rho**n`` keeps the prior as the fixed point,
        so a channel unvisited for many half-lives returns to "I don't know"
        rather than to "empty".

        Args:
            n_slots: Number of slots elapsed since the last update.
        """
        r = self.rho**n_slots
        self.alpha = self.alpha_prior + (self.alpha - self.alpha_prior) * r
        self.beta = self.beta_prior + (self.beta - self.beta_prior) * r

    def update(self, obs: Observation) -> None:
        """Fold one dwell's observation into the belief.

        Args:
            obs: The receiver's report. ``pfa_flags`` and ``truth_ids`` are
                ignored here by design -- the belief must be buildable on real
                hardware, where neither exists.
        """
        n = max(int(obs.slots_elapsed), 1)
        self.decay(n)
        self.time_since_visit += n
        self.time_since_hit += n
        self.t = int(obs.t)

        lo, hi = obs.window
        idx = np.arange(lo, hi)
        hits = np.asarray(obs.hits, dtype=bool)

        self.alpha[idx] += hits
        self.beta[idx] += ~hits
        self.time_since_visit[idx] = 0.0
        self.n_visits[idx] += 1
        self._visited_mask[idx] = True
        self.total_dwells += 1

        a = self.cfg.belief.ewma_activity_alpha
        self.ewma_activity[idx] = (1 - a) * self.ewma_activity[idx] + a * hits

        if self.track_visits:
            for c in idx:
                vis = self._visit_times[int(c)]
                vis.append(self.t)
                if len(vis) > _MAX_VISIT_HISTORY:
                    del vis[0]

        hit_idx = idx[hits]
        if hit_idx.size:
            self.time_since_hit[hit_idx] = 0.0
            self.n_hits[hit_idx] += 1
            self.last_hit_slot[hit_idx] = self.t
            self.total_hits += int(hit_idx.size)
            snr = np.nan_to_num(np.asarray(obs.snr_est_db, dtype=np.float64)[hits], nan=0.0)
            s = self.cfg.belief.ewma_snr_alpha
            self.ewma_snr_db[hit_idx] = (1 - s) * self.ewma_snr_db[hit_idx] + s * snr
            for c in hit_idx:
                hist = self._hit_times[int(c)]
                hist.append(self.t)
                if len(hist) > _MAX_HIT_HISTORY:
                    del hist[0]

        if self.current_center is not None and obs.window != self._window_of(self.current_center):
            self.slots_since_retune = 0
        else:
            self.slots_since_retune += n

    def _window_of(self, center: int) -> tuple[int, int]:
        """Return the channel window a centre-index action would observe."""
        k = self.cfg.receiver.ibw_channels
        lo = int(center) - (k - 1) // 2
        return lo, lo + k

    def note_action(self, center: int) -> None:
        """Record the action taken, so retune bookkeeping stays correct."""
        self.current_center = int(center)

    # -- derived quantities ------------------------------------------------ #
    @property
    def p_occupied(self) -> np.ndarray:
        """Posterior mean occupancy probability per channel."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> np.ndarray:
        """Posterior standard deviation per channel."""
        a, b = self.alpha, self.beta
        s = a + b
        return np.sqrt(a * b / (s * s * (s + 1.0)))

    def hit_times(self, channel: int) -> np.ndarray:
        """Return the retained hit timestamps for one channel, in slots."""
        return np.asarray(self._hit_times[int(channel)], dtype=np.float64)

    def visit_times(self, channel: int) -> np.ndarray:
        """Return the retained visit timestamps for one channel, in slots."""
        return np.asarray(self._visit_times[int(channel)], dtype=np.float64)

    def interferer_score(self) -> np.ndarray:
        """Observation-only proxy for "loud, always-on, low-value".

        Ground-truth threat is not observable, so the agent must infer it. An
        interferer is a channel that is *always* occupied and *loud*; genuine
        threats (scanning radars, pop-ups, bursty comms) are intermittent. The
        score therefore multiplies a sharpened occupancy probability by a
        confidence factor in the visit count and a soft threshold on estimated
        SNR. It is a heuristic and is labelled as one; the RL agent is free to
        learn to ignore it.

        Returns:
            Float64 array in ``[0, 1]``, shape ``(B,)``.
        """
        p = self.p_occupied
        confidence = self.n_visits / (self.n_visits + 20.0)
        loud = 1.0 / (1.0 + np.exp(-(self.ewma_snr_db - 20.0) / 5.0))
        return np.clip(p * p * confidence * loud, 0.0, 1.0)

    def phase_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sin, cos)`` of the predicted phase within the scan cycle.

        Zeroed where the period estimate is absent or below the configured
        confidence, so an unmeasured channel contributes no spurious phase.

        Returns:
            Two float64 arrays of shape ``(B,)``.
        """
        valid = (self.period_hat_slots > 0) & (
            self.period_confidence >= self.cfg.belief.period_min_confidence
        )
        phase = np.zeros(self.n_channels, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            elapsed = self.t - self.last_hit_slot
            frac = np.where(valid, np.mod(elapsed, np.maximum(self.period_hat_slots, 1.0)), 0.0)
            phase = np.where(valid, frac / np.maximum(self.period_hat_slots, 1.0), 0.0)
        ang = 2.0 * np.pi * phase
        return np.where(valid, np.sin(ang), 0.0), np.where(valid, np.cos(ang), 0.0)

    def time_to_next_arrival(self) -> np.ndarray:
        """Predicted slots until the next beam arrival, ``inf`` where unknown.

        Returns:
            Float64 array of shape ``(B,)``.
        """
        out = np.full(self.n_channels, np.inf, dtype=np.float64)
        valid = (
            (self.period_hat_slots > 0)
            & (self.period_confidence >= self.cfg.belief.period_min_confidence)
            & (self.last_hit_slot >= 0)
        )
        if not np.any(valid):
            return out
        p = self.period_hat_slots[valid]
        elapsed = self.t - self.last_hit_slot[valid]
        out[valid] = p - np.mod(elapsed, p)
        return out

    # -- feature vector ---------------------------------------------------- #
    def features(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose the belief as a fixed-length feature vector for the ML agents.

        Returns:
            ``(channel_features, global_features)`` with shapes ``(B, 12)`` and
            ``(5,)``, both float32. Every feature is scaled to roughly ``[0, 1]``
            or ``[-1, 1]`` so no normalisation layer is required upstream.
        """
        sin_p, cos_p = self.phase_features()
        f = np.empty((self.n_channels, N_CHANNEL_FEATURES), dtype=np.float32)
        f[:, 0] = self.p_occupied
        f[:, 1] = self.posterior_std
        f[:, 2] = np.log1p(self.alpha) / 5.0
        f[:, 3] = np.log1p(self.beta) / 5.0
        f[:, 4] = np.log1p(self.time_since_visit) / self._log_t
        f[:, 5] = np.log1p(self.time_since_hit) / self._log_t
        f[:, 6] = self.ewma_activity
        f[:, 7] = self.ewma_snr_db / 40.0
        f[:, 8] = self.period_confidence
        f[:, 9] = sin_p
        f[:, 10] = cos_p
        f[:, 11] = self.interferer_score()

        g = np.array(
            [
                self.t / max(self.n_slots, 1),
                (self.current_center or 0) / max(self.n_channels, 1),
                min(self.slots_since_retune, 1000) / 1000.0,
                self._visited_mask.mean(),
                self.total_hits / max(self.total_dwells, 1),
            ],
            dtype=np.float32,
        )
        return f, g

    def flat_features(self) -> np.ndarray:
        """Return the belief as one flat float32 vector of length ``B*12 + 5``."""
        f, g = self.features()
        return np.concatenate([f.ravel(), g])

    # -- periodicity ------------------------------------------------------- #
    def refresh_periods(
        self, channels: np.ndarray | None = None, min_hits: int = 5, n_bins: int | None = None
    ) -> None:
        """Re-estimate per-channel scan periods from retained hit timestamps.

        Delegates to :mod:`smartscan.analysis.estimators`. Called on a coarse
        cadence rather than every slot -- a Lomb-Scargle periodogram per channel
        per slot would dominate the runtime and the estimate cannot meaningfully
        change between consecutive dwells anyway.

        Args:
            channels: Channels to refresh; ``None`` means all with enough hits.
            min_hits: Minimum retained hits before an estimate is attempted.
            n_bins: Period-grid resolution; defaults to the online value.
        """
        if self.cfg.belief.period_estimator == "none":
            return
        from smartscan.analysis.estimators import estimate_period

        targets = range(self.n_channels) if channels is None else [int(c) for c in channels]
        est_cfg = self.cfg.analysis.estimators
        bins = int(n_bins or _ONLINE_PERIOD_BINS)
        lo = est_cfg.period_grid_s.lo / self.cfg.time.dt_s
        hi = est_cfg.period_grid_s.hi / self.cfg.time.dt_s

        for c in targets:
            times = self.hit_times(c)
            if times.size < min_hits:
                continue
            result = estimate_period(
                times, self.visit_times(c) if self.track_visits else None,
                method=self.cfg.belief.period_estimator,
                period_min=lo, period_max=hi, config=est_cfg, t_now=self.t,
                n_bins_override=bins,
            )
            self.period_hat_slots[c] = result.period
            self.period_confidence[c] = result.confidence
            self.period_sigma_slots[c] = result.sigma if np.isfinite(result.sigma) else 0.0

    def reset(self) -> None:
        """Return the belief to its prior, keeping configuration."""
        self.__init__(self.cfg, self.n_slots)
