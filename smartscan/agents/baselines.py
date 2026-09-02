"""Open-loop baselines -- the strawmen the problem statement criticises.

These do not look at the belief. That is precisely the point: they are the
incumbent practice, and the case for a closed-loop scheduler is exactly the
margin by which it beats them.

A caution recorded up front (``docs/architecture.md`` §17-C):
:class:`SequentialSweep` is a **stronger** baseline than it looks. At ``K = 4``
and ``t_settle = 2`` a full-band revisit costs 16 tunes x 3 slots = 48 ms, so it
revisits every channel about 208 times in a 10 s episode. Against *static*
emitters it is close to optimal and should not be beaten by much. The gains have
to come from scanning, agile and pop-up emitters.
"""

from __future__ import annotations

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = ["PriorityRoundRobin", "RandomScan", "SequentialSweep"]


class SequentialSweep(Scheduler):
    """Classic saw-tooth sweep: the fastest possible full-band revisit.

    Steps by ``step_channels`` (default ``K``, i.e. non-overlapping windows) so
    no dwell is wasted re-observing channels the previous dwell already covered.
    This is the incumbent to beat.

    Args:
        config: Resolved configuration.
        seed: Unused; accepted for interface uniformity.
    """

    key = "sequential"

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        self.step = max(int(config.agents.sequential_sweep.step_channels), 1)
        self.direction = config.agents.sequential_sweep.direction
        self.dwell = max(int(config.agents.sequential_sweep.dwell_slots), 1)
        self._cursor = 0
        self._sign = 1
        self._held = 0

    def reset(self) -> None:
        """Return the sweep to the bottom of the band."""
        super().reset()
        self._cursor = 0
        self._sign = 1
        self._held = 0

    def act(self, belief: BeliefState, t: int) -> int:
        """Advance the saw-tooth by one step and return the next centre channel."""
        action = int(self.legal_indices[self._cursor % self.legal_indices.size])
        # Hold each window for `dwell` slots so the settling overhead is amortised.
        self._held += 1
        if self._held < self.dwell:
            self.last_action = action
            return action
        self._held = 0
        n = self.legal_indices.size
        stride = max(self.step, 1)
        if self.direction == "updown":
            nxt = self._cursor + self._sign * stride
            if nxt >= n or nxt < 0:
                self._sign *= -1
                nxt = self._cursor + self._sign * stride
            self._cursor = int(np.clip(nxt, 0, n - 1))
        else:
            self._cursor = (self._cursor + stride) % n
        self.last_action = action
        return action

    @property
    def total_observation_slots_per_channel(self) -> float:
        """Slots each channel is observed over an episode, for the current dwell.

        ``T * d / (N_windows * (d + t_settle))``. This is the quantity a sweep
        should maximise against sparse targets, and it is why ``dwell_slots``
        matters: it rises from 208 slots at ``d = 1`` to 446 at ``d = 5`` in the
        default configuration, at the cost of a proportionally slower revisit.
        """
        d = self.dwell
        n_win = int(np.ceil(self.legal_indices.size / max(self.step, 1)))
        return self.cfg.n_slots * d / (n_win * (d + self.cfg.receiver.t_settle_slots))

    @property
    def revisit_period_slots(self) -> int:
        """Slots for one complete band sweep, including settling.

        Used by the scan-on-scan module: this is the receiver period ``Tr`` whose
        commensurability with an emitter's scan period ``Te`` decides whether the
        sweep is guaranteed to intercept or is provably blind (``scan_on_scan``).
        """
        n_tunes = int(np.ceil(self.legal_indices.size / max(self.step, 1)))
        return n_tunes * (self.dwell + self.cfg.receiver.t_settle_slots)


class RandomScan(Scheduler):
    """Uniform random tuning over legal actions.

    Included because it is the honest lower bound on "closed loop": it pays the
    full retune cost on almost every dwell while extracting no information from
    the belief.
    """

    key = "random"

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        self.dwell = max(int(config.agents.random_scan.dwell_slots), 1)
        self._held = 0
        self._action = 0

    def reset(self) -> None:
        """Clear the held action."""
        super().reset()
        self._held = 0

    def act(self, belief: BeliefState, t: int) -> int:
        """Return a uniformly random legal action, held for ``dwell`` slots."""
        if self._held <= 0:
            self._action = int(self.rng.choice(self.legal_indices))
            self._held = self.dwell
        self._held -= 1
        self.last_action = self._action
        return self._action


class PriorityRoundRobin(Scheduler):
    """Weighted round robin over an *a priori* band list.

    Simulates pre-mission intelligence (an EOB, an ELINT library) that is
    **wrong 40 % of the time**, per the problem statement. The corrupted entries
    point at empty channels while genuinely active channels are deprioritised.

    ``weight_floor`` is what makes the degradation *graceful* rather than total:
    a wrongly-deprioritised channel still gets visited occasionally, so the
    scheduler recovers instead of being permanently blind to the emitters its
    briefing got wrong. Removing the floor is an instructive ablation.

    Args:
        config: Resolved configuration.
        seed: Seed for the prior corruption and the interleaving order.
        truth_channels: Genuinely occupied channels, used **only** to construct
            a realistically-wrong prior at episode setup. The policy itself never
            reads ground truth at decision time.
    """

    key = "priority_rr"

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        name: str | None = None,
        truth_channels: np.ndarray | None = None,
    ) -> None:
        super().__init__(config, seed, name)
        self.weights = self._build_prior(truth_channels)
        self._schedule = self._build_schedule()
        self._cursor = 0

    def _build_prior(self, truth_channels: np.ndarray | None) -> np.ndarray:
        """Build a deliberately imperfect a-priori channel weighting."""
        cfg = self.cfg.agents.priority_round_robin
        w = np.full(self.n_channels, cfg.weight_floor, dtype=np.float64)
        if truth_channels is None or len(truth_channels) == 0:
            return w / w.sum()

        truth = np.unique(np.asarray(truth_channels, dtype=int))
        n_wrong = int(round(cfg.prior_wrong_frac * truth.size))
        keep = self.rng.permutation(truth)[: truth.size - n_wrong]
        # The corrupted half of the briefing points somewhere else entirely.
        decoys = self.rng.choice(
            np.setdiff1d(np.arange(self.n_channels), truth), size=min(n_wrong, self.n_channels - truth.size),
            replace=False,
        ) if n_wrong > 0 and truth.size < self.n_channels else np.zeros(0, dtype=int)

        w[keep] = 1.0
        w[decoys] = 1.0
        return w / w.sum()

    def _build_schedule(self, length: int = 512) -> np.ndarray:
        """Expand the weights into a deterministic interleaved visit order."""
        scores = self.window_value(self.weights)
        scores[~np.isfinite(scores)] = 0.0
        p = scores[self.legal_indices]
        p = p / p.sum() if p.sum() > 0 else np.full(p.size, 1.0 / p.size)
        counts = np.maximum(np.round(p * length).astype(int), 1)
        schedule = np.repeat(self.legal_indices, counts)
        return self.rng.permutation(schedule).astype(np.int32)

    def reset(self) -> None:
        """Restart the round-robin cursor."""
        super().reset()
        self._cursor = 0

    def act(self, belief: BeliefState, t: int) -> int:
        """Return the next entry of the weighted round-robin schedule."""
        action = int(self._schedule[self._cursor % self._schedule.size])
        self._cursor += 1
        self.last_action = action
        return action
