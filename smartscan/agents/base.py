"""Common scheduler interface.

Every policy in the package -- the sweep baseline, the bandits, the Whittle index
and the learned agents alike -- implements ``act(belief, t) -> action``. They all
see the same :class:`~smartscan.agents.belief.BeliefState` and nothing else, so
a comparison between them is a comparison of *policies*, not of information.

The base class also owns the action geometry (which channels a centre-index
action covers, and which actions are legal), so no subclass reimplements it and
gets the edge cases subtly wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = ["Scheduler", "window_matrix"]


def window_matrix(n_channels: int, k: int, action_space: str = "center_index") -> np.ndarray:
    """Precompute the channel window covered by every action.

    Args:
        n_channels: Band size ``B``.
        k: IBW in channels.
        action_space: ``"center_index"`` or ``"window_start"``.

    Returns:
        Int32 array of shape ``(B, k)``. Rows for illegal actions are filled with
        ``-1`` and must be masked out before use.
    """
    out = np.full((n_channels, k), -1, dtype=np.int32)
    for a in range(n_channels):
        lo = a if action_space == "window_start" else a - (k - 1) // 2
        if 0 <= lo <= n_channels - k:
            out[a] = np.arange(lo, lo + k, dtype=np.int32)
    return out


class Scheduler(ABC):
    """Abstract receiver scheduler.

    Args:
        config: Resolved configuration.
        seed: Seed for any policy-internal randomness. Drawn from the
            ``("agent",)`` substream so the world is unaffected by the policy.
        name: Optional display name; defaults to the class name.
    """

    #: Short identifier used in configs, CLI flags and results tables.
    key: str = "base"

    #: Whether this policy consumes ``belief.period_hat_slots``. Period
    #: estimation is by far the most expensive thing the belief can do (a
    #: Lomb-Scargle per channel), so the runner only pays for it when a policy
    #: actually reads the result.
    needs_periods: bool = False

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        self.cfg = config
        self.n_channels = config.n_channels
        self.k = config.receiver.ibw_channels
        self.name = name or type(self).__name__
        self.windows = window_matrix(self.n_channels, self.k, config.receiver.action_space)
        self.legal = self.windows[:, 0] >= 0
        self.legal_indices = np.flatnonzero(self.legal).astype(np.int32)
        from smartscan.seeding import SeedTree

        self.rng = SeedTree(int(seed)).rng("agent")
        self.last_action: int | None = None
        self.t = 0

    # -- interface --------------------------------------------------------- #
    @abstractmethod
    def act(self, belief: BeliefState, t: int) -> int:
        """Choose the centre channel to tune to for the next dwell.

        Args:
            belief: Shared belief state, already updated with all prior dwells.
            t: Current slot index.

        Returns:
            A legal action index in ``[0, B)``.
        """

    def observe(self, obs: object) -> None:
        """Optional hook called after each dwell. No-op by default.

        Args:
            obs: The :class:`~smartscan.env.types.Observation` just received.
        """
        return None

    def reset(self) -> None:
        """Reset any per-episode policy state."""
        self.last_action = None
        self.t = 0

    # -- helpers ----------------------------------------------------------- #
    def channels_of(self, action: int) -> np.ndarray:
        """Return the channel indices an action observes."""
        return self.windows[int(action)]

    def window_value(self, per_channel: np.ndarray) -> np.ndarray:
        """Sum a per-channel score over each action's window.

        This is the workhorse of every value-based policy here: a scheduler
        scores channels, and the action that matters is the *window* that
        collects the most score. Illegal actions receive ``-inf``.

        Args:
            per_channel: Float array of shape ``(B,)``.

        Returns:
            Float64 array of shape ``(B,)`` of window totals.
        """
        vals = np.full(self.n_channels, -np.inf, dtype=np.float64)
        w = self.windows[self.legal]
        vals[self.legal] = np.asarray(per_channel, dtype=np.float64)[w].sum(axis=1)
        return vals

    def argmax_legal(self, scores: np.ndarray, retune_penalty: float = 0.0) -> int:
        """Return the best legal action, optionally charging for a retune.

        Args:
            scores: Per-action scores of shape ``(B,)``.
            retune_penalty: Score subtracted from every action other than the
                one currently tuned. This is how a policy is made aware that
                moving costs ``t_settle`` slots of blindness.

        Returns:
            The chosen action index.
        """
        s = np.asarray(scores, dtype=np.float64).copy()
        s[~self.legal] = -np.inf
        if retune_penalty and self.last_action is not None:
            s -= retune_penalty
            s[self.last_action] += retune_penalty
        best = np.flatnonzero(s == s.max())
        # Ties broken at random rather than by index: a deterministic tie-break
        # makes an all-equal prior degenerate into "always tune to channel 1".
        return int(best[0] if best.size == 1 else self.rng.choice(best))
