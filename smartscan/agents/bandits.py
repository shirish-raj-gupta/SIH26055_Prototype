"""Statistical bandit schedulers over the shared Beta beliefs.

These are *rested*-bandit algorithms applied to a **restless** problem: emitter
state advances while we are looking elsewhere. They are therefore expected to
lose to the Whittle index policy (:mod:`smartscan.agents.whittle`), and
demonstrating exactly that -- with confidence intervals -- is a result rather
than a failure.

Two adaptations keep them competitive rather than strawmen:

* They act on the **decayed** posterior, so a channel confirmed empty 4 s ago is
  no longer believed empty (discounted UCB, Garivier & Moulines; discounted
  Thompson sampling, Raj & Kalyani).
* They score **windows**, not channels. The receiver collects ``K`` channels per
  dwell, so the arm is really the window, and a policy that ignores this leaves
  most of its IBW on the table.
"""

from __future__ import annotations

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = ["UCB1", "EpsilonGreedy", "ThompsonSampling"]


class _BanditBase(Scheduler):
    """Shared retune-cost accounting for the bandit family."""

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        # Charge the policy for the slots a retune actually costs, expressed in
        # the same units as the score (expected detections per dwell). Without
        # this the bandits hop every slot and lose a third of their dwell time.
        self.retune_penalty = (
            config.receiver.t_settle_slots / (1.0 + config.receiver.t_settle_slots)
        ) * config.reward.w4_retune
        self.coverage_weight = config.agents.coverage_weight

    def score(self, value: np.ndarray, belief: BeliefState) -> np.ndarray:
        """Add the shared coverage term to a per-channel value.

        The objective includes a max-staleness penalty, so a policy that scores
        only expected detections is optimising a proxy rather than the mission.

        Args:
            value: Per-channel value, shape ``(B,)``.
            belief: Shared belief state.

        Returns:
            Per-channel score of shape ``(B,)``.
        """
        stale = belief.time_since_visit / max(belief.n_slots, 1)
        return np.asarray(value, dtype=np.float64) + self.coverage_weight * stale


class EpsilonGreedy(_BanditBase):
    """Epsilon-greedy over expected window occupancy, with decaying epsilon."""

    key = "epsilon_greedy"

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        cfg = config.agents.epsilon_greedy
        self.eps0, self.decay, self.eps_min = cfg.epsilon, cfg.epsilon_decay, cfg.epsilon_min
        self.eps = self.eps0

    def reset(self) -> None:
        """Restore the initial exploration rate."""
        super().reset()
        self.eps = self.eps0

    def act(self, belief: BeliefState, t: int) -> int:
        """Explore with probability ``eps``, otherwise take the best window."""
        self.eps = max(self.eps * self.decay, self.eps_min)
        if self.rng.random() < self.eps:
            action = int(self.rng.choice(self.legal_indices))
        else:
            action = self.argmax_legal(
                self.window_value(self.score(belief.p_occupied, belief)), self.retune_penalty
            )
        self.last_action = action
        return action


class UCB1(_BanditBase):
    """UCB1 with optional discounting for non-stationarity.

    The exploration bonus uses ``time_since_visit`` rather than a global pull
    count when discounting is enabled. That is the discounted-UCB form, and it is
    the right one here: what matters is not how often a channel has ever been
    visited but how *stale* the last visit is, because the emitter has been
    evolving in the meantime.
    """

    key = "ucb1"

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        cfg = config.agents.ucb1
        self.c, self.discounted, self.gamma = cfg.c, cfg.discounted, cfg.gamma

    def act(self, belief: BeliefState, t: int) -> int:
        """Return the window maximising the upper confidence bound."""
        mean = belief.p_occupied
        if self.discounted:
            # Effective sample size decays with staleness, so the bonus grows
            # for channels we have not looked at recently.
            n_eff = np.maximum(
                (belief.alpha + belief.beta - belief.alpha_prior - belief.beta_prior)
                * np.power(self.gamma, belief.time_since_visit),
                1e-6,
            )
        else:
            n_eff = np.maximum(belief.n_visits.astype(np.float64), 1e-6)
        total = max(float(n_eff.sum()), np.e)
        bonus = self.c * np.sqrt(np.log(total) / n_eff)
        action = self.argmax_legal(
            self.window_value(self.score(mean + bonus, belief)), self.retune_penalty
        )
        self.last_action = action
        return action


class ThompsonSampling(_BanditBase):
    """Thompson sampling on the decayed Beta posteriors.

    Sampling from the *decayed* posterior is what makes this work under
    non-stationarity: as a channel goes stale its posterior widens back toward
    the prior, so the sampled value regains enough variance to trigger a revisit.
    A non-decayed posterior would converge and the receiver would stop looking.
    """

    key = "thompson"

    def act(self, belief: BeliefState, t: int) -> int:
        """Sample an occupancy draw per channel and take the best window."""
        draw = self.rng.beta(belief.alpha, belief.beta)
        action = self.argmax_legal(self.window_value(self.score(draw, belief)), self.retune_penalty)
        self.last_action = action
        return action
