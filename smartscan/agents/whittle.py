"""Restless multi-armed bandit formulation and the Whittle index policy.

Why this and not a classical bandit
-----------------------------------
Each channel is an arm whose state **evolves whether or not we observe it**. That
is the definition of a *restless* bandit (Whittle 1988), and it voids the
optimality guarantees of UCB/Thompson, which assume a rested world. The Whittle
index is the standard near-optimal relaxation: it decouples the arms via a
Lagrange multiplier (a "subsidy for passivity") and ranks them by the subsidy at
which the optimal single-arm policy becomes indifferent between acting and not.

Model
-----
Each channel is a **Gilbert-Elliott** two-state chain, ``idle -> busy`` with
probability ``p01`` and ``busy -> busy`` with ``p11``. Since a channel is only
observed when visited, the sufficient statistic is the scalar belief
``omega = P(busy)``, which under the passive action evolves as::

    T(omega) = p01 + omega * (p11 - p01)

Single-arm problem with subsidy ``lambda``, discount ``beta``::

    V(w) = max(  w + beta * [ w*V(p11) + (1-w)*V(p01) ],      # act: reward w, state revealed
                 lambda + beta * V(T(w))                       # stay passive, collect subsidy
              )

**Indexability** requires the passive set ``P(lambda) = {w : passive is optimal}``
to be monotone non-decreasing in ``lambda``. We do not assume it -- we verify it
numerically per channel and report violations, as the problem brief asks.
Given indexability, ``W(w) = inf{ lambda : w in P(lambda) }``.

Liu & Zhao (2010) prove indexability and give a closed form for the positively
correlated case ``p11 >= p01``. We use their structural result as a **test**
(for identical positively-correlated channels the Whittle policy must coincide
with the myopic policy) rather than as the implementation, because the numerical
route also covers the negatively correlated case that a frequency-agile emitter
actually produces.

Activation constraint
---------------------
Standard Whittle activates the ``K`` highest-index arms. Our receiver cannot:
its ``K`` channels must be **contiguous**. We therefore select the contiguous
window maximising the summed index -- a constrained activation that is a
restriction of, not a departure from, the index policy.

Full derivation: ``docs/theory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config
from smartscan.env.types import Observation

__all__ = [
    "GilbertElliott",
    "WhittleIndexScheduler",
    "estimate_gilbert_elliott",
    "whittle_index_curve",
]

#: Belief-grid resolution for the single-arm value iteration.
_N_OMEGA: int = 101

#: Number of subsidy levels swept when building an index curve.
_N_LAMBDA: int = 64


@dataclass(frozen=True)
class GilbertElliott:
    """Two-state channel dynamics.

    Attributes:
        p01: ``P(busy at t+1 | idle at t)``.
        p11: ``P(busy at t+1 | busy at t)``.
        n_obs: Number of observations the estimate rests on.
    """

    p01: float
    p11: float
    n_obs: int = 0

    @property
    def stationary(self) -> float:
        """Stationary probability of the busy state."""
        denom = 1.0 + self.p01 - self.p11
        return self.p01 / denom if abs(denom) > 1e-12 else 0.5

    @property
    def positively_correlated(self) -> bool:
        """Whether ``p11 >= p01`` (the Liu & Zhao closed-form regime)."""
        return self.p11 >= self.p01


def estimate_gilbert_elliott(
    belief: BeliefState, channel: int, min_obs: int = 8, prior: GilbertElliott | None = None
) -> GilbertElliott:
    """Estimate two-state dynamics for one channel from sparse observations.

    A channel is observed only when visited, so consecutive observations are
    separated by arbitrary gaps and the one-step transition matrix cannot be
    counted directly. For a two-state chain the ``k``-step matrix has the closed
    form::

        P(busy_{t+k} | busy_t) = w0 + (1 - w0) * mu**k
        P(busy_{t+k} | idle_t) = w0 * (1 - mu**k)

    with ``w0`` the stationary probability and ``mu = p11 - p01`` the second
    eigenvalue. We estimate ``w0`` from the smoothed marginal hit rate and ``mu``
    by regressing the observed lag correlation on the gap, then invert to
    ``(p01, p11)``. This is a moment estimator, not a maximum-likelihood fit --
    documented as such because it is deliberately cheap enough to run every
    refresh.

    Args:
        belief: Shared belief state.
        channel: Channel index.
        min_obs: Minimum observations before an estimate is attempted.
        prior: Fallback returned when data is insufficient.

    Returns:
        The estimated :class:`GilbertElliott` dynamics.
    """
    fallback = prior or GilbertElliott(0.02, 0.9)
    n_vis = int(belief.n_visits[channel])
    if n_vis < min_obs:
        return fallback

    w0 = float(np.clip((belief.n_hits[channel] + 0.5) / (n_vis + 1.0), 1e-3, 1.0 - 1e-3))

    hits = belief.hit_times(channel)
    visits = belief.visit_times(channel)
    if visits.size < 3:
        return GilbertElliott(fallback.p01, fallback.p11, n_vis)

    states = np.isin(visits, hits).astype(np.float64)
    gaps = np.diff(visits)
    s0, s1 = states[:-1], states[1:]
    valid = gaps > 0
    if valid.sum() < 2:
        return GilbertElliott(fallback.p01, fallback.p11, n_vis)

    # Lag correlation of the observed state sequence: E[(s_t - w0)(s_{t+k} - w0)]
    # decays as mu**k * Var, so a log-linear fit recovers mu.
    var = max(w0 * (1.0 - w0), 1e-6)
    prod = (s0[valid] - w0) * (s1[valid] - w0) / var
    g = gaps[valid]
    order = np.argsort(g)
    g, prod = g[order], prod[order]
    n_bins = min(6, max(2, g.size // 4))
    edges = np.quantile(g, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1.0
    xs, ys = [], []
    for i in range(n_bins):
        sel = (g >= edges[i]) & (g < edges[i + 1])
        if sel.sum() >= 2:
            c = float(np.mean(prod[sel]))
            if c > 1e-3:  # log of a non-positive correlation is meaningless
                xs.append(float(np.mean(g[sel])))
                ys.append(np.log(min(c, 1.0)))
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0])
        mu = float(np.clip(np.exp(slope), -0.999, 0.999))
    else:
        # Single usable bin: read mu off directly at the mean gap.
        c = float(np.clip(np.mean(prod), 1e-3, 1.0))
        mu = float(np.clip(c ** (1.0 / max(np.mean(g), 1.0)), 0.0, 0.999))

    # Invert (w0, mu) -> (p01, p11).
    p01 = float(np.clip(w0 * (1.0 - mu), 1e-4, 1.0 - 1e-4))
    p11 = float(np.clip(mu + p01, 1e-4, 1.0 - 1e-4))
    return GilbertElliott(p01, p11, n_vis)


def _interp_weights(points: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolation indices and weights of ``points`` on a uniform ``grid``."""
    n = grid.size
    x = np.clip(points, grid[0], grid[-1])
    pos = (x - grid[0]) / (grid[-1] - grid[0]) * (n - 1)
    i = np.clip(np.floor(pos).astype(int), 0, n - 2)
    return i, pos - i


def whittle_index_curve(
    dynamics: GilbertElliott,
    beta: float = 0.99,
    n_omega: int = _N_OMEGA,
    n_lambda: int = _N_LAMBDA,
    subsidy_lo: float = 0.0,
    subsidy_hi: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 400,
    check_indexability: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute the Whittle index as a function of belief, numerically.

    One value iteration is run per subsidy level, **vectorised across all subsidy
    levels simultaneously** (the belief transition ``T`` does not depend on
    ``lambda``, so the interpolation weights are shared). The index is then read
    off as the smallest subsidy at which each belief enters the passive set.

    Args:
        dynamics: Estimated channel dynamics.
        beta: Discount factor for the single-arm problem.
        n_omega: Belief-grid resolution.
        n_lambda: Number of subsidy levels.
        subsidy_lo: Lowest subsidy considered.
        subsidy_hi: Highest subsidy considered.
        tol: Value-iteration convergence tolerance.
        max_iter: Value-iteration iteration cap.
        check_indexability: Verify the passive set is monotone in ``lambda``.

    Returns:
        ``(omega_grid, index, indexable)``: the belief grid, the Whittle index on
        it, and whether the indexability condition held. A channel that fails
        indexability still receives an index (the monotone envelope), but the
        failure is reported rather than hidden.
    """
    omega = np.linspace(0.0, 1.0, n_omega)
    lam = np.linspace(subsidy_lo, subsidy_hi, n_lambda)
    p01, p11 = dynamics.p01, dynamics.p11

    t_omega = p01 + omega * (p11 - p01)
    i_t, w_t = _interp_weights(t_omega, omega)
    i_11, w_11 = _interp_weights(np.array([p11]), omega)
    i_01, w_01 = _interp_weights(np.array([p01]), omega)

    v = np.zeros((n_lambda, n_omega), dtype=np.float64)
    for _ in range(max_iter):
        v_at_t = v[:, i_t] * (1.0 - w_t) + v[:, i_t + 1] * w_t
        v_11 = v[:, i_11] * (1.0 - w_11) + v[:, i_11 + 1] * w_11
        v_01 = v[:, i_01] * (1.0 - w_01) + v[:, i_01 + 1] * w_01
        q_active = omega[None, :] + beta * (omega[None, :] * v_11 + (1.0 - omega[None, :]) * v_01)
        q_passive = lam[:, None] + beta * v_at_t
        v_new = np.maximum(q_active, q_passive)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    v_at_t = v[:, i_t] * (1.0 - w_t) + v[:, i_t + 1] * w_t
    v_11 = v[:, i_11] * (1.0 - w_11) + v[:, i_11 + 1] * w_11
    v_01 = v[:, i_01] * (1.0 - w_01) + v[:, i_01 + 1] * w_01
    q_active = omega[None, :] + beta * (omega[None, :] * v_11 + (1.0 - omega[None, :]) * v_01)
    q_passive = lam[:, None] + beta * v_at_t
    passive = q_passive >= q_active  # shape (n_lambda, n_omega)

    indexable = True
    if check_indexability:
        # The passive set must only grow as the subsidy rises.
        indexable = bool(np.all(np.diff(passive.astype(np.int8), axis=0) >= 0))

    # W(w) = smallest subsidy making w passive; never passive -> top of range.
    first = np.argmax(passive, axis=0)
    never = ~passive.any(axis=0)
    index = lam[first]
    index[never] = subsidy_hi
    # Enforce monotonicity in belief, which holds for every indexable instance
    # and repairs grid-level ragged edges without hiding a genuine violation.
    index = np.maximum.accumulate(index)
    return omega, index, indexable


class WhittleIndexScheduler(Scheduler):
    """Restless-bandit index policy over the shared belief.

    Per-channel dynamics are re-estimated, and the index curves rebuilt, on a
    coarse cadence: the estimate cannot meaningfully change between consecutive
    dwells, and rebuilding every slot would dominate the runtime.

    Args:
        config: Resolved configuration.
        seed: Seed for tie-breaking.
        name: Optional display name.
        refresh_slots: Slots between index rebuilds.
    """

    key = "whittle"

    def __init__(
        self, config: Config, seed: int = 0, name: str | None = None, refresh_slots: int = 250
    ) -> None:
        super().__init__(config, seed, name)
        wc = config.agents.whittle
        self.wc = wc
        self.beta = float(getattr(wc, "discount", 0.99))
        self.refresh_slots = int(refresh_slots)
        self.retune_penalty = (
            config.receiver.t_settle_slots / (1.0 + config.receiver.t_settle_slots)
        ) * config.reward.w4_retune
        self.coverage_weight = config.agents.coverage_weight

        self._omega_grid = np.linspace(0.0, 1.0, _N_OMEGA)
        self._curves: dict[tuple[int, int], np.ndarray] = {}
        self._channel_curve = np.zeros((self.n_channels, _N_OMEGA), dtype=np.float64)
        self._last_refresh = -10**9
        #: Channels whose estimated dynamics failed the indexability check.
        self.indexability_violations: set[int] = set()
        self._seed_default_curve()

    def _seed_default_curve(self) -> None:
        """Populate every channel with the prior-dynamics index curve."""
        _, curve, _ = whittle_index_curve(
            GilbertElliott(0.02, 0.9), self.beta,
            subsidy_lo=self.wc.subsidy_lo, subsidy_hi=self.wc.subsidy_hi,
            check_indexability=False,
        )
        self._channel_curve[:] = curve

    def _curve_for(self, dyn: GilbertElliott, channel: int) -> np.ndarray:
        """Return the index curve for these dynamics, using a quantised cache."""
        key = (int(dyn.p01 * 50), int(dyn.p11 * 50))
        cached = self._curves.get(key)
        if cached is None:
            _, cached, ok = whittle_index_curve(
                dyn, self.beta,
                subsidy_lo=self.wc.subsidy_lo, subsidy_hi=self.wc.subsidy_hi,
                tol=self.wc.bisect_tol, check_indexability=self.wc.check_indexability,
            )
            self._curves[key] = cached
            if self.wc.check_indexability and not ok:
                self.indexability_violations.add(channel)
        return cached

    def refresh(self, belief: BeliefState) -> None:
        """Re-estimate dynamics and rebuild index curves for every channel."""
        for c in range(self.n_channels):
            dyn = estimate_gilbert_elliott(belief, c, self.wc.min_transitions_for_estimate)
            self._channel_curve[c] = self._curve_for(dyn, c)

    def indices(self, belief: BeliefState) -> np.ndarray:
        """Return the current Whittle index of every channel.

        Args:
            belief: Shared belief state.

        Returns:
            Float64 array of shape ``(B,)``.
        """
        omega = np.clip(belief.p_occupied, 0.0, 1.0)
        pos = omega * (_N_OMEGA - 1)
        i = np.clip(np.floor(pos).astype(int), 0, _N_OMEGA - 2)
        w = pos - i
        rows = np.arange(self.n_channels)
        return self._channel_curve[rows, i] * (1.0 - w) + self._channel_curve[rows, i + 1] * w

    def act(self, belief: BeliefState, t: int) -> int:
        """Select the contiguous window of highest summed Whittle index."""
        if t - self._last_refresh >= self.refresh_slots:
            self.refresh(belief)
            self._last_refresh = t

        idx = self.indices(belief)
        # Threat weighting: ground-truth priority is unobservable, so we discount
        # channels the belief has learned look like always-on interferers.
        idx = idx * (1.0 - 0.9 * belief.interferer_score())
        # The mission objective carries a max-staleness penalty, so the index
        # policy is applied to the objective, not to a detection-only proxy.
        idx = idx + self.coverage_weight * (belief.time_since_visit / max(belief.n_slots, 1))
        action = self.argmax_legal(self.window_value(idx), self.retune_penalty)
        self.last_action = action
        return action

    def observe(self, obs: Observation) -> None:
        """No-op: all state the policy needs already lives in the belief."""
        return None

    def reset(self) -> None:
        """Clear cached curves and refresh bookkeeping."""
        super().reset()
        self._last_refresh = -10**9
        self.indexability_violations.clear()
        self._seed_default_curve()
