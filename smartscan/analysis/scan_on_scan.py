"""Deterministic scan-on-scan coincidence theory, and the schedulers it implies.

The problem statement asks for "approaches to intercept a periodic scan receiver
optimally". The classical answer is not probabilistic. Two periodic processes do
not intercept "with probability p per look": they either drift into coincidence
or they lock out **permanently**.

Setup
-----
Our receiver revisits a given channel every ``Tr`` seconds and dwells there for
``wr``. A scanning emitter illuminates us every ``Te`` seconds for
``we = (beamwidth / 360) * Te``. Writing the relative phase after ``n`` sweeps
as ``psi_n = (n*Tr - phi) mod Te``, an intercept occurs iff::

    psi_n in [0, we)   or   psi_n in (Te - wr, Te)

i.e. iff the phase lands in a window of width ``wr + we``.

Incommensurate ``Tr/Te``
    ``psi_n`` is equidistributed on ``[0, Te)`` (Weyl), so intercept is certain
    and ``E[TTI] ~ Tr*Te / (wr + we)``. The *distribution* of gaps obeys the
    **three-distance theorem**: gaps take at most three distinct values, set by
    the continued-fraction convergents of ``Tr/Te``.

Commensurate ``Tr/Te = p/q``
    ``psi_n`` takes only ``q`` distinct values, spaced ``Te/q`` apart. If
    ``Te/q > wr + we`` then for most initial phases **no n ever intercepts** --
    ``POI = 0`` for all time. This is the synchronism/blindness pathology, and a
    uniform sweep is exactly the policy most likely to fall into it.

``POI(t)`` is therefore a staircase, not ``1 - exp(-t/tau)``. Both are
implemented; the exponential is provided so the commonly-assumed (and often
wrong) approximation can be plotted alongside the truth.

References: Self & Smith, "Intercept time and its prediction", IEE Proc. F
132(4), 215-222, 1985; Clarkson & Pollington, "Performance limits of
sensor-scheduling strategies in electronic support", IEEE Trans. AES 43(2),
645-650, 2007; Stein & Johansen, Proc. IRE 46, 1958 (the random-pulse-train
counterpart, and the origin of the exponential model we show to be wrong
here); Wiley, ELINT, Artech House, 2006. US Patent 6,020,842 documents the
same blind-zone pathology in fielded ESM and fixes it by random duty
dithering; the golden-ratio sequence below is the deterministic version.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = [
    "GOLDEN",
    "SILVER",
    "CoincidenceResult",
    "CoprimeSweepScheduler",
    "PhaseLockedScheduler",
    "analyse_coincidence",
    "beam_dwell_s",
    "expected_time_to_intercept",
    "poi_exponential",
    "probability_of_intercept",
    "three_distance_gaps",
]

#: The golden ratio. Its continued-fraction expansion is all ones, making it the
#: worst-approximable irrational -- which is exactly the property that maximises
#: the minimum phase gap and so prevents synchronism lockout.
GOLDEN: float = (1.0 + 5.0**0.5) / 2.0
SILVER: float = 1.0 + 2.0**0.5


def beam_dwell_s(beamwidth_deg: float, scan_period_s: float) -> float:
    """Time a rotating beam illuminates a fixed observer, per revolution.

    ``we = (beamwidth / 360) * Ts``. For a 1 deg beam on a 4 s scan this is
    **11 ms out of every 4000 ms** -- the needle this whole module exists to find.

    Args:
        beamwidth_deg: 3 dB beamwidth in degrees.
        scan_period_s: Revolution period in seconds.

    Returns:
        Illumination window in seconds.
    """
    return beamwidth_deg / 360.0 * scan_period_s


@dataclass(frozen=True)
class CoincidenceResult:
    """Outcome of a scan-on-scan analysis.

    Attributes:
        mean_tti_s: Mean time to first intercept over initial phases that ever
            intercept, in seconds (``inf`` if none do).
        median_tti_s: Median of the same distribution.
        blind_fraction: Fraction of initial phases that **never** intercept
            within the horizon -- the synchronism pathology, in one number.
        closed_form_tti_s: The classical ``Tr*Te/(wr+we)`` estimate.
        commensurate: Whether ``Tr/Te`` is a low-order rational.
        ratio: ``Tr / Te``.
        rational: Best low-order rational approximation to the ratio.
    """

    mean_tti_s: float
    median_tti_s: float
    blind_fraction: float
    closed_form_tti_s: float
    commensurate: bool
    ratio: float
    rational: tuple[int, int]


def _first_intercept_sweeps(
    tr: float, te: float, wr: float, we: float, phases: np.ndarray, max_sweeps: int
) -> np.ndarray:
    """Sweep index of the first intercept for each initial phase.

    Returns ``-1`` where no intercept occurs within ``max_sweeps``.
    """
    n = np.arange(max_sweeps, dtype=np.float64)
    # psi[p, n] = relative phase of sweep n given initial emitter phase p.
    psi = np.mod(n[None, :] * tr - phases[:, None], te)
    hit = (psi < we) | (psi > te - wr)
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), -1)
    return first


def probability_of_intercept(
    tr: float,
    te: float,
    wr: float,
    we: float,
    t_s: np.ndarray | float,
    n_phase: int = 1024,
    model: str = "deterministic",
) -> np.ndarray:
    """Probability of having intercepted by time ``t``, over random initial phase.

    Args:
        tr: Receiver revisit period, seconds.
        te: Emitter scan period, seconds.
        wr: Receiver dwell on the emitter's channel per revisit, seconds.
        we: Emitter illumination window per scan, seconds.
        t_s: Time(s) at which to evaluate ``POI``.
        n_phase: Number of initial emitter phases averaged over.
        model: ``"deterministic"`` for the true staircase, ``"exponential"`` for
            the common ``1 - exp(-t/tau)`` approximation.

    Returns:
        ``POI`` values in ``[0, 1]``, same shape as ``t_s``.
    """
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    if model == "exponential":
        return poi_exponential(tr, te, wr, we, t)

    max_sweeps = max(int(np.ceil(t.max() / tr)) + 1, 2)
    phases = np.linspace(0.0, te, n_phase, endpoint=False)
    first = _first_intercept_sweeps(tr, te, wr, we, phases, max_sweeps)
    t_first = np.where(first >= 0, first * tr, np.inf)
    return (t_first[None, :] <= t[:, None]).mean(axis=1)


def poi_exponential(
    tr: float, te: float, wr: float, we: float, t_s: np.ndarray | float
) -> np.ndarray:
    """The commonly assumed ``1 - exp(-t/tau)`` intercept model.

    Provided for comparison only. It is correct for a *randomly* retuning
    receiver and wrong for a periodic one, which is the entire point of this
    module: it cannot represent blindness at all, since it reaches 1 for every
    parameter combination.

    Args:
        tr: Receiver revisit period, seconds.
        te: Emitter scan period, seconds.
        wr: Receiver dwell, seconds.
        we: Emitter illumination window, seconds.
        t_s: Time(s) at which to evaluate.

    Returns:
        ``POI`` values in ``[0, 1]``.
    """
    tau = tr * te / max(wr + we, 1e-12)
    return 1.0 - np.exp(-np.asarray(t_s, dtype=np.float64) / tau)


def expected_time_to_intercept(tr: float, te: float, wr: float, we: float) -> float:
    """Classical closed-form mean time to intercept.

    ``E[TTI] ~ Tr * Te / (wr + we)`` -- valid when ``Tr/Te`` is incommensurate so
    the relative phase is equidistributed. It says nothing about the commensurate
    case, where the true answer can be infinity.

    Args:
        tr: Receiver revisit period, seconds.
        te: Emitter scan period, seconds.
        wr: Receiver dwell, seconds.
        we: Emitter illumination window, seconds.

    Returns:
        Expected time to intercept in seconds.
    """
    return tr * te / max(wr + we, 1e-12)


def three_distance_gaps(ratio: float, n_points: int = 200) -> np.ndarray:
    """Distinct gap lengths of ``{n * ratio mod 1}`` -- the three-distance theorem.

    For any irrational ``alpha`` and any ``N``, the points ``{n*alpha}`` for
    ``n = 1..N`` partition the unit circle into intervals of **at most three**
    distinct lengths. The largest of these bounds the worst-case phase we can
    fail to sample, so a sweep ratio whose largest gap is small is one that
    cannot lock out for long.

    Args:
        ratio: The ratio ``Tr / Te``.
        n_points: Number of sweeps considered.

    Returns:
        Sorted array of the distinct gap lengths present (up to floating-point
        tolerance).
    """
    pts = np.sort(np.mod(np.arange(1, n_points + 1) * ratio, 1.0))
    gaps = np.diff(np.concatenate([pts, [pts[0] + 1.0]]))
    return np.unique(np.round(gaps, 9))


def analyse_coincidence(
    tr: float,
    te: float,
    wr: float,
    we: float,
    horizon_s: float = 120.0,
    n_phase: int = 1024,
    max_denominator: int = 32,
) -> CoincidenceResult:
    """Full scan-on-scan analysis of one receiver/emitter period pair.

    Args:
        tr: Receiver revisit period, seconds.
        te: Emitter scan period, seconds.
        wr: Receiver dwell on the channel per revisit, seconds.
        we: Emitter illumination window per scan, seconds.
        horizon_s: How long to search before declaring a phase blind.
        n_phase: Number of initial emitter phases averaged over.
        max_denominator: Largest denominator considered "commensurate".

    Returns:
        A :class:`CoincidenceResult`.
    """
    ratio = tr / te
    frac = Fraction(ratio).limit_denominator(max_denominator)
    commensurate = abs(float(frac) - ratio) < 1e-9

    max_sweeps = max(int(np.ceil(horizon_s / tr)) + 1, 2)
    phases = np.linspace(0.0, te, n_phase, endpoint=False)
    first = _first_intercept_sweeps(tr, te, wr, we, phases, max_sweeps)
    t_first = np.where(first >= 0, first * tr, np.inf)
    ok = np.isfinite(t_first)

    return CoincidenceResult(
        mean_tti_s=float(t_first[ok].mean()) if ok.any() else float("inf"),
        median_tti_s=float(np.median(t_first[ok])) if ok.any() else float("inf"),
        blind_fraction=float(1.0 - ok.mean()),
        closed_form_tti_s=expected_time_to_intercept(tr, te, wr, we),
        commensurate=commensurate,
        ratio=float(ratio),
        rational=(frac.numerator, frac.denominator),
    )


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #
class CoprimeSweepScheduler(Scheduler):
    """Sweep whose revisit pattern is deliberately incommensurate with everything.

    Rather than a saw-tooth (whose revisit period is a single fixed ``Tr``, and
    which is therefore the policy most exposed to lockout), the visit order
    follows a **golden-ratio Weyl sequence**::

        a_n = floor(N_legal * frac(n * phi))

    The golden ratio is the worst-approximable irrational, so by the
    three-distance theorem this sequence has the smallest possible largest gap
    for any ``n`` -- it is the low-discrepancy sequence that maximises the
    minimum phase separation. No emitter period can stay in lockstep with it.

    When ``avoid_detected_periods`` is set and the belief has confidently
    estimated an emitter's scan period, the step is nudged away from any ratio
    that would be near-commensurate with it.

    Args:
        config: Resolved configuration.
        seed: Seed for the initial phase offset.
        name: Optional display name.
    """

    key = "coprime_sweep"
    needs_periods = True

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        cfg = config.agents.coprime_sweep
        self.irrational = {"golden": GOLDEN, "silver": SILVER}.get(cfg.irrationality, GOLDEN)
        self.avoid = cfg.avoid_detected_periods
        self._alpha = self.irrational % 1.0
        self._n = 0
        self._offset = float(self.rng.random())
        self._last_adapt = -10**9
        self.dwell = max(int(cfg.dwell_slots), 1)
        self._held = 0

    def reset(self) -> None:
        """Restart the Weyl sequence."""
        super().reset()
        self._n = 0
        self._alpha = self.irrational % 1.0

    def _adapt(self, belief: BeliefState) -> None:
        """Nudge the step away from resonance with any detected scan period."""
        conf = belief.period_confidence >= self.cfg.belief.period_min_confidence
        if not np.any(conf):
            return
        n_legal = self.legal_indices.size
        # Our effective revisit period, in slots, at the current step size.
        tr = n_legal * (1 + self.cfg.receiver.t_settle_slots)
        for te in belief.period_hat_slots[conf]:
            if te <= 0:
                continue
            ratio = tr / te
            frac = Fraction(float(ratio)).limit_denominator(12)
            if abs(float(frac) - ratio) < 0.02:
                # Near-commensurate: shift the Weyl step by a golden increment,
                # which cannot itself be commensurate with anything.
                self._alpha = (self._alpha + (GOLDEN - 1.0) / n_legal) % 1.0
                break

    def act(self, belief: BeliefState, t: int) -> int:
        """Return the next term of the (possibly adapted) Weyl sequence."""
        if self.avoid and t - self._last_adapt >= 500:
            self._adapt(belief)
            self._last_adapt = t
        n_legal = self.legal_indices.size
        pos = int(np.floor(n_legal * ((self._offset + self._n * self._alpha) % 1.0)))
        self._held += 1
        if self._held >= self.dwell:
            self._held = 0
            self._n += 1
        action = int(self.legal_indices[min(pos, n_legal - 1)])
        self.last_action = action
        return action


class PhaseLockedScheduler(Scheduler):
    """Predict the next beam arrival and park the receiver there just before it.

    Once a channel's scan period ``Te`` has been estimated with enough
    confidence, the next arrival is at ``t_last_hit + k*Te``. We park on that
    channel from ``t_arrival - guard`` where
    ``guard = guard_sigma * sigma_est + t_settle`` -- early enough to have
    finished settling, late enough not to waste dwell.

    Between predicted arrivals the receiver is not idle: it falls back to the
    configured policy (Whittle by default), so coverage continues while we wait.

    This is the payoff of the whole analysis chain: estimate the period, predict
    the phase, and be looking in the right place at the right moment instead of
    hoping to drift into coincidence.

    Args:
        config: Resolved configuration.
        seed: Seed forwarded to the fallback policy.
        name: Optional display name.
        fallback: Explicit fallback scheduler; built from config if omitted.
    """

    key = "phase_locked"
    needs_periods = True

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        name: str | None = None,
        fallback: Scheduler | None = None,
    ) -> None:
        super().__init__(config, seed, name)
        cfg = config.agents.phase_locked
        self.guard_sigma = cfg.guard_sigma
        self.min_conf = cfg.min_confidence
        self.refresh_slots = 200
        self._last_refresh = -10**9
        self.n_parks = 0

        if fallback is not None:
            self.fallback = fallback
        elif cfg.fallback == "whittle":
            from smartscan.agents.whittle import WhittleIndexScheduler

            self.fallback = WhittleIndexScheduler(config, seed)
        elif cfg.fallback == "thompson":
            from smartscan.agents.bandits import ThompsonSampling

            self.fallback = ThompsonSampling(config, seed)
        else:
            from smartscan.agents.baselines import SequentialSweep

            self.fallback = SequentialSweep(config, seed)

    def reset(self) -> None:
        """Reset both this policy and its fallback."""
        super().reset()
        self.fallback.reset()
        self._last_refresh = -10**9
        self.n_parks = 0

    def act(self, belief: BeliefState, t: int) -> int:
        """Park on an imminent predicted arrival, else defer to the fallback."""
        if t - self._last_refresh >= self.refresh_slots:
            belief.refresh_periods()
            self._last_refresh = t

        ttna = belief.time_to_next_arrival()
        conf = belief.period_confidence
        sigma = np.where(belief.period_sigma_slots > 0, belief.period_sigma_slots, 1.0)
        guard = self.guard_sigma * sigma + self.cfg.receiver.t_settle_slots

        imminent = (ttna <= guard) & (conf >= self.min_conf) & np.isfinite(ttna)
        if np.any(imminent):
            # Among imminent arrivals, prefer the one we believe is most valuable
            # and least like an interferer.
            value = np.where(imminent, conf * (1.0 - 0.9 * belief.interferer_score()), -np.inf)
            action = self.argmax_legal(self.window_value(np.where(imminent, value, 0.0)))
            self.n_parks += 1
            self.last_action = action
            self.fallback.last_action = action
            return action

        action = self.fallback.act(belief, t)
        self.last_action = action
        return action
