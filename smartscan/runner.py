"""Episode execution: drive a scheduler through an episode and record everything.

This is the single place where an environment, a receiver, a belief and a policy
are wired together, so every scheduler is evaluated through exactly the same
loop. The recorded :class:`EpisodeResult` carries enough to compute every figure
of merit in :mod:`smartscan.analysis.metrics` without replaying anything.

Reward uses ground truth. That is legitimate -- it is a *training* signal
available in simulation -- and it is confined to this module. The belief, and
therefore every policy, still sees observations alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config
from smartscan.env.receiver import Receiver
from smartscan.env.rf_environment import Scenario, build_episode, generate_scenario
from smartscan.env.types import EpisodeTensors

__all__ = ["EpisodeResult", "RewardAccountant", "run_episode"]


@dataclass
class EpisodeResult:
    """Everything one episode produced.

    Attributes:
        agent: Scheduler key.
        seed: Scenario seed.
        actions: Chosen action per decision, shape ``(n_steps,)``.
        dwell_slots: Slot at which each dwell was observed, shape ``(n_steps,)``.
        visit_mask: ``(B, T)`` bool, channel observed in that slot.
        hit_mask: ``(B, T)`` bool, detection declared.
        true_hit_mask: ``(B, T)`` bool, declared detection that was genuine.
        window_lo: First observed channel per decision, shape ``(n_steps,)``.
        snr_est_db: The receiver's REPORTED SNR per decision, shape
            ``(n_steps, K)``, ``nan`` where no hit was declared. This is the
            noisy estimate a real receiver would produce -- not the true SNR --
            so anything downstream that consumes it stays deployable.
        rewards: Per-decision reward, shape ``(n_steps,)``.
        first_intercept: Slot of first true detection per emitter id (``-1`` if
            never), indexed by emitter id.
        n_detections: True detections per emitter id.
        n_retunes: Number of frequency changes.
        settle_slots_lost: Slots lost to LO settling.
        interferer_dwells: Dwells spent on a channel whose strongest emitter is
            an interferer.
        staleness_max: Running maximum of ``time_since_visit`` over the episode.
        wall_time_s: Wall-clock execution time.
        episode: The ground-truth tensors (kept for metrics and plotting).
        belief: The final belief state.
    """

    agent: str
    seed: int
    actions: np.ndarray
    dwell_slots: np.ndarray
    visit_mask: np.ndarray
    hit_mask: np.ndarray
    true_hit_mask: np.ndarray
    window_lo: np.ndarray
    snr_est_db: np.ndarray
    rewards: np.ndarray
    first_intercept: dict[int, int]
    n_detections: dict[int, int]
    n_retunes: int
    settle_slots_lost: int
    interferer_dwells: int
    staleness_max: float
    wall_time_s: float
    episode: EpisodeTensors | None = None
    belief: BeliefState | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of decisions taken."""
        return int(self.actions.size)

    @property
    def total_reward(self) -> float:
        """Undiscounted return."""
        return float(self.rewards.sum())

    def snr_plane(self) -> np.ndarray:
        """Rebuild a ``(B, T)`` plane of the receiver's reported SNR.

        Only cells the receiver actually observed and declared a hit on carry a
        value; everything else is zero. Built from the reported estimate, never
        from ground truth, so a model trained on it is deployable.

        Returns:
            Float32 array of shape ``(B, T)``.
        """
        plane = np.zeros(self.hit_mask.shape, dtype=np.float32)
        k = self.snr_est_db.shape[1] if self.snr_est_db.size else 0
        for i in range(self.n_steps):
            lo, t = int(self.window_lo[i]), int(self.dwell_slots[i])
            plane[lo : lo + k, t] = np.nan_to_num(self.snr_est_db[i], nan=0.0)
        return plane

    def discounted_return(self, gamma: float = 0.99) -> float:
        """Discounted return.

        Args:
            gamma: Discount factor.

        Returns:
            ``sum_t gamma**t * r_t``.
        """
        g = np.power(gamma, np.arange(self.rewards.size))
        return float((self.rewards * g).sum())


class RewardAccountant:
    """Computes the configured reward signal for each dwell.

    ``r_t = w1*threat  + w2*novelty + w3*reconfirm
            - w4*retune - w5*interferer_dwell - w6*max_staleness/T``

    Two details that matter (``docs/architecture.md`` §11.4):

    * ``w6`` penalises the **maximum** staleness over channels, not the mean. A
      mean can be gamed by starving a few channels; a max cannot, so this is a
      genuine coverage guarantee rather than an average.
    * ``w3`` is capped per emitter by ``reconfirm_cap_per_emitter``. Without the
      cap an agent farms re-confirmation reward by parking on one loud emitter,
      which is the most obvious reward hack this objective admits.

    Args:
        config: Resolved configuration.
        episode: Ground-truth tensors.
    """

    def __init__(self, config: Config, episode: EpisodeTensors) -> None:
        self.cfg = config.reward
        self.n_slots = episode.n_slots
        self.threat = {t.emitter_id: t.threat_priority for t in episode.truth}
        self.is_interferer = {t.emitter_id: t.is_interferer for t in episode.truth}
        self.seen: set[int] = set()
        self.reconfirms: dict[int, int] = {}

    def reset(self) -> None:
        """Forget which emitters have been seen."""
        self.seen.clear()
        self.reconfirms.clear()

    def step(
        self, detected_ids: np.ndarray, retuned: bool, interferer_dwell: bool, max_staleness: float
    ) -> float:
        """Score one dwell.

        Args:
            detected_ids: Ground-truth emitter ids genuinely detected this dwell.
            retuned: Whether the receiver changed frequency.
            interferer_dwell: Whether the dwell sat on a believed interferer.
            max_staleness: Current ``max_b time_since_visit[b]``, in slots.

        Returns:
            The scalar reward.
        """
        c = self.cfg
        r = 0.0
        for eid in np.unique(detected_ids):
            eid = int(eid)
            if eid <= 0:
                continue
            if eid not in self.seen:
                self.seen.add(eid)
                r += c.w1_threat_intercept * self.threat.get(eid, 0.5) + c.w2_novelty
            else:
                n = self.reconfirms.get(eid, 0)
                if n < c.reconfirm_cap_per_emitter:
                    self.reconfirms[eid] = n + 1
                    r += c.w3_reconfirm
        if retuned:
            r -= c.w4_retune
        if interferer_dwell:
            r -= c.w5_interferer_dwell
        stale_term = c.w6_staleness * (max_staleness / max(self.n_slots, 1))
        if c.normalise_by_episode:
            # Divide by the episode length so the term integrates to roughly
            # w6 * mean(max_staleness / T) rather than growing with n_steps.
            stale_term /= max(self.n_slots, 1)
        r -= stale_term
        return r


def run_episode(
    config: Config,
    seed: int,
    scheduler: Scheduler | Callable[[Config, int, Scenario], Scheduler],
    *,
    scenario: Scenario | None = None,
    episode: EpisodeTensors | None = None,
    keep_episode: bool = True,
    period_refresh_slots: int = 500,
) -> EpisodeResult:
    """Run one scheduler through one episode.

    Args:
        config: Resolved configuration.
        seed: Scenario seed. The same seed gives every scheduler the identical
            world *and* the identical detection luck (common random numbers), so
            comparisons are paired by construction.
        scheduler: A :class:`Scheduler`, or a factory
            ``(config, seed, scenario) -> Scheduler``.
        scenario: Pre-built scenario; generated from ``seed`` if omitted.
        episode: Pre-built tensors; built from ``scenario`` if omitted.
        keep_episode: Retain ground truth on the result (needed for metrics and
            plots; set ``False`` in RL rollouts to save memory).
        period_refresh_slots: Cadence of belief period re-estimation. A
            Lomb-Scargle per channel per slot would dominate runtime and the
            estimate cannot meaningfully change that fast.

    Returns:
        The populated :class:`EpisodeResult`.
    """
    t_start = time.perf_counter()
    scenario = scenario or generate_scenario(seed, config=config)
    episode = episode or build_episode(scenario)
    receiver = Receiver(episode, config, seed=seed)
    belief = BeliefState(config, episode.n_slots)

    sched = scheduler(config, seed, scenario) if callable(scheduler) and not isinstance(
        scheduler, Scheduler
    ) else scheduler
    sched.reset()

    b, t_max = config.n_channels, episode.n_slots
    visit_mask = np.zeros((b, t_max), dtype=bool)
    hit_mask = np.zeros((b, t_max), dtype=bool)
    true_hit_mask = np.zeros((b, t_max), dtype=bool)

    interferer_channels = {
        int(tr.home_channel) for tr in episode.truth if tr.is_interferer
    }
    accountant = RewardAccountant(config, episode)

    actions: list[int] = []
    dwells: list[int] = []
    window_los: list[int] = []
    snr_reports: list[np.ndarray] = []
    rewards: list[float] = []
    first_intercept: dict[int, int] = {}
    n_detections: dict[int, int] = {}
    staleness_max = 0.0
    last_action: int | None = None
    wants_periods = (
        getattr(sched, "needs_periods", False) and config.belief.period_estimator != "none"
    )
    next_period_refresh = period_refresh_slots

    while not receiver.done:
        action = int(sched.act(belief, receiver.t))
        belief.note_action(action)
        obs = receiver.step(action)

        lo, hi = obs.window
        td = obs.t
        visit_mask[lo:hi, td] = True
        hit_mask[lo:hi, td] = obs.hits
        genuine = obs.hits & ~obs.pfa_flags
        true_hit_mask[lo:hi, td] = genuine

        detected_ids = obs.truth_ids[genuine]
        for eid in detected_ids:
            eid = int(eid)
            if eid <= 0:
                continue
            first_intercept.setdefault(eid, td)
            n_detections[eid] = n_detections.get(eid, 0) + 1

        belief.update(obs)
        sched.observe(obs)

        stale = float(belief.time_since_visit.max())
        staleness_max = max(staleness_max, stale)
        retuned = last_action is None or action != last_action
        on_interferer = bool(interferer_channels & set(range(lo, hi)))
        rewards.append(accountant.step(detected_ids, retuned, on_interferer, stale))

        actions.append(action)
        dwells.append(td)
        window_los.append(lo)
        snr_reports.append(np.asarray(obs.snr_est_db, dtype=np.float32))
        last_action = action

        if wants_periods and receiver.t >= next_period_refresh:
            belief.refresh_periods()
            next_period_refresh += period_refresh_slots

    interferer_dwells = int(
        sum(1 for a in actions if interferer_channels & set(range(*receiver.window_of(a))))
    )

    return EpisodeResult(
        agent=sched.key,
        seed=int(seed),
        actions=np.asarray(actions, dtype=np.int32),
        dwell_slots=np.asarray(dwells, dtype=np.int32),
        visit_mask=visit_mask,
        hit_mask=hit_mask,
        true_hit_mask=true_hit_mask,
        window_lo=np.asarray(window_los, dtype=np.int16),
        snr_est_db=(
            np.stack(snr_reports).astype(np.float32)
            if snr_reports
            else np.zeros((0, config.receiver.ibw_channels), dtype=np.float32)
        ),
        rewards=np.asarray(rewards, dtype=np.float64),
        first_intercept=first_intercept,
        n_detections=n_detections,
        n_retunes=receiver.n_retunes,
        settle_slots_lost=receiver.settle_slots_lost,
        interferer_dwells=interferer_dwells,
        staleness_max=staleness_max,
        wall_time_s=time.perf_counter() - t_start,
        episode=episode if keep_episode else None,
        belief=belief,
    )
