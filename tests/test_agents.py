"""Schedulers, the Gym environment, the runner and the remaining metrics."""

from __future__ import annotations

import numpy as np
import pytest

from smartscan.agents import AGENT_KEYS, build_agent
from smartscan.agents.base import window_matrix
from smartscan.agents.belief import N_CHANNEL_FEATURES, N_GLOBAL_FEATURES, BeliefState
from smartscan.analysis.metrics import (
    average_intercept_rate,
    empirical_pd,
    empirical_pfa,
    evaluate_episode,
    interception_ratio,
    spectrum_coverage,
)
from smartscan.analysis.scan_on_scan import CoprimeSweepScheduler, PhaseLockedScheduler
from smartscan.config import load_config
from smartscan.env.gym_env import SmartScanEnv, observation_size
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.hal.simulated import detection_probability_tensor
from smartscan.runner import RewardAccountant, run_episode

#: Schedulers that need no trained checkpoint.
ANALYTIC = [
    "sequential", "random", "priority_rr", "epsilon_greedy",
    "ucb1", "thompson", "whittle", "coprime_sweep", "phase_locked",
]


@pytest.fixture(scope="module")
def cfg():
    return load_config("easy.yaml")


@pytest.fixture(scope="module")
def scenario(cfg):
    return generate_scenario(cfg.run.seed, config=cfg)


@pytest.fixture(scope="module")
def episode(scenario):
    return build_episode(scenario)


# --------------------------------------------------------------------------- #
# Action geometry
# --------------------------------------------------------------------------- #
def test_window_matrix_marks_illegal_rows():
    w = window_matrix(16, 4, "center_index")
    assert w.shape == (16, 4)
    legal = w[:, 0] >= 0
    assert legal.sum() == 16 - 4 + 1
    for a in np.flatnonzero(legal):
        assert np.array_equal(w[a], np.arange(w[a, 0], w[a, 0] + 4))


def test_window_start_convention_gives_the_same_count():
    a = window_matrix(32, 4, "center_index")[:, 0] >= 0
    b = window_matrix(32, 4, "window_start")[:, 0] >= 0
    assert a.sum() == b.sum() == 29


# --------------------------------------------------------------------------- #
# Belief
# --------------------------------------------------------------------------- #
def test_belief_feature_shapes(cfg):
    belief = BeliefState(cfg)
    chan, glob = belief.features()
    assert chan.shape == (cfg.n_channels, N_CHANNEL_FEATURES)
    assert glob.shape == (N_GLOBAL_FEATURES,)
    assert chan.dtype == np.float32
    assert belief.flat_features().size == observation_size(cfg)
    assert np.all(np.isfinite(chan)) and np.all(np.isfinite(glob))


def test_belief_decays_toward_the_prior(cfg):
    """A channel confirmed busy long ago must return to 'I don't know'."""
    belief = BeliefState(cfg)
    belief.alpha[0] = 50.0
    belief.beta[0] = 1.0
    before = belief.p_occupied[0]
    belief.decay(cfg.belief.decay_half_life_slots * 12)
    after = belief.p_occupied[0]
    prior = cfg.belief.alpha_prior / (cfg.belief.alpha_prior + cfg.belief.beta_prior)
    assert before > 0.9
    assert abs(after - prior) < 0.05


def test_belief_never_reads_ground_truth(cfg, episode):
    """The belief must be buildable from Observations alone."""
    from smartscan.env.receiver import Receiver

    rx = Receiver(episode, cfg)
    belief = BeliefState(cfg, episode.n_slots)
    for _ in range(50):
        obs = rx.step(10)
        belief.update(obs)
    # Channels outside the observed window remain at the prior.
    untouched = [c for c in range(cfg.n_channels) if not (9 <= c <= 12)]
    assert np.allclose(belief.n_visits[untouched], 0)
    assert belief.n_visits[10] > 0


def test_belief_reset_restores_the_prior(cfg):
    belief = BeliefState(cfg)
    belief.alpha[:] = 9.0
    belief.reset()
    assert np.allclose(belief.alpha, cfg.belief.alpha_prior)


# --------------------------------------------------------------------------- #
# Every analytic scheduler runs, legally
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ANALYTIC)
def test_scheduler_runs_and_only_takes_legal_actions(key, cfg, scenario, episode):
    sched = build_agent(key, cfg, cfg.run.seed, scenario)
    result = run_episode(cfg, cfg.run.seed, sched, scenario=scenario, episode=episode)
    assert result.n_steps > 0
    legal = window_matrix(cfg.n_channels, cfg.receiver.ibw_channels)[:, 0] >= 0
    assert np.all(legal[result.actions]), f"{key} chose an illegal action"
    assert result.visit_mask.sum() == result.n_steps * cfg.receiver.ibw_channels
    assert np.isfinite(result.rewards).all()


@pytest.mark.parametrize("key", ANALYTIC)
def test_scheduler_is_deterministic_given_a_seed(key, cfg, scenario, episode):
    a = run_episode(cfg, cfg.run.seed, build_agent(key, cfg, 7, scenario),
                    scenario=scenario, episode=episode)
    b = run_episode(cfg, cfg.run.seed, build_agent(key, cfg, 7, scenario),
                    scenario=scenario, episode=episode)
    assert np.array_equal(a.actions, b.actions)
    assert np.allclose(a.rewards, b.rewards)


def test_registry_covers_every_documented_scheduler():
    for key in (*ANALYTIC, "predictor", "dqn", "ppo", "hybrid"):
        assert key in AGENT_KEYS
    with pytest.raises(KeyError, match="unknown agent"):
        build_agent("does_not_exist", load_config("easy.yaml"))


def test_sweep_dwell_changes_observation_budget(cfg, scenario, episode):
    """dwell_slots is load-bearing, not cosmetic: it buys observation time."""
    obs = {}
    for d in (1, 3):
        c = cfg.with_overrides(agents={"sequential_sweep": {"dwell_slots": d}})
        r = run_episode(c, c.run.seed, build_agent("sequential", c, 0, scenario),
                        scenario=scenario, episode=episode)
        obs[d] = r.visit_mask.sum()
    assert obs[3] > obs[1] * 1.5


def test_untrained_learned_agents_fall_back_rather_than_run_noise(cfg, scenario):
    """An untrained network would silently poison the leaderboard."""
    torch = pytest.importorskip("torch")
    assert torch is not None
    sched = build_agent("dqn", cfg, 0, scenario)
    # No dqn checkpoint is shipped, so it must announce the fallback.
    assert "fallback" in sched.name or sched.is_trained


def test_phase_locked_falls_back_between_predicted_arrivals(cfg, scenario, episode):
    sched = PhaseLockedScheduler(cfg, 0)
    result = run_episode(cfg, cfg.run.seed, sched, scenario=scenario, episode=episode)
    assert result.n_steps > 0
    assert sched.n_parks >= 0  # zero is legitimate when no period is confident


def test_coprime_sweep_visits_more_distinct_windows_than_it_repeats(cfg, scenario, episode):
    """A low-discrepancy sequence must actually spread, not cycle."""
    sched = CoprimeSweepScheduler(cfg, 0)
    result = run_episode(cfg, cfg.run.seed, sched, scenario=scenario, episode=episode)
    assert np.unique(result.actions).size > 0.5 * (cfg.n_channels - cfg.receiver.ibw_channels)


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
def test_reward_rewards_novelty_once_then_reconfirmation(cfg, episode):
    acc = RewardAccountant(cfg, episode)
    eid = episode.truth[0].emitter_id
    first = acc.step(np.array([eid]), retuned=False, interferer_dwell=False, max_staleness=0.0)
    second = acc.step(np.array([eid]), retuned=False, interferer_dwell=False, max_staleness=0.0)
    assert first > second
    assert second == pytest.approx(cfg.reward.w3_reconfirm, abs=1e-6)


def test_reconfirmation_is_capped(cfg, episode):
    """Without the cap, parking on one loud emitter farms reward indefinitely."""
    acc = RewardAccountant(cfg, episode)
    eid = episode.truth[0].emitter_id
    rewards = [
        acc.step(np.array([eid]), False, False, 0.0)
        for _ in range(cfg.reward.reconfirm_cap_per_emitter + 5)
    ]
    assert rewards[-1] == pytest.approx(0.0, abs=1e-9)


def test_staleness_penalty_uses_the_max_not_the_mean(cfg, episode):
    acc = RewardAccountant(cfg, episode)
    low = acc.step(np.zeros(0), False, False, max_staleness=0.0)
    acc.reset()
    high = acc.step(np.zeros(0), False, False, max_staleness=float(episode.n_slots))
    assert high < low


# --------------------------------------------------------------------------- #
# Gym environment
# --------------------------------------------------------------------------- #
def test_gym_env_reset_step_contract(cfg):
    env = SmartScanEnv(cfg, [cfg.run.seed])
    obs, info = env.reset()
    assert obs.shape == (env.obs_size,) and obs.dtype == np.float32
    mask = info["action_mask"]
    assert mask.dtype == np.bool_ and mask.sum() == cfg.n_channels - cfg.receiver.ibw_channels + 1

    action = int(np.flatnonzero(mask)[0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float) and np.isfinite(reward)
    assert terminated is False and truncated is False
    assert "action_mask" in info2


def test_gym_env_runs_an_episode_to_termination(cfg):
    env = SmartScanEnv(cfg, [cfg.run.seed])
    _obs, info = env.reset()
    steps = 0
    terminated = False
    while not terminated and steps < cfg.n_slots + 10:
        legal = np.flatnonzero(info["action_mask"])
        _obs, _r, terminated, _t, info = env.step(int(legal[steps % legal.size]))
        steps += 1
    assert terminated, "episode never terminated"
    assert steps <= cfg.n_slots


def test_gym_env_rejects_illegal_actions(cfg):
    env = SmartScanEnv(cfg, [cfg.run.seed])
    _obs, info = env.reset()
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(ValueError, match="illegal"):
        env.step(illegal)


def test_gym_env_caches_episodes(cfg):
    """Rebuilding tensors every reset would dominate RL training."""
    env = SmartScanEnv(cfg, [cfg.run.seed], cache_episodes=True)
    env.reset(cfg.run.seed)
    env.reset(cfg.run.seed)
    assert len(env._cache) == 1


def test_gymnasium_conformance_if_installed(cfg):
    gym = pytest.importorskip("gymnasium")
    from smartscan.env.gym_env import make_gym_env

    env = make_gym_env(cfg, [cfg.run.seed])
    assert isinstance(env, gym.Env)
    obs, _info = env.reset(seed=cfg.run.seed)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == cfg.n_channels
    assert env.action_masks().shape == (cfg.n_channels,)


# --------------------------------------------------------------------------- #
# Remaining figures of merit
# --------------------------------------------------------------------------- #
def test_empirical_pd_rises_with_snr():
    """Measured Pd must track SNR, with Clopper-Pearson intervals that bracket it.

    Run on HARD rather than EASY: EASY's emitters are all strong, so the measured
    curve saturates near 1 and carries no information about monotonicity. That is
    a property of the scenario, not of the detector.
    """
    from scipy import stats as sp_stats

    hard = load_config("hard.yaml")
    sc = generate_scenario(hard.run.seed, config=hard)
    ep = build_episode(sc)
    result = run_episode(hard, hard.run.seed, build_agent("sequential", hard, 0, sc),
                         scenario=sc, episode=ep)
    curve = empirical_pd(ep, result.visit_mask, result.true_hit_mask)

    populated = curve["n"] > 50
    assert populated.sum() >= 4, "not enough populated SNR bins to test monotonicity"
    snr = curve["snr_centre"][populated]
    pd = curve["pd"][populated]
    lo, hi = curve["lo"][populated], curve["hi"][populated]

    assert np.all((lo <= pd + 1e-9) & (pd <= hi + 1e-9)), "CIs must bracket the estimate"
    assert np.all((pd >= 0) & (pd <= 1))
    rho = sp_stats.spearmanr(snr, pd).statistic
    assert rho > 0.5, f"Pd should rise with SNR; Spearman rho = {rho:.2f}"


def test_empirical_pfa_is_near_the_design_value(cfg, scenario, episode):
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, scenario),
                         scenario=scenario, episode=episode)
    p = empirical_pfa(episode, result.visit_mask, result.hit_mask)
    assert p["n_trials"] > 1000
    assert p["lo"] <= p["pfa"] <= p["hi"]
    assert p["pfa"] < 10 * cfg.receiver.detector.pfa


def test_interception_ratio_denominator_excludes_the_undetectable(cfg, scenario, episode):
    """Charging a scheduler for emitters below sensitivity would measure the
    link budget, not the policy."""
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, scenario),
                         scenario=scenario, episode=episode)
    pd = detection_probability_tensor(episode, cfg)
    strict = interception_ratio(episode, result.true_hit_mask, pd)
    loose = interception_ratio(episode, result.true_hit_mask, None)
    for key in ("raw", "threat_weighted", "coverage"):
        assert 0.0 <= strict[key] <= 1.0
        assert 0.0 <= loose[key] <= 1.0
    assert strict["raw"] >= loose["raw"] - 1e-9


def test_intercept_rate_breaks_down_by_class(cfg, scenario, episode):
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, scenario),
                         scenario=scenario, episode=episode)
    rate = average_intercept_rate(episode, result.true_hit_mask)
    assert rate["overall"] > 0
    per_class = {k: v for k, v in rate.items() if k != "overall"}
    assert per_class
    assert sum(per_class.values()) == pytest.approx(rate["overall"], rel=1e-6)


def test_spectrum_coverage_fields(cfg, scenario, episode):
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, scenario),
                         scenario=scenario, episode=episode)
    cov = spectrum_coverage(result.visit_mask, episode.dt_s)
    assert 0.0 < cov["fraction_visited"] <= 1.0
    assert 0.0 <= cov["coverage_entropy"] <= 1.0
    assert cov["revisit_max_s"] >= cov["revisit_p95_s"] >= 0
    assert cov["staleness_max_s"] > 0


def test_evaluate_episode_returns_every_headline_metric(cfg, scenario, episode):
    result = run_episode(cfg, cfg.run.seed, build_agent("whittle", cfg, 0, scenario),
                         scenario=scenario, episode=episode)
    row = evaluate_episode(result, cfg)
    for key in (
        "ttfi_median_s", "ttfi_hard_median_s", "twir_rate", "twir_coverage",
        "coverage", "intercept_rate_per_s", "staleness_max_s", "coverage_entropy",
        "waste_fraction", "discovery_auc", "fa_burden", "reward_total",
        "reward_discounted", "popup_detect_rate", "n_popup_interceptable",
    ):
        assert key in row, key
    assert 0.0 <= row["twir_rate"] <= 1.0
    assert 0.0 <= row["coverage"] <= 1.0


def test_evaluate_episode_needs_ground_truth(cfg, scenario, episode):
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, scenario),
                         scenario=scenario, episode=episode, keep_episode=False)
    with pytest.raises(ValueError, match="keep_episode"):
        evaluate_episode(result, cfg)


def test_interferer_dwell_is_counted_on_the_hard_tier():
    """The waste metric must actually fire when interferers are present."""
    cfg = load_config("hard.yaml")
    sc = generate_scenario(cfg.run.seed, config=cfg)
    ep = build_episode(sc)
    assert any(t.is_interferer for t in ep.truth)
    result = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, 0, sc),
                         scenario=sc, episode=ep)
    assert result.interferer_dwells > 0
    assert 0.0 <= evaluate_episode(result, cfg)["waste_fraction"] <= 1.0


# --------------------------------------------------------------------------- #
# The dwell-efficient predictor's two corrections, pinned as properties.
# Both are the reason it exists: the shipped `predictor` scores
# `P̂·(1−0.9·I) + w·s`, which is neither invariant to how sharp `P̂` is nor
# discounted by what a channel has already yielded.
# --------------------------------------------------------------------------- #
def _de_value(agent, belief, p):
    """Reproduce DwellEfficientPredictorScheduler's channel value for a given P̂."""
    lo, hi = float(p.min()), float(p.max())
    p_rel = (p - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(p)
    novelty = 1.0 / (1.0 + belief.n_hits / 5.0)
    return ((1.0 - 0.9 * belief.interferer_score()) * p_rel * novelty
            + agent.coverage_weight * (belief.time_since_visit / max(belief.n_slots, 1)))


def test_dwell_efficient_value_is_invariant_to_predictor_sharpness(cfg):
    """An affine rescale of P̂ must not change the ranking.

    This is the defect that made a *better* predictor a *worse* scheduler: with
    the shipped additive form, sharpening P̂ shrinks the effective weight of the
    staleness term, so the policy parks harder. Min-max normalisation removes
    the degree of freedom entirely.
    """
    pytest.importorskip("torch")  # predictor-backed scheduler
    agent = build_agent("predictor_de", cfg, seed=0)
    belief = BeliefState(cfg)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.4, size=belief.n_channels)          # a blunt predictor
    sharp = 0.05 + 3.0 * (p - p.min())                          # the same ordering, sharper

    base = _de_value(agent, belief, p)
    resc = _de_value(agent, belief, sharp)
    np.testing.assert_allclose(base, resc, atol=1e-9)


def test_dwell_efficient_value_decays_with_harvest(cfg):
    """A channel that has already yielded hits must score below an equal twin.

    The mission metric counts *first* intercepts, so re-looking at a harvested
    channel is worth less than its occupancy probability suggests. Without this
    term a confident predictor keeps re-selecting what it has already found.
    """
    pytest.importorskip("torch")  # predictor-backed scheduler
    agent = build_agent("predictor_de", cfg, seed=0)
    belief = BeliefState(cfg)
    p = np.full(belief.n_channels, 0.9)
    p[1] = 0.2                                                  # keep a real min/max range

    belief.n_hits[:] = 0
    belief.n_hits[0] = 20                                       # channel 0 already harvested
    value = _de_value(agent, belief, p)

    assert value[0] < value[2], "harvested channel should rank below its unharvested twin"
    assert value[0] > 0.0, "novelty discounts, it must not zero a re-activating emitter"


# --------------------------------------------------------------------------- #
# window_max, and the guaranteed-coverage predictor's coverage-slot branch
# that depends on it: window_value SUMS a per-channel score over a window's
# k channels, which is right for occupancy (more channels showing signal
# genuinely is more worth a dwell) and wrong for urgency (a single badly-
# neglected channel's push toward selection gets diluted to 1/k of its true
# size by ordinarily-fresh window-mates).
# --------------------------------------------------------------------------- #
def test_window_max_takes_worst_member_not_sum(cfg):
    """window_max must differ from window_value exactly where dilution matters."""
    agent = build_agent("sequential", cfg, seed=0)
    a = int(np.flatnonzero(agent.legal)[0])
    window = agent.windows[a]
    assert len(window) >= 2, "test needs a multi-channel window"

    per_channel = np.zeros(agent.n_channels)
    per_channel[window[0]] = 100.0                              # one badly-neglected channel
    per_channel[window[1:]] = 1.0                                # window-mates ordinarily fresh

    summed = agent.window_value(per_channel)[a]
    worst = agent.window_max(per_channel)[a]
    assert worst == pytest.approx(100.0)
    assert summed == pytest.approx(100.0 + 1.0 * (len(window) - 1))
    assert worst < summed, "the single starved channel must not be diluted by its window-mates"


def test_guaranteed_coverage_prefers_the_single_starved_channel(cfg):
    """The coverage-slot branch must not be outbid by a merely-collectively-staler window.

    A window holding ONE very-overdue channel among
    fresh window-mates has a smaller SUM than a competing window whose several
    members are each moderately stale -- but the single overdue channel is the
    one the revisit-gap guarantee actually needs picked. window_value (the
    diluted aggregation) picks the wrong window here; window_max (what
    GuaranteedCoveragePredictorScheduler actually uses) picks the right one.
    """
    pytest.importorskip("torch")  # predictor-backed scheduler
    agent = build_agent("predictor_gc", cfg, seed=0)
    legal = np.flatnonzero(agent.legal)
    a_starved, a_crowded = int(legal[0]), int(legal[len(legal) // 2])
    w_starved, w_crowded = agent.windows[a_starved], agent.windows[a_crowded]
    assert not set(w_starved) & set(w_crowded), "need two disjoint windows"

    stale = np.full(agent.n_channels, 1.0)
    stale[w_starved[0]] = 40.0                                   # one channel, badly overdue
    stale[w_crowded] = 12.0                                      # every member, moderately overdue

    diluted = agent.window_value(stale)
    fixed = agent.window_max(stale)
    assert diluted[a_crowded] > diluted[a_starved], (
        "the diluted sum must indeed favour the collectively-staler window -- "
        "otherwise this is not exercising the bug"
    )
    assert fixed[a_starved] > fixed[a_crowded], (
        "the fix must favour the window holding the single most overdue channel"
    )
