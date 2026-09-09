"""The demo dashboard's simulation logic.

Streamlit itself is not exercised here -- what matters is that the panels are
driven by the same physics as the benchmark, and that a live run agrees with
``run_episode``. A demo that quietly disagrees with the reported numbers is
worse than no demo.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from dashboard import app
from smartscan.agents import build_agent
from smartscan.config import load_config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.hal.simulated import detection_probability_tensor
from smartscan.runner import run_episode


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("easy.yaml")
    scenario = generate_scenario(cfg.run.seed, config=cfg)
    episode = build_episode(scenario)
    pd_tensor = detection_probability_tensor(episode, cfg)
    return cfg, scenario, episode, pd_tensor


def test_every_offered_agent_exists_in_the_registry():
    """A dropdown entry that cannot be built is a demo that dies on stage."""
    from smartscan.agents import AGENT_KEYS

    for key in app.AGENT_LABELS:
        assert key in AGENT_KEYS, f"dashboard offers unknown agent {key!r}"


def test_track_advances_and_records(setup):
    cfg, scenario, episode, _pd = setup
    track = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    assert track.t == 0 and not track.done

    app._advance(track, cfg, 50, interferers=set())
    # 50 *slots*, not 50 dwells: a dwell spans several slots and a retune costs
    # more on top, so the dwell count is an outcome here, not the budget.
    n = len(track.actions)
    assert n > 0
    assert len(track.rewards) == n, "a reward was recorded without an action"
    assert track.t >= 50, f"advanced {track.t} slots, asked for 50"
    assert track.visit_mask.sum() == n * cfg.receiver.ibw_channels
    assert np.isfinite(track.total_reward)


def test_live_run_matches_run_episode(setup):
    """The dashboard must not be a second, divergent implementation.

    Same seed, same scenario, same scheduler: the actions the panel shows have
    to be the actions the benchmark would have recorded.
    """
    cfg, scenario, episode, _pd = setup
    track = app._new_track("whittle", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, 400, interferers=set())

    reference = run_episode(
        cfg, cfg.run.seed, build_agent("whittle", cfg, cfg.run.seed, scenario),
        scenario=scenario, episode=episode,
    )
    n = len(track.actions)
    assert np.array_equal(np.asarray(track.actions), reference.actions[:n])
    assert np.allclose(np.asarray(track.rewards), reference.rewards[:n])


def test_metrics_are_bounded_and_sane(setup):
    cfg, scenario, episode, pd_tensor = setup
    track = app._new_track("ucb1", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, 600, interferers=set())

    m = app._metrics(track, cfg, episode, pd_tensor)
    assert 0 <= m["found"] <= m["total"]
    assert 0.0 <= m["twir"] <= 1.0
    assert 0.0 <= m["pd"] <= 1.0
    assert 0.0 <= m["pfa"] <= 1.0
    assert 0.0 <= m["coverage"] <= 1.0
    assert np.isnan(m["ttfi_s"]) or m["ttfi_s"] >= 0.0


def test_reasoning_string_is_populated_and_names_the_window(setup):
    """Explainability is the point of the bottom panel; it must say something."""
    cfg, scenario, episode, _pd = setup
    track = app._new_track("whittle", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, 200, interferers=set())

    assert track.last_reason
    assert track.last_reason.startswith("ch ")
    assert any(
        token in track.last_reason
        for token in ("P(active)", "stale", "beam due", "exploring")
    )


def test_ab_mode_gives_both_tracks_identical_conditions(setup):
    """The A/B claim rests on this: same world, same luck, different policy."""
    cfg, scenario, episode, _pd = setup
    a = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    b = app._new_track("whittle", cfg, scenario, episode, cfg.run.seed)

    # Identical detection realisation -- common random numbers.
    assert np.array_equal(a.receiver.backend.declared, b.receiver.backend.declared)
    assert np.array_equal(a.receiver.backend.true_hit, b.receiver.backend.true_hit)

    app._advance(a, cfg, 300, interferers=set())
    app._advance(b, cfg, 300, interferers=set())

    # Same elapsed *time*. This is the assertion the original A/B test was
    # missing: it checked the world and the luck but never the clock, so the
    # panel spent its life advancing both by a fixed dwell count. Because a
    # retune costs t_settle slots on top of the dwell, that quietly handed the
    # restless policy ~74% more of the episode than the incumbent (0.582 s
    # against 0.334 s at 200 dwells) under a caption promising identical
    # conditions. Skew is now bounded by one atomic dwell.
    tolerance = cfg.receiver.t_settle_slots + 4
    assert abs(a.t - b.t) <= tolerance, f"tracks desynchronised: {a.t} vs {b.t}"

    # ...but different behaviour, and a different number of dwells inside the
    # same time, which is precisely the cost the incumbent does not pay.
    assert not np.array_equal(np.asarray(a.actions), np.asarray(b.actions))
    assert len(a.actions) != len(b.actions)


def test_waterfall_builds_a_figure(setup):
    cfg, scenario, episode, pd_tensor = setup
    track = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, 120, interferers=set())

    x_max = app._x_max(track.t * cfg.time.dt_s, cfg.time.episode_s)
    fig = app._waterfall(track, cfg, episode, pd_tensor, "test", x_max)
    assert fig.data, "waterfall produced no traces"
    assert fig.layout.xaxis.title.text == "time (s)"
    assert fig.layout.yaxis.title.text == "channel"
    assert fig.layout.xaxis.range == (0, x_max)


def test_axis_window_never_lags_the_clock(setup):
    """The window may lead the elapsed time but must never trail it.

    A window shorter than the clock crops the right-hand edge of the
    waterfall, hiding the most recent intercepts -- the ones an audience is
    actually watching for -- with no visible sign anything is missing.
    """
    cfg, _, _, _ = setup
    episode_s = cfg.time.episode_s
    for i in range(0, 1001):
        elapsed = episode_s * i / 1000.0
        x = app._x_max(elapsed, episode_s)
        assert x + 1e-9 >= elapsed, f"axis {x} crops elapsed {elapsed}"
        assert x <= episode_s + 1e-9, f"axis {x} overruns the episode"

    assert app._x_max(0.0, episode_s) > 0.0, "a zero-width axis cannot render"
    assert app._x_max(episode_s, episode_s) == pytest.approx(episode_s)
    # Past the horizon the axis must pin, not keep growing.
    assert app._x_max(episode_s * 3, episode_s) == pytest.approx(episode_s)


def test_axis_window_is_monotonic_and_steps_rather_than_slides(setup):
    """It grows, never shrinks, and changes only a handful of times."""
    cfg, _, _, _ = setup
    episode_s = cfg.time.episode_s
    seen = [app._x_max(episode_s * i / 500.0, episode_s) for i in range(501)]

    assert seen == sorted(seen), "axis window went backwards"
    # Ten tenths: a window that changed every frame would be far larger.
    assert len(set(seen)) <= 11, f"axis relabels too often: {sorted(set(seen))}"


def test_track_stops_cleanly_at_the_horizon(setup):
    """Over-running the episode must end the demo, not raise on stage."""
    cfg, scenario, episode, _pd = setup
    track = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, episode.n_slots + 500, interferers=set())
    assert track.done
    n = len(track.actions)
    app._advance(track, cfg, 50, interferers=set())
    assert len(track.actions) == n, "advancing a finished track must be a no-op"
