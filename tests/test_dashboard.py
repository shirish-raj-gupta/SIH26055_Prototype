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
    assert len(track.actions) == 50
    assert len(track.rewards) == 50
    assert track.t > 0
    assert track.visit_mask.sum() == 50 * cfg.receiver.ibw_channels
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
    # ...but different behaviour.
    assert not np.array_equal(np.asarray(a.actions), np.asarray(b.actions))


def test_waterfall_builds_a_figure(setup):
    cfg, scenario, episode, pd_tensor = setup
    track = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, 120, interferers=set())

    fig = app._waterfall(track, cfg, episode, pd_tensor, "test")
    assert fig.data, "waterfall produced no traces"
    assert fig.layout.xaxis.title.text == "time (s)"
    assert fig.layout.yaxis.title.text == "channel"


def test_track_stops_cleanly_at_the_horizon(setup):
    """Over-running the episode must end the demo, not raise on stage."""
    cfg, scenario, episode, _pd = setup
    track = app._new_track("sequential", cfg, scenario, episode, cfg.run.seed)
    app._advance(track, cfg, episode.n_slots + 500, interferers=set())
    assert track.done
    n = len(track.actions)
    app._advance(track, cfg, 50, interferers=set())
    assert len(track.actions) == n, "advancing a finished track must be a no-op"
