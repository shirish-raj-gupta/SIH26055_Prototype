"""Figures must not lie by omission.

A plot that renders titled, axed and empty is worse than one that fails: it
looks like a result. F2 did exactly that on EASY for as long as it existed,
because its default class filter selects scanning radars and EASY has none.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from smartscan.agents import build_agent
from smartscan.analysis.metrics import HARD_CLASSES
from smartscan.config import load_config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.eval import plots
from smartscan.runner import run_episode


@pytest.fixture(scope="module")
def easy_runs():
    cfg = load_config("easy.yaml")
    scenario = generate_scenario(cfg.run.seed, config=cfg)
    episode = build_episode(scenario)
    per_agent = {
        key: [run_episode(cfg, cfg.run.seed,
                          build_agent(key, cfg, cfg.run.seed, scenario),
                          scenario=scenario, episode=episode)]
        for key in ("sequential", "whittle")
    }
    return cfg, per_agent


def test_easy_tier_really_has_no_hard_class_emitters():
    """The premise of the regression below. If this changes, F2 changes too."""
    cfg = load_config("easy.yaml")
    scenario = generate_scenario(cfg.run.seed, config=cfg)
    present = {type(e).__name__ for e in scenario.emitters}
    assert not (present & HARD_CLASSES), (
        "EASY gained a scanning emitter; the F2 fallback path is now untested"
    )


@pytest.fixture
def capture_axes(monkeypatch):
    """Grab each figure's axes before ``_save`` closes it."""
    captured = []
    original = plots._save

    def spy(fig, path):
        captured.append(fig.axes[0])
        return original(fig, path)

    monkeypatch.setattr(plots, "_save", spy)
    return captured


def test_survival_plot_is_never_silently_empty(easy_runs, tmp_path, capture_axes):
    """On a tier with no hard-class emitters, F2 must fall back, not blank out."""
    _cfg, per_agent = easy_runs
    fig_path = plots.plot_survival(per_agent, tmp_path / "f2.png")
    assert fig_path.is_file()

    ax = capture_axes[0]
    labelled = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
    assert labelled, "F2 drew no labelled curves -- the blank-figure bug is back"
    assert ax.get_legend() is not None
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "showing all emitter classes" in texts, (
        "the fallback must be stated on the figure, not applied silently"
    )


def test_survival_plot_says_so_when_there_is_genuinely_nothing(tmp_path, capture_axes):
    """With no results at all, the figure must carry a message, not a void."""
    plots.plot_survival({}, tmp_path / "f2_empty.png")
    ax = capture_axes[0]
    assert ax.get_legend() is None
    assert any("no interceptable emitters" in t.get_text() for t in ax.texts)


def test_survival_plot_keeps_the_filter_when_the_tier_has_hard_emitters(tmp_path, capture_axes):
    """MEDIUM has scanning radars, so no fallback note should appear."""
    cfg = load_config("medium.yaml")
    scenario = generate_scenario(cfg.run.seed, config=cfg)
    episode = build_episode(scenario)
    per_agent = {
        "whittle": [run_episode(cfg, cfg.run.seed,
                                build_agent("whittle", cfg, cfg.run.seed, scenario),
                                scenario=scenario, episode=episode)]
    }
    plots.plot_survival(per_agent, tmp_path / "f2_medium.png")
    ax = capture_axes[0]
    assert [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "showing all emitter classes" not in texts
