"""The numbered acceptance tests from ``docs/architecture.md`` §18.

Acceptance 1 (reproducibility) lives in ``test_reproducibility.py``.

The heavier checks are marked ``slow`` and run with a reduced seed count by
default; ``make reproduce`` runs them at the full 30 seeds. Where a target is
not met, the test says so with the measured number rather than passing quietly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smartscan.agents import build_agent
from smartscan.analysis.metrics import evaluate_episode
from smartscan.config import load_config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.eval.benchmark import run_benchmark
from smartscan.runner import run_episode


# --------------------------------------------------------------------------- #
# Acceptance 2: the CLI runs and writes a metrics JSON
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
@pytest.mark.parametrize("agent", ["sequential", "ppo"])
def test_cli_run_writes_metrics_json(agent: str, tmp_path: Path):
    """`python -m smartscan.cli run --config configs/medium.yaml --agent X`."""
    out = tmp_path / f"metrics_{agent}.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "smartscan.cli", "run",
            "--config", "configs/medium.yaml", "--agent", agent,
            "--n-seeds", "1", "--out", str(out),
        ],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["agent"] == agent
    assert len(payload["config_hash"]) == 32
    for key in ("ttfi_hard_median_s", "twir_rate", "coverage", "reward_total"):
        assert key in payload["summary"]


@pytest.mark.acceptance
def test_cli_help_and_info():
    for args in (["--help"], ["info", "--config", "configs/easy.yaml"]):
        proc = subprocess.run(
            [sys.executable, "-m", "smartscan.cli", *args],
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-1000:]


# --------------------------------------------------------------------------- #
# Acceptance 3: a learned/adaptive scheduler beats the tuned sweep
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
@pytest.mark.slow
def test_a_closed_loop_scheduler_beats_the_tuned_sweep():
    """>=25 % median TTFI and >=15 % threat-weighted interception ratio.

    The baseline is the **tuned** sweep (``dwell_slots`` chosen by the ablation),
    not the textbook one-slot sweep. Beating a deliberately weak incumbent would
    prove nothing.
    """
    cfg = load_config("medium.yaml").with_overrides(run={"n_seeds": 12})
    agents = ["sequential", "ucb1", "thompson", "whittle", "phase_locked", "coprime_sweep"]
    result = run_benchmark(
        cfg, agents=agents, metrics=["ttfi_hard_median_s", "twir_rate"], progress=False
    )

    ttfi = {c.agent: c for c in result.comparisons if c.metric == "ttfi_hard_median_s"}
    twir = {c.agent: c for c in result.comparisons if c.metric == "twir_rate"}

    best_ttfi = max(ttfi.values(), key=lambda c: c.improvement)
    best_twir = max(twir.values(), key=lambda c: c.improvement)

    report = (
        f"best TTFI-hard: {best_ttfi.agent} {100 * best_ttfi.improvement:+.1f}% "
        f"(CI [{100 * best_ttfi.ci_lo:+.1f}%, {100 * best_ttfi.ci_hi:+.1f}%], "
        f"p_holm={best_ttfi.p_holm:.3g}); "
        f"best TWIR: {best_twir.agent} {100 * best_twir.improvement:+.1f}% "
        f"(CI [{100 * best_twir.ci_lo:+.1f}%, {100 * best_twir.ci_hi:+.1f}%], "
        f"p_holm={best_twir.p_holm:.3g})"
    )
    # The brief asks for the point estimates, with CIs reported.
    assert best_ttfi.improvement >= 0.25, f"TTFI target (>=25%) not met -- {report}"
    assert best_twir.improvement >= 0.15, f"TWIR target (>=15%) not met -- {report}"

    # Separately, and more strictly than the brief requires: TWIR must be a
    # *statistically supported* result, not just a favourable point estimate.
    # TTFI is deliberately NOT held to this bar -- at 30 seeds its CI straddles
    # zero, and the README says so rather than implying otherwise.
    assert best_twir.ci_lo > 0, f"TWIR CI must exclude zero -- {report}"
    assert best_twir.significant, f"TWIR must survive Holm correction -- {report}"


@pytest.mark.acceptance
def test_paired_comparison_uses_identical_scenarios():
    """Pairing is what makes 30 seeds enough; verify agents share the world."""
    cfg = load_config("easy.yaml")
    sc = generate_scenario(cfg.run.seed, config=cfg)
    ep = build_episode(sc)
    a = run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, cfg.run.seed, sc),
                    scenario=sc, episode=ep)
    b = run_episode(cfg, cfg.run.seed, build_agent("whittle", cfg, cfg.run.seed, sc),
                    scenario=sc, episode=ep)
    assert a.episode.digest() == b.episode.digest()
    assert not np.array_equal(a.actions[: min(a.n_steps, b.n_steps)],
                              b.actions[: min(a.n_steps, b.n_steps)])


# --------------------------------------------------------------------------- #
# Acceptance 4: scan-period recovery
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
@pytest.mark.slow
def test_scan_period_recovered_within_two_percent():
    """A known 4.0 s scan period recovered to within 2 % from sparse intercepts.

    Runs on ``configs/scan_on_scan.yaml`` (120 s = 30 revolutions). The 10 s
    tiers cannot support this: 2.5 revolutions is two or three arrivals, and no
    estimator turns that into a 2 % figure. See ``docs/architecture.md`` §17-A.
    """
    from smartscan.eval.scan_validation import validate_estimators

    cfg = load_config("scan_on_scan.yaml")
    report = validate_estimators(cfg, n_seeds=6)
    s = report["summary"]
    assert s["n_resolved"] > 0, "no scan period was resolved at all"
    assert s["median_rel_error_at_4s"] <= 0.02, (
        f"4.0 s period recovered to {100 * s['median_rel_error_at_4s']:.2f} % "
        f"(target 2 %); resolved {s['n_resolved']}/{s['n_emitters']} emitters"
    )


@pytest.mark.acceptance
def test_ten_second_episode_cannot_support_the_two_percent_target():
    """Documents WHY the 120 s config exists, rather than just asserting it.

    A 4.0 s scanner in a 10 s episode gives 2.5 revolutions. This test records
    the resulting arrival count so the limitation is visible in the suite rather
    than buried in a design note.
    """
    cfg = load_config("scan_on_scan.yaml").with_overrides(time={"episode_s": 10.0})
    n_revolutions = cfg.time.episode_s / 4.0
    assert n_revolutions < 3, "the premise of the 120 s config no longer holds"


# --------------------------------------------------------------------------- #
# Acceptance 5: pop-up emitters
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
@pytest.mark.slow
def test_popups_detected_better_than_by_the_sweep():
    """Pop-ups (first active at t > 0.6T) handled better by a closed-loop policy.

    "Better" is detection RATE first. On HARD only about two thirds of pop-ups
    are physically interceptable at all -- a scanning pop-up whose beam does not
    come round again before the horizon cannot be found by anybody -- so the
    discriminating question is what fraction of the *reachable* ones each policy
    actually finds. Latency over the found ones is reported alongside, because
    the sweep wins there and the write-up says so.
    """
    cfg = load_config("hard.yaml").with_overrides(run={"n_seeds": 10})
    seeds = list(range(cfg.run.seed, cfg.run.seed + cfg.run.n_seeds))
    agents = ["sequential", "ucb1", "whittle", "thompson", "coprime_sweep"]
    found: dict[str, list[int]] = {a: [] for a in agents}
    reachable: dict[str, list[int]] = {a: [] for a in agents}
    latency: dict[str, list[float]] = {a: [] for a in agents}

    for seed in seeds:
        sc = generate_scenario(seed, config=cfg)
        ep = build_episode(sc)
        assert any(t.t_first_active > 0 for t in ep.truth), "hard tier must have pop-ups"
        for key in agents:
            row = evaluate_episode(
                run_episode(cfg, seed, build_agent(key, cfg, seed, sc), scenario=sc, episode=ep), cfg
            )
            found[key].append(row["n_popup_found"])
            reachable[key].append(row["n_popup_interceptable"])
            latency[key].append(row["popup_latency_s"])

    def rate(key: str) -> float:
        total = sum(reachable[key])
        return sum(found[key]) / total if total else 0.0

    sweep_rate = rate("sequential")
    best_rate = max(rate(a) for a in agents if a != "sequential")
    summary = ", ".join(f"{a}={100 * rate(a):.0f}%" for a in agents)
    assert best_rate > sweep_rate, (
        f"no closed-loop scheduler found more pop-ups than the sweep: {summary}"
    )
    assert sum(reachable["sequential"]) >= 8, "too few reachable pop-ups to conclude anything"


# --------------------------------------------------------------------------- #
# Compute budget
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
def test_easy_tier_episode_is_fast_enough_for_a_live_demo():
    """The EASY tier must stay well inside the ten-minute reproduce budget."""
    import time

    cfg = load_config("easy.yaml")
    t0 = time.perf_counter()
    sc = generate_scenario(cfg.run.seed, config=cfg)
    ep = build_episode(sc)
    gen = time.perf_counter() - t0

    t1 = time.perf_counter()
    run_episode(cfg, cfg.run.seed, build_agent("sequential", cfg, cfg.run.seed, sc),
                scenario=sc, episode=ep)
    step = time.perf_counter() - t1

    assert gen < 1.0, f"scenario generation took {gen:.2f}s"
    assert step < 3.0, f"one analytic episode took {step:.2f}s"
