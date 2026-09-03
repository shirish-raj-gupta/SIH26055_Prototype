#!/usr/bin/env python
"""Tune ``agents.coverage_weight`` against hard-target hazard, not TWIR.

The earlier sweep (``sweep_coverage_weight.py``) optimised threat-weighted
interception ratio. The log-rank analysis has since shown TWIR to be the
misleading objective: the policies that maximise it are significantly WORSE at
intercepting the scanning and agile emitters the brief is about, because TWIR
counts interceptions without asking which emitter was intercepted.

So this sweeps the same knob against the quantity that survived scrutiny --
the log-rank hazard ratio for hard-class time-to-first-intercept, which keeps
never-intercepted emitters as right-censored observations instead of dropping
them. TWIR is still reported alongside, because a weight that wins on hazard by
destroying interception ratio is not an improvement either.

    python scripts/sweep_coverage_weight_hazard.py --n-seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smartscan.agents import build_agent  # noqa: E402
from smartscan.analysis.metrics import evaluate_episode, logrank_test  # noqa: E402
from smartscan.config import load_config  # noqa: E402
from smartscan.env.rf_environment import build_episode, generate_scenario  # noqa: E402
from smartscan.runner import run_episode  # noqa: E402

WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
AGENTS = ("whittle", "phase_locked", "ucb1")


def run(tier: str, n_seeds: int, out: Path) -> dict:
    """Sweep the weight and score each setting by hazard ratio.

    Args:
        tier: Config tier.
        n_seeds: Paired seeds per cell.
        out: Destination JSON.

    Returns:
        The report dict.
    """
    base = load_config(f"{tier}.yaml")
    seeds = [base.run.seed + i for i in range(n_seeds)]
    worlds = {s: (lambda sc: (sc, build_episode(sc)))(generate_scenario(s, config=base))
              for s in seeds}

    def collect(cfg, agent: str) -> tuple[list[float], list[bool], list[float]]:
        dur: list[float] = []
        obs: list[bool] = []
        twir: list[float] = []
        for s in seeds:
            sc, ep = worlds[s]
            res = run_episode(cfg, s, build_agent(agent, cfg, s, sc), scenario=sc, episode=ep)
            row = evaluate_episode(res, cfg)
            hard = (row.pop("_detail", None) or {}).get("ttfi_hard") or {}
            dur.extend(np.asarray(hard.get("durations_s", []), dtype=float).tolist())
            obs.extend(np.asarray(hard.get("observed", []), dtype=bool).tolist())
            twir.append(row["twir_rate"])
        return dur, obs, twir

    # Baseline: the tuned sweep, which is what every hazard ratio is measured against.
    bd, bo, _ = collect(base, base.eval.baseline_agent)

    t0 = time.perf_counter()
    report: dict = {"tier": tier, "n_seeds": n_seeds, "weights": list(WEIGHTS),
                    "baseline": base.eval.baseline_agent, "cells": {}}
    for w in WEIGHTS:
        cfg = base.with_overrides(agents={"coverage_weight": w})
        for agent in AGENTS:
            d, o, twir = collect(cfg, agent)
            lr = logrank_test(np.array(d), np.array(o), np.array(bd), np.array(bo))
            hr = lr.observed_a / lr.expected_a if lr.expected_a > 0 else float("nan")
            report["cells"][f"{w}|{agent}"] = {
                "weight": w, "agent": agent,
                "hazard_ratio": hr, "p_value": lr.p_value,
                "never_intercepted": lr.censored_a, "n_emitters": lr.n_a,
                "twir_median": float(np.median(twir)),
            }
        print(f"  weight {w:<5} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def render(rep: dict) -> None:
    """Print the sweep as a table."""
    print()
    print(f"coverage_weight vs hard-target hazard — {rep['tier']}, {rep['n_seeds']} seeds")
    print(f"baseline: {rep['baseline']}   (hazard > 1 = intercepts faster)")
    for agent in AGENTS:
        print(f"\n  {agent}")
        print(f"    {'weight':<8}{'hazard':>9}{'p':>11}{'never int.':>13}{'TWIR':>10}")
        for w in rep["weights"]:
            c = rep["cells"].get(f"{w}|{agent}")
            if not c:
                continue
            mark = "  <- default" if w == 1.0 else ""
            print(f"    {w:<8}{c['hazard_ratio']:>9.3f}{c['p_value']:>11.2e}"
                  f"{c['never_intercepted']:>8}/{c['n_emitters']:<4}{c['twir_median']:>10.4f}{mark}")


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="medium")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out", default="reports/coverage_weight_hazard.json")
    args = ap.parse_args()
    render(run(args.tier, args.n_seeds, Path(args.out)))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
