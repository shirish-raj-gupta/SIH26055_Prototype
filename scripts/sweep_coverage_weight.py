#!/usr/bin/env python
"""Sweep ``agents.coverage_weight`` and pick a defensible default.

The reward ablation flagged that the shipped default of 1.0 left ~29 % of
threat-weighted interception ratio on the table for the Whittle policy, on five
seeds. Five seeds is a hint, not a result, and TWIR alone is not the objective:
a weight that buys interception ratio by wrecking time-to-first-intercept or
band coverage is a worse default, not a better one.

So this sweeps a wider range across every value-based policy, on enough paired
seeds to separate signal from scenario variance, and reports **all four**
competing quantities side by side rather than the one that happens to improve.

    python scripts/sweep_coverage_weight.py --n-seeds 12

**Result, MEDIUM, 12 paired seeds.** The 29 % did not replicate -- it was a
small-sample artefact. Whittle at 2.0 is +14.7 % TWIR with a 95 % CI of
[-26 %, +58 %], and TWIR *falls* beyond that (0.023 at 1.0, 0.027 at 2.0, 0.019
at 4.0, 0.014 at 8.0). No weight beats the default on TWIR at any usable
confidence, for any agent, in either direction -- bar one regression
(phase_locked at 8.0, -48 %).

What does survive is ``staleness_max_s``, monotone in the weight and significant
from 4.0 upward (Whittle 1.82 s -> 1.01 s -> 0.61 s; -44.6 % and -66.6 %), with
``coverage`` flat at 0.80-0.87 throughout. That is close to tautological -- the
term penalises staleness -- so it is a knob, not a discovery: raise it when a
worst-case revisit guarantee matters more than interception ratio. ``ttfi_hard``
is non-monotone and noisy at this sample size; do not read it here. The default
stays at 1.0.
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
from smartscan.analysis.metrics import evaluate_episode, paired_bootstrap_delta  # noqa: E402
from smartscan.config import load_config  # noqa: E402
from smartscan.env.rf_environment import build_episode, generate_scenario  # noqa: E402
from smartscan.runner import run_episode  # noqa: E402

#: Values swept. Spans two orders of magnitude around the shipped default.
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Policies that actually read the weight. The sweep and random baselines do
#: not consult the belief at all, so including them would only add noise.
AGENTS = ("ucb1", "thompson", "whittle", "phase_locked")

#: Reported together, because they trade against each other.
METRICS = ("twir_rate", "ttfi_hard_median_s", "coverage", "staleness_max_s")

#: Direction of improvement per metric.
LOWER_BETTER = {"ttfi_hard_median_s", "staleness_max_s"}


def run(tier: str, n_seeds: int, out: Path) -> dict:
    """Run the sweep and write the report.

    Args:
        tier: Config tier to sweep on.
        n_seeds: Paired seeds per cell.
        out: Destination JSON path.

    Returns:
        The report dict.
    """
    base_cfg = load_config(f"{tier}.yaml")
    seeds = [base_cfg.run.seed + i for i in range(n_seeds)]

    # Build each scenario once and share it across every cell: the comparison is
    # paired on scenario AND on detection luck, so differences are the weight.
    worlds = {}
    for seed in seeds:
        scenario = generate_scenario(seed, config=base_cfg)
        worlds[seed] = (scenario, build_episode(scenario))

    t0 = time.perf_counter()
    raw: dict[str, dict[str, dict[str, list[float]]]] = {}
    for weight in WEIGHTS:
        cfg = base_cfg.with_overrides(agents={"coverage_weight": weight})
        raw[str(weight)] = {}
        for agent in AGENTS:
            rows = []
            for seed in seeds:
                scenario, episode = worlds[seed]
                result = run_episode(
                    cfg, seed, build_agent(agent, cfg, seed, scenario),
                    scenario=scenario, episode=episode,
                )
                rows.append(evaluate_episode(result, cfg))
            raw[str(weight)][agent] = {m: [r[m] for r in rows] for m in METRICS}
        print(f"  weight {weight:<5} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # -- summarise -------------------------------------------------------- #
    def med(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else float("nan")

    summary: dict[str, dict[str, dict[str, float]]] = {
        w: {a: {m: med(v) for m, v in per_metric.items()} for a, per_metric in per_agent.items()}
        for w, per_agent in raw.items()
    }

    # Paired improvement of each weight over the shipped default, per agent.
    deltas: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for weight in WEIGHTS:
        if weight == 1.0:
            continue
        deltas[str(weight)] = {}
        for agent in AGENTS:
            deltas[str(weight)][agent] = {}
            for metric in METRICS:
                cand = np.asarray(raw[str(weight)][agent][metric], dtype=np.float64)
                base = np.asarray(raw["1.0"][agent][metric], dtype=np.float64)
                ok = np.isfinite(cand) & np.isfinite(base)
                if ok.sum() < 3:
                    continue
                d = paired_bootstrap_delta(
                    cand[ok], base[ok], relative=True, n_boot=4000, seed=base_cfg.run.seed
                )
                sign = 1.0 if metric in LOWER_BETTER else -1.0
                lo, hi = sorted((sign * d.lo, sign * d.hi))
                deltas[str(weight)][agent][metric] = {
                    "improvement": sign * d.point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "significant": bool(lo > 0 or hi < 0),
                }

    report = {
        "tier": tier,
        "n_seeds": n_seeds,
        "weights": list(WEIGHTS),
        "agents": list(AGENTS),
        "metrics": list(METRICS),
        "config_hash": base_cfg.hash(),
        "wall_time_s": time.perf_counter() - t0,
        "summary": summary,
        "delta_vs_default": deltas,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def render(report: dict) -> None:
    """Print the sweep as a table a reviewer can read."""
    print()
    print(f"coverage_weight sweep — {report['tier']}, {report['n_seeds']} paired seeds")
    for metric in report["metrics"]:
        arrow = "lower better" if metric in LOWER_BETTER else "higher better"
        print()
        print(f"  {metric}  ({arrow})")
        header = "    weight " + "".join(f"{a:>16}" for a in report["agents"])
        print(header)
        for weight in report["weights"]:
            cells = ""
            for agent in report["agents"]:
                value = report["summary"][str(weight)][agent][metric]
                cells += f"{value:>16.4g}"
            mark = "  <- default" if weight == 1.0 else ""
            print(f"    {weight:<7}{cells}{mark}")

    print()
    print("  Significant improvements over the default (CI excludes zero):")
    found = False
    for weight, per_agent in report["delta_vs_default"].items():
        for agent, per_metric in per_agent.items():
            for metric, d in per_metric.items():
                if d["significant"] and d["improvement"] > 0:
                    found = True
                    print(
                        f"    weight {weight:<5} {agent:<14} {metric:<22} "
                        f"{100 * d['improvement']:+7.1f}%  "
                        f"[{100 * d['ci_lo']:+.1f}, {100 * d['ci_hi']:+.1f}]"
                    )
    if not found:
        print("    none — the default is not measurably beaten on any metric.")


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--out", default="reports/coverage_weight_sweep.json")
    args = parser.parse_args()

    print(f"sweeping coverage_weight over {WEIGHTS} on {args.tier} ...")
    render(run(args.tier, args.n_seeds, Path(args.out)))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
