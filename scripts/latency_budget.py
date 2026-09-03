#!/usr/bin/env python
"""Per-decision latency against the receiver's real-time budget.

The risk table in the submission promises a "latency budget" for embedded
deployment, and until now that was an intention rather than a number. A
scheduler is only deployable if it can decide, tune and dwell inside one dwell
period; a policy that needs longer than its own dwell cannot run in the loop at
all, however good its interception ratio looks offline.

Budget = (t_settle_slots + 1) * dt. On MEDIUM that is 3.00 ms.

Measured on the host CPU, single-observation inference -- NOT batched, because
a live receiver decides one dwell at a time and batch throughput is the wrong
number entirely. Embedded silicon will differ; what transfers is the ratio
between policies and the size of the margin.

    python scripts/latency_budget.py --tier medium
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
from smartscan.agents.belief import BeliefState  # noqa: E402
from smartscan.config import load_config  # noqa: E402
from smartscan.env.receiver import Receiver  # noqa: E402
from smartscan.env.rf_environment import build_episode, generate_scenario  # noqa: E402

AGENTS = ("sequential", "ucb1", "whittle", "phase_locked",
          "predictor", "dqn", "ppo", "hybrid")


def measure(tier: str, n: int = 300, warmup: int = 30) -> dict:
    """Time ``act()`` per decision for every scheduler.

    Args:
        tier: Config tier.
        n: Timed decisions per scheduler.
        warmup: Untimed decisions first, so lazy imports and the first CUDA
            context do not land in the measurement.

    Returns:
        Report dict with the budget and per-scheduler percentiles in ms.
    """
    cfg = load_config(f"{tier}.yaml")
    sc = generate_scenario(cfg.run.seed, config=cfg)
    ep = build_episode(sc)
    dt_ms = ep.dt_s * 1000.0
    budget_ms = (cfg.receiver.t_settle_slots + 1) * dt_ms

    rows: dict[str, dict[str, float]] = {}
    for key in AGENTS:
        agent = build_agent(key, cfg, cfg.run.seed, sc)
        rx = Receiver(ep, cfg, seed=cfg.run.seed)
        belief = BeliefState(cfg, ep.n_slots)
        for _ in range(warmup):
            obs = rx.step(agent.act(belief, rx.t))
            belief.update(obs)
            agent.observe(obs)

        lat: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            action = agent.act(belief, rx.t)
            lat.append((time.perf_counter() - t0) * 1000.0)
            obs = rx.step(action)
            belief.update(obs)
            agent.observe(obs)

        arr = np.asarray(lat)
        p50, p95, p99 = (float(x) for x in np.percentile(arr, [50, 95, 99]))
        rows[key] = {
            "median_ms": p50, "p95_ms": p95, "p99_ms": p99,
            # p99 is the number that matters: a scheduler that misses its dwell
            # one time in a hundred drops an intercept, and the emitter it
            # misses is the one that was only briefly illuminating us.
            "fits_budget": bool(p99 < budget_ms),
            "headroom_x": budget_ms / p99 if p99 > 0 else float("inf"),
        }
    return {"tier": tier, "slot_ms": dt_ms, "budget_ms": budget_ms,
            "n_decisions": n, "schedulers": rows}


def render(rep: dict) -> None:
    """Print the table."""
    print(f"\nper-decision latency — {rep['tier']}, {rep['n_decisions']} decisions")
    print(f"budget = (t_settle + 1) x dt = {rep['budget_ms']:.2f} ms\n")
    print(f"{'scheduler':<16}{'median':>10}{'p95':>10}{'p99':>10}{'headroom':>11}   verdict")
    print("-" * 72)
    for k, v in rep["schedulers"].items():
        verdict = "OK" if v["fits_budget"] else "MISSES DWELL"
        head = f"{v['headroom_x']:.0f}x" if v["fits_budget"] else "-"
        print(f"{k:<16}{v['median_ms']:>8.3f}ms{v['p95_ms']:>8.3f}ms"
              f"{v['p99_ms']:>8.3f}ms{head:>11}   {verdict}")


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="medium")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="reports/latency_budget.json")
    args = ap.parse_args()
    rep = measure(args.tier, args.n)
    render(rep)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
