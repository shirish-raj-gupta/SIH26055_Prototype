#!/usr/bin/env python
"""Turn a better HARD predictor into a better HARD scheduler, or show it cannot.

A sharper predictor does not give a better scheduler for free, and on MEDIUM it
gave a worse one: retraining lifted AUC 0.684 -> 0.763 while never-intercepted
went 112 -> 126. The cause is one line in ``SequencePredictorScheduler.act``::

    value = p * (1 - 0.9 * interferer) + coverage_weight * staleness
    action = argmax(window_value(value))

``argmax(P + w*s)`` is not invariant under rescaling of ``P``. Sharpen the
predictor and its dynamic range grows, so the same ``coverage_weight`` becomes
effectively smaller, the schedule stops covering, and emitters are never seen.
The better model is spent making the schedule worse.

There are two ways out and this sweeps both:

``coverage_weight``
    Keep the additive rule and re-tune ``w`` for the new predictor's scale.
    Cheap, but the right value is a property of the checkpoint, so it must be
    re-tuned every time the model changes -- which is the underlying fragility,
    not a fix for it.

``coverage_fraction``
    ``predictor_gc`` splits the SLOT BUDGET instead of the score: a fraction of
    dwells are coverage dwells chosen by staleness alone, the rest are exploit
    dwells chosen by the predictor. No addition, so no scale to get wrong. This
    is the design that should survive a model swap.

Also runs ``predictor_de`` (min-max normalised scores, invariant by
construction) and ``whittle_predictor``.

Everything is scored on emitters never intercepted, against ``sequential`` and
``coprime_sweep``, which are what actually lead on HARD.

Usage::

    python scripts/sweep_hard_scheduler.py --checkpoint runs/checkpoints/predictor_hard_fullcorpus.pt
    python scripts/sweep_hard_scheduler.py --n-seeds 8 --out reports/hard_scheduler_sweep.json
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CKPT = REPO_ROOT / "runs" / "checkpoints"
LIVE = CKPT / "predictor_hard.pt"
BASELINES = ("sequential", "coprime_sweep", "whittle")

#: The additive rule's knob. Sweeping it upward tests whether the new model's
#: sharper scores simply need a larger staleness term to stay covered.
COVERAGE_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0)

#: The budget-split knob: fraction of dwells reserved for coverage.
COVERAGE_FRACTIONS = (0.5, 0.7, 0.85, 0.95)


def _prepare(tier: str, n_seeds: int):
    """Build the episodes once so every arm sees identical worlds and luck."""
    from smartscan.config import load_config
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.hal.simulated import detection_probability_tensor

    cfg = load_config(f"{tier}.yaml")
    out = []
    for i in range(n_seeds):
        seed = cfg.run.seed + i
        scenario = generate_scenario(seed, config=cfg)
        episode = build_episode(scenario)
        pd_tensor = detection_probability_tensor(episode, cfg)
        interceptable = pd_tensor > 0.01
        real = [t for t in episode.truth if not t.is_interferer]
        catchable = {
            t.emitter_id for t in real
            if (interceptable & (episode.emitter_id == t.emitter_id)).any()
        }
        out.append((seed, scenario, episode, real, catchable))
    return cfg, out


def _score(agent: str, cfg, episodes) -> tuple[int, collections.Counter]:
    """Emitters never intercepted, and the per-class breakdown."""
    from smartscan.agents import build_agent
    from smartscan.runner import run_episode

    miss, per = 0, collections.Counter()
    for seed, scenario, episode, real, catchable in episodes:
        result = run_episode(
            cfg, seed, build_agent(agent, cfg, seed, scenario),
            scenario=scenario, episode=episode,
        )
        found = set(result.first_intercept)
        for truth in real:
            if truth.emitter_id in catchable and truth.emitter_id not in found:
                miss += 1
                per[truth.emitter_class] += 1
    return miss, per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(CKPT / "predictor_hard_fullcorpus.pt"),
                    help="Predictor checkpoint to sweep with.")
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "hard_scheduler_sweep.json"))
    args = ap.parse_args()

    from smartscan.config import load_config

    new_ckpt = Path(args.checkpoint)
    if not new_ckpt.is_file():
        raise SystemExit(f"no checkpoint at {new_ckpt}")

    # Swap the candidate in, and put the original back whatever happens: the
    # repo's shipped checkpoint is the comparison baseline and losing it would
    # cost another training run.
    backup = None
    if LIVE.is_file():
        backup = LIVE.with_suffix(".pt.sweepbak")
        shutil.copy2(LIVE, backup)
        print(f"backed up live checkpoint -> {backup.name}", flush=True)

    t0 = time.time()
    results: dict[str, dict] = {}
    try:
        shutil.copy2(new_ckpt, LIVE)
        print(f"sweeping with {new_ckpt.name}\n", flush=True)

        cfg, episodes = _prepare(args.tier, args.n_seeds)
        n_catchable = sum(len(c) for *_, c in episodes)
        print(f"{args.tier}: {args.n_seeds} seeds, {n_catchable} catchable emitters\n", flush=True)

        print(f"{'arm':<38}{'missed':>8}   worst classes", flush=True)
        print("-" * 84, flush=True)

        def run(label: str, agent: str, overrides: dict | None = None) -> None:
            c = load_config(f"{args.tier}.yaml", overrides) if overrides else cfg
            miss, per = _score(agent, c, episodes)
            results[label] = {"agent": agent, "overrides": overrides or {},
                              "missed": miss, "per_class": dict(per)}
            top = ", ".join(f"{k}:{v}" for k, v in per.most_common(2))
            print(f"{label:<38}{miss:>8}   {top}", flush=True)

        for a in BASELINES:
            run(a, a)
        print("-" * 84, flush=True)

        # Additive rule, re-tuned for the new scale.
        for w in COVERAGE_WEIGHTS:
            run(f"predictor (coverage_weight={w})", "predictor",
                {"agents.coverage_weight": w})
        print("-" * 84, flush=True)

        # Budget split: no scale to get wrong.
        for rho in COVERAGE_FRACTIONS:
            run(f"predictor_gc (coverage_fraction={rho})", "predictor_gc",
                {"agents.coverage_fraction": rho})
        print("-" * 84, flush=True)

        for a in ("predictor_de", "whittle_predictor"):
            run(a, a)
    finally:
        if backup is not None:
            shutil.copy2(backup, LIVE)
            backup.unlink(missing_ok=True)
            print("\nrestored the original live checkpoint", flush=True)

    best_base = min((results[a]["missed"] for a in BASELINES if a in results), default=10**9)
    learned = {k: v["missed"] for k, v in results.items() if k not in BASELINES}
    best_arm = min(learned, key=learned.get) if learned else None

    print("\n" + "=" * 84)
    print(f"best baseline      : {best_base}")
    if best_arm:
        print(f"best learned arm   : {learned[best_arm]}  ({best_arm})")
        print("VERDICT            : " + (
            f"*** beats the baseline by {best_base - learned[best_arm]} ***"
            if learned[best_arm] < best_base else
            "does NOT beat the baseline"))
    print(f"elapsed            : {(time.time() - t0) / 60:.1f} min")

    payload = {"tier": args.tier, "n_seeds": args.n_seeds,
               "checkpoint": str(new_ckpt), "results": results,
               "best_baseline": best_base}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
