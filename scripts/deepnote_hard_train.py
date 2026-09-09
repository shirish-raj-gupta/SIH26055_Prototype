#!/usr/bin/env python
"""Train the HARD tier on the whole published corpus, and judge it honestly.

Built for Deepnote (or any GPU box). Two things make this different from the
training that produced the shipped checkpoints.

**It uses all the data.** The shipped predictors were trained on episodes
regenerated from seeds -- ``--episodes 40`` or so, a few percent of what
exists. The Kaggle corpus holds 800 HARD episodes with replayed dwell traces,
and ``cli train --dataset`` streams them, so there is no RAM ceiling and no
reason to sample.

**It is judged on the mission metric, not the loss.** This is the lesson from
the shipped runs: ``dqn_hard``'s return climbed from -570 to +312 across 3M
steps -- textbook learning -- and it still missed more emitters than a plain
sweep. Training return is not evidence. So every model trained here is scored
on emitters never intercepted, per emitter class, against the same baselines
and against the physical floor computed by ``scripts/hard_tier_ceiling.py``.

What that floor says, and why it is quoted next to every result: on HARD,
AgileBeamRadar offers a median 0.81 expected looks per episode under any
uniform-coverage policy, so ~44% of them cannot be caught by anything. Do not
read a failure to fix that class as a training failure. CircularScanRadar is
the opposite case -- a 0.5% floor against a 22% real miss rate -- and is where
a better model can actually show up.

Usage on Deepnote::

    export KAGGLE_USERNAME=... KAGGLE_KEY=...
    python scripts/deepnote_hard_train.py --stage all

    # or step at a time
    python scripts/deepnote_hard_train.py --stage data
    python scripts/deepnote_hard_train.py --stage predictor
    python scripts/deepnote_hard_train.py --stage rl --rl-steps 6000000
    python scripts/deepnote_hard_train.py --stage evaluate
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATASET_SLUG = "ew-smart-scan-rf-environment"
TIER = "hard"
CKPT = REPO_ROOT / "runs" / "checkpoints"
REPORTS = REPO_ROOT / "reports"

#: Compared against every trained model. These are the ones to beat: on HARD
#: the plain sweep is the incumbent and nothing has beaten it yet.
BASELINES = ("sequential", "coprime_sweep", "whittle", "phase_locked")


def _run(cmd: list[str], **kw) -> int:
    """Run a subprocess, streaming output, returning its exit code."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), **kw).returncode


# --------------------------------------------------------------------------- #
# Stage 1: the corpus
# --------------------------------------------------------------------------- #
def stage_data(root: str) -> Path:
    """Fetch and verify the published corpus. Returns its root."""
    from smartscan.data.kaggle_io import load_dataset, resolve_dataset_root

    found, source = resolve_dataset_root(
        root if Path(root).exists() else None, allow_download=True
    )
    if found is None:
        raise SystemExit(
            "No corpus found and none could be downloaded.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY, or pass --dataset-root to a "
            f"directory containing index.parquet. Slug: {DATASET_SLUG}"
        )
    print(f"corpus at {found}  (source: {source})")

    counts = {}
    for split in ("train", "val"):
        ds = load_dataset(split, tier=TIER, root=found, allow_download=False)
        counts[split] = len(ds)
        if ds.source == "regenerated":
            raise SystemExit(
                f"The {split!r} split fell back to regenerating from seeds. That is "
                "exactly the sampling this script exists to avoid -- the corpus is "
                "not being read. Check the root."
            )
    print(f"HARD episodes: {counts['train']} train / {counts['val']} val")
    if counts["train"] < 100:
        print(f"WARNING: only {counts['train']} train episodes; expected ~600 for HARD.")
    return Path(found)


# --------------------------------------------------------------------------- #
# Stage 2: the predictor, on everything
# --------------------------------------------------------------------------- #
def stage_predictor(root: Path, windows: int, epochs: int | None) -> None:
    """Train the occupancy predictor by streaming the whole HARD corpus."""
    # Keep the shipped checkpoint: the comparison at the end is against it, and
    # an unlabelled overwrite would destroy the only evidence of what changed.
    for name in (f"predictor_{TIER}.pt", f"predictor_{TIER}_history.json"):
        src = CKPT / name
        if src.exists():
            keep = src.with_name(src.stem + "_shipped" + src.suffix)
            if not keep.exists():
                keep.write_bytes(src.read_bytes())
                print(f"kept shipped {name} -> {keep.name}")

    cmd = [sys.executable, "-m", "smartscan.cli", "train",
           "--config", f"{TIER}.yaml", "--what", "predictor",
           "--dataset", str(root), "--windows-per-episode", str(windows),
           "--workers", "-1"]
    if epochs:
        cmd += ["--steps", str(epochs)]
    if _run(cmd):
        raise SystemExit("predictor training failed")


# --------------------------------------------------------------------------- #
# Stage 3: the RL agents
# --------------------------------------------------------------------------- #
def stage_rl(steps: int, which: tuple[str, ...]) -> None:
    """Train the RL schedulers on HARD."""
    for what in which:
        for name in (f"{what}_{TIER}.pt", f"{what}_{TIER}_trainlog.json"):
            src = CKPT / name
            if src.exists():
                keep = src.with_name(src.stem + "_shipped" + src.suffix)
                if not keep.exists():
                    keep.write_bytes(src.read_bytes())
        t0 = time.perf_counter()
        rc = _run([sys.executable, "-m", "smartscan.cli", "train",
                   "--config", f"{TIER}.yaml", "--what", what, "--steps", str(steps)])
        print(f"[{what}] {'ok' if rc == 0 else 'FAILED'} in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")
        if rc:
            raise SystemExit(f"{what} training failed")


# --------------------------------------------------------------------------- #
# Stage 4: judge it on the mission metric
# --------------------------------------------------------------------------- #
def stage_evaluate(n_seeds: int, agents: tuple[str, ...]) -> dict:
    """Score every agent on emitters never intercepted, broken down by class.

    Deliberately not the training return. A model whose return went up while
    its misses went up has not improved, and that is the exact failure the
    shipped checkpoints exhibit.
    """
    from smartscan.agents import build_agent
    from smartscan.config import load_config
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.hal.simulated import detection_probability_tensor
    from smartscan.runner import run_episode

    cfg = load_config(f"{TIER}.yaml")
    duty = cfg.receiver.ibw_channels / cfg.spectrum.n_channels
    misses: dict[str, collections.Counter] = {a: collections.Counter() for a in agents}
    totals: collections.Counter = collections.Counter()
    chances: dict[str, list[float]] = collections.defaultdict(list)
    failed: dict[str, str] = {}

    for i in range(n_seeds):
        seed = cfg.run.seed + i
        scenario = generate_scenario(seed, config=cfg)
        episode = build_episode(scenario)
        pd_tensor = detection_probability_tensor(episode, cfg)
        interceptable = pd_tensor > 0.01

        real = [t for t in episode.truth if not t.is_interferer]
        catchable: set[int] = set()
        for truth in real:
            cells = interceptable & (episode.emitter_id == truth.emitter_id)
            n = int(cells.any(axis=0).sum())
            chances[truth.emitter_class].append(n * duty)
            totals[truth.emitter_class] += 1
            if n:
                catchable.add(truth.emitter_id)

        for agent in agents:
            if agent in failed:
                continue
            try:
                result = run_episode(
                    cfg, seed, build_agent(agent, cfg, seed, scenario),
                    scenario=scenario, episode=episode,
                )
            except Exception as exc:
                failed[agent] = f"{type(exc).__name__}: {exc}"
                continue
            found = set(result.first_intercept)
            for truth in real:
                if truth.emitter_id in catchable and truth.emitter_id not in found:
                    misses[agent][truth.emitter_class] += 1

    import numpy as np
    classes = sorted(totals)
    floors = {c: float(np.exp(-np.median(chances[c]))) for c in classes}
    ok = [a for a in agents if a not in failed]

    width = max(len(a) for a in ok) + 2 if ok else 12
    print(f"\n{'=' * 78}\nHARD, {n_seeds} seeds — emitters never intercepted (lower is better)\n")
    print(f"{'class':<20}{'n':>4}{'floor':>8}" + "".join(f"{a:>{width}}" for a in ok))
    print("-" * (32 + width * len(ok)))
    for c in classes:
        print(f"{c:<20}{totals[c]:>4}{floors[c]:>7.1%}"
              + "".join(f"{misses[a][c]:>{width}}" for a in ok))
    print("-" * (32 + width * len(ok)))
    print(f"{'TOTAL':<20}{sum(totals.values()):>4}{'':>8}"
          + "".join(f"{sum(misses[a].values()):>{width}}" for a in ok))
    for a, why in failed.items():
        print(f"  {a}: could not run — {why}")

    best = min(ok, key=lambda a: sum(misses[a].values())) if ok else None
    if best:
        base = min(BASELINES, key=lambda a: sum(misses[a].values())
                   if a in misses and a not in failed else 10**9)
        nb, nl = sum(misses[best].values()), sum(misses[base].values())
        print(f"\nbest: {best} ({nb})   best baseline: {base} ({nl})")
        print("A trained model only counts if it is BELOW the baseline here."
              if nb >= nl else f"*** {best} beats {base} by {nl - nb} misses. ***")

    out = {
        "tier": TIER, "n_seeds": n_seeds,
        "poisson_floor": floors,
        "totals": {c: int(totals[c]) for c in classes},
        "misses": {a: {c: int(misses[a][c]) for c in classes} for a in ok},
        "total_misses": {a: int(sum(misses[a].values())) for a in ok},
        "failed": failed,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hard_retrain_eval.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'hard_retrain_eval.json'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "predictor", "rl", "evaluate"])
    ap.add_argument("--dataset-root", default="build/dataset")
    ap.add_argument("--windows-per-episode", type=int, default=600,
                    help="Windows sampled per episode. Higher uses more of each.")
    ap.add_argument("--predictor-epochs", type=int, default=None,
                    help="Override; hard.yaml already asks for 60.")
    ap.add_argument("--rl-steps", type=int, default=3_000_000)
    ap.add_argument("--rl", default="ppo,dqn,hybrid")
    ap.add_argument("--n-seeds", type=int, default=8)
    args = ap.parse_args()

    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
              + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except ImportError:
        print("torch missing -- pip install -e '.[ml,viz]'")
        return 1

    root = Path(args.dataset_root)
    if args.stage in ("all", "data", "predictor"):
        root = stage_data(args.dataset_root)
    if args.stage in ("all", "predictor"):
        stage_predictor(root, args.windows_per_episode, args.predictor_epochs)
    if args.stage in ("all", "rl"):
        stage_rl(args.rl_steps, tuple(w.strip() for w in args.rl.split(",") if w.strip()))
    if args.stage in ("all", "evaluate"):
        trained = ("predictor", "predictor_gc", "dqn", "ppo", "hybrid")
        stage_evaluate(args.n_seeds, BASELINES + trained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
