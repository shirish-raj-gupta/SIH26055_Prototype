#!/usr/bin/env python
"""Train every remaining checkpoint locally, in dependency order.

Everything runs on this machine. There is no cloud step: the predictors are the
only GPU-friendly work here, and the RL agents are bound by Python simulator
rollouts across parallel envs rather than by matmuls, so 24 local cores beat a
2-vCPU hosted runtime.

Two constraints drive the ordering:

* **A hybrid needs its tier's predictor.** ``train_ppo(hybrid=True)`` refuses to
  start otherwise rather than augmenting observations with an untrained model
  and teaching the policy to read noise.
* **Predictors need RAM.** ``build_windows`` materialises one dense float32
  array at ~105 MB/episode with no streaming path, so a predictor cannot run
  beside anything large. This waits for headroom instead of racing for it --
  an earlier 128-episode attempt climbed to 5.9 GB with 4.1 GB free and had to
  be killed before it took a 4.5-hour DQN run down with it.

    python scripts/train_remaining.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from smartscan.agents.predictors import _available_memory_bytes  # noqa: E402

#: (what, tier, extra args). Ordered so each tier's predictor precedes its hybrid.
JOBS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("predictor", "easy", ("--arch", "transformer", "--episodes", "40")),
    ("hybrid", "easy", ()),
    ("predictor", "hard", ("--arch", "transformer", "--episodes", "40")),
    ("hybrid", "hard", ()),
)

#: A predictor corpus needs ~4.2 GB at 40 episodes; leave room to breathe.
NEED_GB = 5.5


def done(what: str, tier: str) -> bool:
    """True if a finished (not partial) checkpoint already exists."""
    ck = REPO / "runs" / "checkpoints"
    return (ck / f"{what}_{tier}.pt").is_file() and not (
        ck / f"{what}_{tier}_progress.json"
    ).is_file()


def wait_for_memory(need_gb: float, timeout_s: float = 7200) -> bool:
    """Block until free RAM reaches ``need_gb``, or give up.

    Args:
        need_gb: Required free memory in GB.
        timeout_s: Seconds to wait before proceeding anyway.

    Returns:
        True if the requirement was met, False if it timed out.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free = _available_memory_bytes() / 1e9
        if free >= need_gb:
            return True
        print(f"  waiting for RAM: {free:.1f} GB free, need {need_gb:.1f}", flush=True)
        time.sleep(120)
    return False


def main() -> int:
    """Run the queue."""
    for what, tier, extra in JOBS:
        if done(what, tier):
            print(f"== {what}_{tier}: already trained, skipping", flush=True)
            continue
        if what == "predictor" and not wait_for_memory(NEED_GB):
            print(f"== {what}_{tier}: never got {NEED_GB} GB free; skipping", flush=True)
            continue

        print(f"== {what}_{tier}: starting", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, "-u", "-m", "smartscan.cli", "train",
             "--what", what, "--config", f"configs/{tier}.yaml", *extra],
            cwd=REPO,
        )
        mins = (time.perf_counter() - t0) / 60
        status = "done" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"== {what}_{tier}: {status} in {mins:.1f} min", flush=True)

    print("queue complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
