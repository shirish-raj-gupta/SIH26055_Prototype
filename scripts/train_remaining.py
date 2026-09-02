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

#: (what, tier). Ordered so each tier's predictor precedes its hybrid.
JOBS: tuple[tuple[str, str], ...] = (
    ("predictor", "easy"),
    ("hybrid", "easy"),
    ("predictor", "hard"),
    ("hybrid", "hard"),
)

#: Episodes to aim for. Scenario diversity is what the predictor is short of.
MIN_EPISODES = 12

#: Episodes to aim for. Scenario diversity is what the predictor is short of.
TARGET_EPISODES = 40

#: Windows drawn per episode, and the floor below which an episode is barely
#: sampled. Memory is linear in this, and windows within one episode overlap
#: heavily (stride 16), so it is the right thing to trade away first.
MAX_WPE = 400
MIN_WPE = 96


def plan(tier: str) -> tuple[int, int]:
    """Choose ``(episodes, windows_per_episode)`` that fit in memory now.

    Holds the episode count and trades window density, which is the better
    exchange: 400 windows from one episode overlap at stride 16 and are highly
    correlated, whereas 40 scenarios at 200 windows cost half the memory of
    25 at 400 for a comparable window count and far more variety.

    Args:
        tier: Config tier, which fixes the channel count and window length.

    Returns:
        ``(episodes, windows_per_episode)``.
    """
    from smartscan.config import load_config

    cfg = load_config(f"{tier}.yaml")
    per_window = 4 * cfg.n_channels * cfg.predictor.window_slots * 4
    avail = _available_memory_bytes()
    if not avail:
        return TARGET_EPISODES, MIN_WPE
    # 0.8 margin over the guard's own 0.6: the count is chosen here and checked
    # moments later in the child, and free memory drifts in between.
    budget = 0.6 * avail * 0.8
    wpe = int(budget / (TARGET_EPISODES * per_window))
    if wpe >= MIN_WPE:
        return TARGET_EPISODES, min(MAX_WPE, wpe)
    # Not even the floor fits: give up episodes rather than sample them thinly.
    return max(MIN_EPISODES, int(budget / (MIN_WPE * per_window))), MIN_WPE


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


def run_once(what: str, tier: str, extra: tuple[str, ...]) -> int:
    """Invoke the training CLI once.

    Args:
        what: Model kind.
        tier: Config tier.
        extra: Additional CLI arguments.

    Returns:
        The process return code.
    """
    return subprocess.run(
        [sys.executable, "-u", "-m", "smartscan.cli", "train",
         "--what", what, "--config", f"configs/{tier}.yaml", *extra],
        cwd=REPO,
    ).returncode


def main() -> int:
    """Run the queue."""
    for what, tier in JOBS:
        if done(what, tier):
            print(f"== {what}_{tier}: already trained, skipping", flush=True)
            continue

        extra: tuple[str, ...] = ()
        if what == "predictor":
            # Wait only for enough to fit the MINIMUM corpus, then size the run
            # to whatever is actually free. Waiting for a fixed figure the box
            # never reaches is how the previous version stalled and then failed.
            from smartscan.config import load_config

            cfg = load_config(f"{tier}.yaml")
            floor_gb = MIN_EPISODES * 400 * 4 * cfg.n_channels * cfg.predictor.window_slots * 4
            floor_gb = floor_gb / 0.6 / 1e9
            if not wait_for_memory(floor_gb):
                print(f"== {what}_{tier}: never got {floor_gb:.1f} GB free; skipping", flush=True)
                continue
            n_ep, wpe = plan(tier)
            extra = ("--arch", "transformer", "--episodes", str(n_ep),
                     "--windows-per-episode", str(wpe))
            gb = n_ep * wpe * 4 * cfg.n_channels * cfg.predictor.window_slots * 4 / 1e9
            print(f"== {what}_{tier}: {n_ep} episodes x {wpe} windows "
                  f"(~{gb:.1f} GB)", flush=True)

        print(f"== {what}_{tier}: starting", flush=True)
        t0 = time.perf_counter()
        rc = run_once(what, tier, extra)

        # A predictor refused for memory is worth retrying smaller: the corpus
        # size is a preference, not a requirement, and a smaller one beats none.
        while rc != 0 and what == "predictor":
            cur_wpe = int(extra[extra.index("--windows-per-episode") + 1])
            nxt = int(cur_wpe * 0.7)
            if nxt < MIN_WPE // 2:
                break
            print(f"== {what}_{tier}: retrying at {nxt} windows/episode", flush=True)
            extra = (*extra[:-1], str(nxt))
            rc = run_once(what, tier, extra)

        mins = (time.perf_counter() - t0) / 60
        status = "done" if rc == 0 else f"FAILED rc={rc}"
        print(f"== {what}_{tier}: {status} in {mins:.1f} min", flush=True)

    print("queue complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
