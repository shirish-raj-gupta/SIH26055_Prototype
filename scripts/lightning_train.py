#!/usr/bin/env python
"""Submit predictor training to Lightning AI.

WHY THIS EXISTS, and what it can and cannot settle.

Local training is capped at ~40 episodes because ``build_windows`` materialises
one dense float32 array at ~105 MB/episode and this workstation has ~6 GB free.
Three Kaggle attempts at the streaming alternative were cancelled without ever
producing a model: the loader decoded parquet on one thread while the T4 sat at
0 % utilisation, so ~40 GPU-hours bought nothing. Lightning removes both
constraints at once -- more RAM for the dense path, and a real GPU for the
16x-faster compute measured locally (27 ms/step against 445 ms).

It uses the SEED-REGENERATED corpus rather than the published Kaggle one. That
is deliberate: the episodes come from the same generator with the same seeds, so
they are scientifically equivalent for the "does more data help" question, and
it means no Kaggle credential has to travel to a cloud job.

Set expectations honestly before spending anything. The predictor's run-to-run
spread is **sd 0.038 AUC** over four independent draws, and the shipped
40-episode model sits at the mean of that distribution. A larger corpus has to
move AUC by roughly 0.08 before the difference carries information. This job is
worth running to CLOSE that question, not because a gain is expected.

    python scripts/lightning_train.py --episodes 200 --machine A100 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO_URL = "https://github.com/shirish-raj-gupta/SIH26055_Prototype.git"

#: Roughly what the dense window corpus costs, from the local measurement.
MB_PER_EPISODE = 105


def build_command(tier: str, episodes: int, wpe: int, arch: str, seed: int | None) -> str:
    """Compose the remote shell command.

    Args:
        tier: Config tier.
        episodes: Training episodes.
        wpe: Windows drawn per episode.
        arch: Predictor architecture.
        seed: Run seed, which selects a DISJOINT block of training episodes.
            Replication needs independent draws, not a rerun of one.

    Returns:
        A single shell command string.
    """
    seed_arg = f" --set run.seed={seed}" if seed is not None else ""
    return " && ".join([
        # onnx is not in the studio image; without it the export step fails
        # AFTER training has already succeeded and the job is marked Failed,
        # which reads as a training failure when it is not.
        f"pip install --quiet 'git+{REPO_URL}' onnx onnxruntime onnxscript",
        "python -c \"import torch; print('torch', torch.__version__, "
        "'cuda', torch.cuda.is_available())\"",
        (
            "python -m smartscan.cli train --what predictor"
            f" --config configs/{tier}.yaml --arch {arch}"
            f" --episodes {episodes} --windows-per-episode {wpe}{seed_arg}"
        ),
        f"python -m smartscan.cli export-onnx --config configs/{tier}.yaml",
        "ls -la runs/checkpoints runs/onnx",
    ])


def main() -> int:
    """Submit the job."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="medium")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--windows-per-episode", type=int, default=400)
    ap.add_argument("--arch", default="transformer")
    # A100/L4 are rejected on this account's AWS cluster ("accelerator lit-a100-1
    # not found"); T4 and the CPU_X_* tiers are what actually launch. Probed, not
    # assumed -- the SDK exposes every machine name regardless of entitlement.
    ap.add_argument("--machine", default="T4")
    ap.add_argument("--studio", default=None,
                    help="Studio supplying the environment. Job.run needs either "
                         "this or an image; with neither it tries to autodetect "
                         "and fails with 'Cannot autodetect Studio'.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="Run seed. Different seeds draw disjoint episode blocks, "
                         "which is what makes repeats independent.")
    ap.add_argument("--max-runtime", type=int, default=3600,
                    help="Seconds. A cap is not optional on metered compute.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from smartscan.credentials import credential_status

    st = credential_status()
    if not st.lightning:
        print("No Lightning credentials. Set LIGHTNING_USER_ID, LIGHTNING_API_KEY,")
        print("LIGHTNING_USERNAME and LIGHTNING_TEAMSPACE in .env — see .env.example.")
        return 1

    gb = args.episodes * args.windows_per_episode / 400 * MB_PER_EPISODE / 1024
    cmd = build_command(args.tier, args.episodes, args.windows_per_episode,
                        args.arch, args.seed)
    name = args.name or f"smartscan-predictor-{args.tier}-{args.episodes}ep"

    print(f"  teamspace   default-project (user {st.lightning_user[:8]}…, key {st.lightning_key_fingerprint})")
    print(f"  machine     {args.machine}")
    print(f"  job         {name}")
    print(f"  corpus      {args.episodes} episodes x {args.windows_per_episode} windows")
    print(f"  dense RAM   ~{gb:.1f} GB   (local ceiling was ~40 episodes)")
    print(f"  max runtime {args.max_runtime}s")
    print(f"\n  command:\n    {cmd}\n")
    if args.dry_run:
        print("--dry-run: nothing submitted.")
        return 0

    import os

    from lightning_sdk import Job, Machine, Teamspace

    # Resolve the teamspace and studio explicitly. This account has TWO
    # teamspaces both named "default-project" -- one org-owned, one user-owned --
    # and letting the SDK autodetect resolves to the org one, where the studio
    # does not exist ("Studio 'scratch-studio-devbox' does not exist").
    ts = Teamspace(
        name=os.environ.get("LIGHTNING_TEAMSPACE", "default-project"),
        user=os.environ.get("LIGHTNING_USERNAME"),
    )
    # Fall back to whatever studio the teamspace actually has. Studios get
    # renamed and recreated in the UI, and hardcoding one turns an unrelated
    # rename into a failed submission.
    available = ts.studios
    if not available:
        print("no studio in this teamspace; Job.run needs a studio or an image")
        return 1
    studio = next((s for s in available if s.name == args.studio), None) if args.studio else available[0]
    if studio is None:
        print(f"studio {args.studio!r} not found. Available: {[s.name for s in available]}")
        return 1
    print(f"  studio      {studio.name}")
    machine = getattr(Machine, args.machine)
    job = Job.run(
        name=name,
        machine=machine,
        command=cmd,
        studio=studio,
        teamspace=ts,
        interruptible=True,   # cheaper, and this job is restartable by design
    )
    print(f"submitted: {job.name}")
    print(f"status   : {job.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
