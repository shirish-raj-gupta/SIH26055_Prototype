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

#: Train-split size of the published corpus, per tier, from
#: ``build/dataset/build_report.json`` (2105 train episodes of 3000 total).
#: These are the counts that make "the full corpus" a definite quantity
#: rather than a round number someone picked.
FULL_CORPUS = {"easy": 694, "medium": 854, "hard": 557}


def build_all_tiers_command(wpe: int, arch: str, seed: int | None, job_name: str,
                           tiers: tuple[str, ...], dataset: str | None = None,
                           workers: int = -1) -> str:
    """Compose a command that trains all three tiers CONCURRENTLY, one per GPU.

    Submitting three separate jobs would rent three machines. The dense corpus
    for all three tiers together is ~216 GB, which fits one 8x L4 box (~499 GB
    available) with room to spare, so one machine does the whole job at a third
    of the price and a third of the wall-clock.

    Each tier is pinned to its own GPU with ``CUDA_VISIBLE_DEVICES``; without
    the pin all three land on cuda:0 and contend for 24 GB of VRAM.

    Args:
        wpe: Windows drawn per episode.
        arch: Predictor architecture.
        seed: Run seed, or None to leave the config default.
        job_name: Job name, which fixes the artifact directory. Anything not
            copied there dies with the machine.
        tiers: Which tiers to train, in order. A subset exists so that a tier
            cut off by the runtime cap can be restaged on THIS path -- the
            single-tier path has no artifact persistence, and restaging there
            would silently reproduce the write-only run that lost the
            100-episode checkpoint.
        dataset: Stream the published corpus from this path instead of
            regenerating episodes densely in RAM. The dense path put 854
            episodes of MEDIUM beyond 4.8 hours on 32 cores; streaming has no
            RAM ceiling and reads the whole split.
        workers: Dataloader workers. The streaming path is data-bound -- the
            CLI's own note measures ~9.5 s/batch single-threaded against 27 ms
            in memory -- so this matters more than the accelerator.

    Returns:
        A single shell command string.
    """
    seed_arg = f" --set run.seed={seed}" if seed is not None else ""
    art = f"/teamspace/jobs/{job_name}/artifacts"
    lines = [
        # onnx is not in the studio image; without it the export step fails
        # AFTER training has already succeeded and the job is marked Failed,
        # which reads as a training failure when it is not.
        f"pip install --quiet 'git+{REPO_URL}' onnx onnxruntime onnxscript || exit 1",
        # The studio image ships matplotlib/pandas/scipy compiled against
        # NumPy 1.x. smartscan requires numpy>=2, so installing it breaks their
        # ABI, and something in site startup imports matplotlib -- so EVERY
        # python process dies before running a line of our code. Pinning numpy
        # down is not open to us; rebuild the dependents against numpy 2.
        "pip install --quiet --upgrade matplotlib pandas scipy pyarrow || true",
        "python -c 'import matplotlib, numpy; print(\"matplotlib\", "
        "matplotlib.__version__, \"numpy\", numpy.__version__)' || exit 1",
        # Record the two numbers that decide whether this job can work at all.
        # The RAM figure is an ESTIMATE until a job prints it; capture it.
        "echo '=== host resources ==='",
        "free -g || true",
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv || true",
        "python -c \"import torch; print('torch', torch.__version__, "
        "'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())\"",
    ]
    if dataset and dataset.endswith(".tar"):
        # Uploading the corpus as 9004 separate files ran at ~15 files/min --
        # a 10-hour upload. One archive took 271 s. Unpack it onto local disk;
        # reading 9004 small files over the teamspace mount during training
        # would reintroduce the same per-file latency the archive avoided.
        lines += [
            "mkdir -p /tmp/ds",
            f"tar -xf {dataset} -C /tmp/ds || exit 1",
            "ls /tmp/ds/dataset | head",
        ]
        dataset = "/tmp/ds/dataset"
    lines += [
        f"mkdir -p {art} || true",
        "echo \"cores=$(nproc)\"",
        # Per-tier copying protects a finished tier from a later failure. It
        # does nothing DURING a tier, and medium ran 4.8 h inside one before
        # being stopped with nothing saved. Sync whatever exists every 5 min.
        f"( while true; do sleep 300;"
        f" cp -r runs/checkpoints runs/onnx {art}/ 2>/dev/null;"
        f" done ) & SYNC=$!",
    ]
    # SEQUENTIAL, not concurrent. `free -g` on DATA_PREP reports 247 GB total
    # and 241 available -- not the 768 the machine picker advertises. All three
    # tiers at once need ~360 GB and would have been OOM-killed. Run one at a
    # time: each fits (medium, the largest, needs ~146 GB), each gets all the
    # cores, and a tier that dies cannot take the others with it.
    # Medium first: it is the tier the dashboard and the docs actually quote.
    for i, tier in enumerate(tiers):
        if dataset:
            # --dataset reads the whole published split, so --episodes does not
            # apply; the CLI errors if the corpus is missing rather than quietly
            # regenerating from seeds.
            source = f" --dataset {dataset} --workers {workers}"
        else:
            source = f" --episodes {FULL_CORPUS[tier]}"
        lines.append(
            f"OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc)"
            f" python -m smartscan.cli train"
            f" --what predictor --config configs/{tier}.yaml --arch {arch}"
            f"{source} --windows-per-episode {wpe}{seed_arg}"
            f" > {tier}.log 2>&1; R{i}=$?"
        )
        # Persist after each tier, so a later failure or a runtime cap cannot
        # cost an earlier tier's model.
        lines.append(f"cp -r runs/checkpoints runs/onnx {art}/ 2>/dev/null || true")
        lines.append(f"echo \"--- {tier} finished, exit $R{i} ---\"")
    for i, tier in enumerate(tiers):
        lines.append(f"echo '=== {tier} (exit '$R{i}') ==='; tail -n 40 {tier}.log")
    # Export is best-effort: a failed export must not mask a trained model.
    for tier in tiers:
        lines.append(f"python -m smartscan.cli export-onnx --config configs/{tier}.yaml || true")
    # Copy BEFORE the exit-code test, so a tier that failed does not discard the
    # two that succeeded.
    lines += [
        "kill $SYNC 2>/dev/null || true",
        f"cp -r runs/checkpoints runs/onnx {art}/ 2>/dev/null || true",
        f"cp -f {' '.join(t + '.log' for t in tiers)} {art}/ 2>/dev/null || true",
        f"echo '=== artifacts persisted ==='; ls -laR {art} || true",
    ]
    # Make the JOB status mean "all three tiers trained".
    lines.append("test " + " -a ".join(f"$R{i} -eq 0" for i in range(len(tiers))))
    return " && ".join(lines[:1]) + " ; " + " ; ".join(lines[1:])


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
    ap.add_argument("--dataset", default=None,
                    help="Stream the corpus from this path in the job (e.g. "
                         "/teamspace/uploads/smartscan-dataset) instead of "
                         "regenerating episodes in RAM.")
    ap.add_argument("--workers", type=int, default=-1,
                    help="Dataloader workers for --dataset. -1 auto-sizes, but "
                         "the CLI caps auto at 8; set it explicitly on a big box.")
    ap.add_argument("--tiers", default="medium,easy,hard",
                    help="Comma-separated tiers for --all-tiers, in order. Use a "
                         "subset to restage a tier the runtime cap cut off.")
    ap.add_argument("--all-tiers", action="store_true",
                    help="Train easy+medium+hard on the FULL published train "
                         "split (694/854/557 episodes), concurrently on one "
                         "machine, one tier per GPU. Overrides --tier/--episodes.")
    ap.add_argument("--interruptible", action="store_true",
                    help="Cheaper, but preemptible. Off by default for "
                         "full-corpus runs: losing hours of a metered job to a "
                         "preemption costs more than the discount saves.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--windows-per-episode", type=int, default=400)
    ap.add_argument("--arch", default="transformer")
    # A100 is rejected on this account's AWS cluster ("accelerator lit-a100-1
    # not found"); T4 launches. The SDK exposes every machine name regardless of
    # entitlement, so availability is only ever known at submit time.
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

    scale = args.windows_per_episode / 400 * MB_PER_EPISODE / 1024
    if args.all_tiers:
        tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
        bad = [t for t in tiers if t not in FULL_CORPUS]
        if bad:
            print(f"unknown tier(s) {bad}; choose from {list(FULL_CORPUS)}")
            return 1
        gb = sum(FULL_CORPUS[t] for t in tiers) * scale
        name = args.name or f"predictor-full-corpus-{'-'.join(tiers)}"
        cmd = build_all_tiers_command(args.windows_per_episode, args.arch,
                                      args.seed, name, tiers, args.dataset,
                                      args.workers)
        corpus = " + ".join(f"{t} {FULL_CORPUS[t]}" for t in tiers)
        corpus += f" = {sum(FULL_CORPUS[t] for t in tiers)} episodes"
        if args.dataset:
            corpus += f"  [STREAMED from {args.dataset}]"
    else:
        gb = args.episodes * scale
        cmd = build_command(args.tier, args.episodes, args.windows_per_episode,
                            args.arch, args.seed)
        name = args.name or f"smartscan-predictor-{args.tier}-{args.episodes}ep"
        corpus = f"{args.episodes} episodes x {args.windows_per_episode} windows"

    import os as _os

    ts_name = _os.environ.get("LIGHTNING_TEAMSPACE", "default-project")
    print(f"  teamspace   {ts_name} (user {st.lightning_user[:8]}…, "
          f"key {st.lightning_key_fingerprint})")
    print(f"  machine     {args.machine}")
    print(f"  job         {name}")
    print(f"  corpus      {corpus}")
    if args.all_tiers and args.dataset:
        # Streaming allocates no dense corpus at all. Printing the dense figure
        # here would suggest a RAM requirement the run does not have, which is
        # the whole reason for using this path.
        print("  dense RAM   n/a - streamed, no dense corpus is materialised")
    else:
        print(f"  dense RAM   ~{gb:.1f} GB peak   (local ceiling was ~40 episodes)")
    # Quoting one machine's price regardless of the machine chosen is how a
    # $55 job gets announced as a $95 one, or worse, the reverse.
    rate = {"DATA_PREP": 9.25, "L4_X_8": 15.90, "T4_X_4": 4.69, "T4": 1.10,
            "CPU": 1.25, "L40S_X_8": 33.68, "H100_X_8": 33.12}.get(args.machine)
    if rate:
        print(f"  max runtime {args.max_runtime}s  -> up to "
              f"${args.max_runtime / 3600 * rate:.2f} at {args.machine} "
              f"(${rate}/hr on-demand)")
    else:
        print(f"  max runtime {args.max_runtime}s  (rate for {args.machine} unknown)")
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
        interruptible=args.interruptible,
    )
    print(f"submitted: {job.name}")
    print(f"status   : {job.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
