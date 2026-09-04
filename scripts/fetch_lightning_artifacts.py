#!/usr/bin/env python
"""Bring a Lightning job's trained checkpoints back to this machine.

WHY THIS EXISTS.

The 100-episode job trained successfully and its checkpoint is gone. It wrote
to ``runs/checkpoints`` relative to a CWD that was destroyed with the machine,
so only the AUC numbers in its logs survived. Nothing in this repo could pull a
model back from a job, which made every cloud run write-only.

Jobs now copy their outputs to ``/teamspace/jobs/<name>/artifacts``. This reads
them back.

It stages into ``runs/incoming/<job>`` rather than over ``runs/checkpoints``.
A remote model is a CANDIDATE: the shipped 40-episode predictor sits at
AUC 0.684 with a run-to-run spread of sd 0.038, so a larger corpus has to move
AUC by roughly 0.08 before the difference carries information. Overwriting the
shipped checkpoint with an unevaluated one would discard a known quantity for
an unknown. Use --install once the comparison justifies it.

    python scripts/fetch_lightning_artifacts.py --job predictor-full-corpus-3tier
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The shipped predictor's score and the noise floor around it, both measured.
BASELINE_AUC = 0.684
NOISE_SD = 0.038


def _summarise(staged: Path) -> list[str]:
    """One line per predictor history found, with its score against baseline.

    Args:
        staged: Directory holding the downloaded artifacts.

    Returns:
        Human-readable lines, empty if no history files were downloaded.
    """
    out: list[str] = []
    for h in sorted(staged.rglob("predictor_*_history.json")):
        try:
            s = json.loads(h.read_text(encoding="utf-8"))["scores_vs_truth"]
        except (KeyError, ValueError):
            out.append(f"  {h.name}: unreadable")
            continue
        auc, ap = s["auc"], s["average_precision"]
        delta = auc - BASELINE_AUC
        # Report the comparison the way the project's own standard requires:
        # against the spread, not against the point estimate.
        verdict = ("above the noise floor" if abs(delta) > 2 * NOISE_SD
                   else "within run-to-run noise")
        out.append(f"  {h.stem}: AUC {auc:.3f} AP {ap:.3f}  "
                   f"({delta:+.3f} vs {BASELINE_AUC}, {verdict})")
    return out


def main() -> int:
    """Download, stage, and report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", default="predictor-full-corpus-3tier")
    ap.add_argument("--install", action="store_true",
                    help="Copy staged .pt/.json over runs/checkpoints. Do this "
                         "only after the AUC comparison justifies it.")
    args = ap.parse_args()

    from smartscan.credentials import load_dotenv

    load_dotenv()
    import os

    from lightning_sdk import Teamspace

    ts = Teamspace(name=os.environ.get("LIGHTNING_TEAMSPACE", "default-project"),
                   user=os.environ.get("LIGHTNING_USERNAME"))
    job = next((x for x in ts.jobs if x.name == args.job), None)
    if job is None:
        print(f"no job named {args.job!r} in teamspace {ts.name}")
        return 1
    print(f"job {job.name}: status={job.status}")

    staged = REPO_ROOT / "runs" / "incoming" / args.job
    staged.mkdir(parents=True, exist_ok=True)
    try:
        ts.download_folder(job.artifact_path, str(staged))
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        print(f"download failed: {type(exc).__name__}: {exc}")
        return 1

    files = sorted(p for p in staged.rglob("*") if p.is_file())
    print(f"staged {len(files)} files into {staged.relative_to(REPO_ROOT)}")
    for f in files[:25]:
        print(f"  {f.relative_to(staged)}  {f.stat().st_size / 1e6:.1f} MB")
    if not files:
        print("nothing downloaded -- the job has not synced artifacts yet")
        return 1

    lines = _summarise(staged)
    if lines:
        print("\nscores:")
        print("\n".join(lines))

    if args.install:
        dest = REPO_ROOT / "runs" / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            if f.suffix in (".pt", ".json", ".onnx"):
                shutil.copy2(f, dest / f.name)
                n += 1
        print(f"\ninstalled {n} files into runs/checkpoints")
    else:
        print("\nstaged only. Re-run with --install to replace the shipped "
              "checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
