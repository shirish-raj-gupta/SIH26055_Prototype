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
A remote model is a CANDIDATE: a run-to-run spread of sd 0.038 means a
candidate has to move AUC by roughly 0.08 against ITS OWN TIER before the
difference carries information. Overwriting the shipped checkpoint with an
unevaluated one would discard a known quantity for an unknown. Use --install
once the comparison justifies it.

The shipped predictors have since been retrained at a larger episode count
(easy 0.957, medium 0.763, hard 0.703), so the bar a new candidate has to clear
is the one in BASELINE_AUC below, not the 0.684 of the superseded 40-episode
run. Those runs still regenerate from seeds; a genuinely full-corpus candidate
has not been trained yet.

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

#: The shipped predictors' scores, per tier, and the noise floor around them.
#: One baseline for all three tiers was a bug: every tier was compared against
#: MEDIUM's 0.684, so an EASY candidate scoring 0.957 was reported as +0.273
#: "above the noise floor" when against EASY's own incumbent it is +0.045, well
#: inside it. That reads as a decisive win where there is none, on the one
#: screen whose whole job is to stop exactly that.
BASELINE_AUC = {"easy": 0.957, "medium": 0.763, "hard": 0.703}
#: Measured on MEDIUM's 31x400 recipe and applied to every tier for want of a
#: per-tier measurement. The full-corpus recipe is far tighter (three MEDIUM
#: draws give sd 0.00075), so against a full-corpus candidate this is a
#: conservative bar rather than an accurate one.
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
        # "predictor_<tier>_history" -> "<tier>". A candidate has to be judged
        # against its OWN tier: the tiers sit almost 0.3 AUC apart, which dwarfs
        # the difference the comparison is trying to detect.
        parts = h.stem.split("_")
        tier = parts[1] if len(parts) > 1 else ""
        base = BASELINE_AUC.get(tier)
        if base is None:
            out.append(f"  {h.stem}: AUC {auc:.3f} AP {ap:.3f}  "
                       f"(no baseline for tier {tier!r}; compare by hand)")
            continue
        delta = auc - base
        # Report the comparison the way the project's own standard requires:
        # against the spread, not against the point estimate.
        verdict = ("above the noise floor" if abs(delta) > 2 * NOISE_SD
                   else "within run-to-run noise")
        out.append(f"  {h.stem}: AUC {auc:.3f} AP {ap:.3f}  "
                   f"({delta:+.3f} vs {tier} {base}, {verdict})")
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
