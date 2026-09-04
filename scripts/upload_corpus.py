#!/usr/bin/env python
"""Put the published corpus into Lightning storage so a job can stream it.

WHY AS ONE ARCHIVE.

``build/dataset`` is 0.85 GB spread over 9004 files. Uploading it with
``Teamspace.upload_folder`` ran at ~15 files/min -- 54 files in 3.5 minutes,
which extrapolates to a TEN HOUR upload before any training could start.
Small-file transfer is latency-bound, so the fix is fewer files, not more
bandwidth. One uncompressed tar took 10 s to build and 271 s to upload on AWS
(1446 s on GCP). No compression: parquet is already compressed, so gzip spends
CPU for nothing.

The job unpacks it to local disk before training. Reading 9004 small files over
the teamspace mount during training would reintroduce the same per-file latency
the archive exists to avoid, and the streaming path is data-bound already.

This needs re-running whenever the teamspace is recreated. Jobs and studios have
twice vanished from a teamspace mid-session; the uploaded corpus survived both
times, but a NEW teamspace starts empty.

    python scripts/upload_corpus.py
    python scripts/upload_corpus.py --keep-tar   # reuse the archive next time
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Where the job expects to find it. build_all_tiers_command untars this.
REMOTE_NAME = "uploads/smartscan-dataset.tar"


def main() -> int:
    """Archive and upload."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="build/dataset")
    ap.add_argument("--tar", default=None,
                    help="Where to write the archive (default: alongside source).")
    ap.add_argument("--remote", default=REMOTE_NAME)
    ap.add_argument("--keep-tar", action="store_true",
                    help="Do not delete the archive afterwards. Worth it -- the "
                         "teamspace may need repopulating.")
    args = ap.parse_args()

    src = REPO_ROOT / args.source
    if not (src / "index.parquet").is_file():
        print(f"no corpus at {src} (expected index.parquet). Build it first.")
        return 1

    tar_path = Path(args.tar) if args.tar else src.parent / "smartscan-dataset.tar"
    if tar_path.is_file():
        print(f"reusing existing archive {tar_path} "
              f"({tar_path.stat().st_size / 1e9:.2f} GB)")
    else:
        t0 = time.time()
        with tarfile.open(tar_path, "w") as tf:
            tf.add(src, arcname="dataset")
        print(f"archived {tar_path.stat().st_size / 1e9:.2f} GB in "
              f"{time.time() - t0:.0f}s")

    # Verify before spending twenty minutes uploading the wrong thing.
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    for required in ("dataset/index.parquet", "dataset/build_report.json"):
        if required not in names:
            print(f"archive is missing {required}; refusing to upload it")
            return 1
    print(f"verified {len(names)} members")

    from smartscan.credentials import load_dotenv

    load_dotenv(override=True)
    from lightning_sdk import Teamspace

    ts = Teamspace(name=os.environ["LIGHTNING_TEAMSPACE"],
                   user=os.environ["LIGHTNING_USERNAME"])
    print(f"uploading to {ts.name}:{args.remote} ...")
    t0 = time.time()
    ts.upload_file(str(tar_path), args.remote, progress_bar=False)
    print(f"UPLOAD OK in {time.time() - t0:.0f}s")
    print(f"\ntrain against it with:\n"
          f"  python scripts/lightning_train.py --all-tiers \\n"
          f"      --dataset /teamspace/{args.remote} \\n"
          f"      --workers '$(( $(nproc) > 3 ? $(nproc) - 2 : 2 ))' \\n"
          f"      --machine T4 --studio <studio-name>")

    if not args.keep_tar:
        tar_path.unlink()
        print(f"removed {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
