#!/usr/bin/env python
"""Publish the SmartScan dataset (and model weights) to Kaggle. Idempotent.

    python scripts/publish_kaggle.py --dry-run          # inspect, upload nothing
    python scripts/publish_kaggle.py                    # create or version
    python scripts/publish_kaggle.py --what models      # publish checkpoints

Behaviour
---------
* **First run** on a slug: ``kaggle datasets create -p <dir> --dir-mode zip``.
* **Every run after**: ``kaggle datasets version -m "<message>"``.

The script decides which by asking Kaggle whether the slug exists, so it is safe
to run repeatedly -- re-running never creates a duplicate and never fails
because the dataset already exists.

Credentials
-----------
Read from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (environment or a gitignored
``.env``), or ``~/.kaggle/kaggle.json``. **Nothing is written into the
repository, and no value is ever printed** -- only a fingerprint, so you can
confirm which key is loaded and that a rotation took effect.

A published dataset is public and effectively permanent: search engines index
it, and other people may fork it. The script therefore refuses to upload without
either an interactive confirmation or an explicit ``--yes``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smartscan.credentials import credential_status  # noqa: E402

#: Kaggle's per-user dataset storage quota, checked before an upload rather than
#: discovered during one. NOT the 20 GB figure -- that is a notebook's writable
#: disk, and attached datasets are mounted read-only against it.
MAX_BYTES = 100 * 1024**3
LICENCE = "CC-BY-SA-4.0"

TARGETS = {
    "dataset": {
        "dir": "build/dataset",
        "slug_env": "SMARTSCAN_DATASET_SLUG",
        "slug_default": "ew-smart-scan-rf-environment",
        "title": "EW Smart Scan: RF Environment Dataset",
        "subtitle": "Simulated wideband ES receiver scheduling episodes with ground truth",
        # Kaggle constrains tags twice over, and fails LOUDLY on the second:
        #  1. Only its own controlled vocabulary is accepted -- "electronic
        #     warfare", "radar", "time series" and "simulation" are all silently
        #     dropped.
        #  2. There is a hard cap on how many categories a dataset may carry;
        #     six produced "You have exceeded the max category limit" and the
        #     whole version was rejected after a 675 MB upload.
        # Four, all valid, is comfortably inside both.
        "keywords": [
            "signal processing", "reinforcement learning",
            "time series analysis", "computer science",
        ],
    },
    "models": {
        "dir": "build/models",
        "slug_env": "SMARTSCAN_MODELS_SLUG",
        "slug_default": "ew-smart-scan-models",
        "title": "EW Smart Scan: Trained Scheduler Models",
        "subtitle": "Predictor and RL checkpoints, plus ONNX exports, for SIH 26055",
        "keywords": [
            "reinforcement learning", "signal processing", "deep learning",
        ],
    },
}


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, returning the completed process without raising."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def kaggle_available() -> tuple[bool, str]:
    """Whether the Kaggle CLI is installed and importable.

    Returns:
        ``(ok, detail)``.
    """
    proc = _run([sys.executable, "-m", "kaggle", "--version"])
    if proc.returncode == 0:
        return True, proc.stdout.strip() or "kaggle CLI present"
    return False, (
        "The Kaggle CLI is not installed.\n"
        '  pip install "smartscan[kaggle]"    (or: pip install kaggle kagglehub)'
    )


def dataset_exists(slug: str) -> bool:
    """Whether a dataset slug already exists on Kaggle.

    Args:
        slug: Full ``owner/name`` slug.

    Returns:
        True if Kaggle reports files for it.
    """
    proc = _run([sys.executable, "-m", "kaggle", "datasets", "files", slug])
    return proc.returncode == 0 and "404" not in (proc.stdout + proc.stderr)


def directory_size(path: Path) -> int:
    """Total bytes of every file beneath ``path``."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def write_metadata(target: Path, slug: str, spec: dict) -> Path:
    """Write ``dataset-metadata.json``, which the Kaggle CLI requires.

    Args:
        target: Directory to publish.
        slug: Full ``owner/name`` slug.
        spec: One entry from :data:`TARGETS`.

    Returns:
        Path to the metadata file.
    """
    meta: dict[str, Any] = {
        "title": spec["title"],
        "subtitle": spec["subtitle"],
        "id": slug,
        "licenses": [{"name": LICENCE}],
        "keywords": spec["keywords"],
    }
    # A description and per-file notes are what Kaggle's usability score is
    # actually measuring; without them a complete dataset still reads as
    # half-finished. The description is the dataset card's own summary, so the
    # page and the shipped card cannot disagree.
    card = target / "dataset_card.md"
    if card.is_file():
        meta["description"] = _description_from_card(card)
    resources = [
        {"path": f.name, "description": _FILE_NOTES.get(f.name, "")}
        for f in sorted(target.iterdir())
        if f.is_file() and f.name != "dataset-metadata.json"
    ]
    if resources:
        meta["resources"] = resources
    path = target / "dataset-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


#: Per-file notes shown on the Kaggle data tab.
_FILE_NOTES: dict[str, str] = {
    "index.parquet": (
        "One row per episode: id, tier, scenario seed, split, emitter count, "
        "occupancy fraction, config hash and relative path. Start here."
    ),
    "dataset_card.md": (
        "Full schema, units, generation parameters, provenance and known "
        "limitations. Read before using the data."
    ),
    "build_report.json": (
        "Provenance: simulator source digest, dataset schema version, build "
        "time and byte accounting."
    ),
    "episodes.zip": (
        "Per-episode directories, each holding truth_occupancy.npz (bit-packed "
        "ground truth), emitter_manifest.parquet (the order of battle) and "
        "observations.parquet (replayed receiver traces from 7 schedulers)."
    ),
}


def _description_from_card(card: Path) -> str:
    """Build the Kaggle page description from the shipped dataset card.

    Reuses the card verbatim so the published page and the file inside the
    archive cannot drift apart.

    Args:
        card: Path to ``dataset_card.md``.

    Returns:
        Markdown description, truncated to Kaggle's practical limit.
    """
    text = card.read_text(encoding="utf-8")
    # Drop the H1; Kaggle renders the title separately.
    body = text.split(chr(10), 1)[1].lstrip() if text.startswith("#") else text
    return body[:48_000]


def preflight(target: Path, spec: dict) -> list[str]:
    """Check everything that can be checked before touching the network.

    Args:
        target: Directory to publish.
        spec: One entry from :data:`TARGETS`.

    Returns:
        Problems found; empty means ready to publish.
    """
    problems: list[str] = []
    if not target.is_dir():
        return [f"{target} does not exist. Run `make dataset` first."]

    size = directory_size(target)
    if size == 0:
        problems.append(f"{target} is empty")
    if size > MAX_BYTES:
        problems.append(
            f"{size / 1024**3:.2f} GB exceeds Kaggle's "
            f"{MAX_BYTES / 1024**3:.0f} GB per-user dataset quota"
        )

    if spec is TARGETS["dataset"]:
        for required in ("index.parquet", "dataset_card.md"):
            if not (target / required).is_file():
                problems.append(f"missing {required}")

    # A credential inside the upload would be published to the world.
    for pattern in ("kaggle.json", ".env", "*.pem", "*.key", "access_token"):
        leaked = list(target.rglob(pattern))
        if leaked:
            problems.append(
                f"REFUSING TO PUBLISH: {len(leaked)} credential-like file(s) matching "
                f"{pattern!r} inside {target} (first: {leaked[0]})"
            )
    return problems


def publish(
    what: str = "dataset",
    message: str | None = None,
    dry_run: bool = False,
    assume_yes: bool = False,
    public: bool = True,
) -> int:
    """Create or version a Kaggle dataset.

    Args:
        what: ``'dataset'`` or ``'models'``.
        message: Version message; a timestamp is used if omitted.
        dry_run: Do everything except the upload.
        assume_yes: Skip the interactive confirmation.
        public: Publish publicly. Kaggle datasets created private can be made
            public later, but not the reverse without deleting.

    Returns:
        Process exit code.
    """
    import os
    import time

    spec = TARGETS[what]
    target = REPO_ROOT / spec["dir"]

    status = credential_status()
    print(status.report())
    print()

    problems = preflight(target, spec)
    if problems:
        print("preflight FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    size = directory_size(target)
    n_files = sum(1 for f in target.rglob("*") if f.is_file())
    print(f"target      {target}")
    print(f"contents    {n_files} files, {size / 1024**3:.3f} GB "
          f"({100 * size / MAX_BYTES:.2f}% of the {MAX_BYTES / 1024**3:.0f} GB quota)")

    if not status.kaggle:
        print("\nNo Kaggle credentials configured; cannot publish.")
        print("  cp .env.example .env   then fill in KAGGLE_USERNAME and KAGGLE_KEY")
        return 1

    ok, detail = kaggle_available()
    if not ok:
        print(f"\n{detail}")
        return 1

    # The slug needs an owner name. A single API token authenticates but does
    # not carry a username, so ask Kaggle who we are rather than guessing.
    owner = os.environ.get("KAGGLE_USERNAME", "")
    if not owner or owner.startswith("("):
        whoami = _run([sys.executable, "-m", "kaggle", "config", "view"])
        for line in (whoami.stdout or "").splitlines():
            if "username" in line.lower():
                owner = line.split(":")[-1].strip().strip("'\"")
                break
    if not owner:
        print(
            "\nCould not determine your Kaggle username, which the dataset slug needs.\n"
            "  Set KAGGLE_USERNAME in .env, or run `kaggle auth login`."
        )
        return 1
    slug = f"{owner}/{os.environ.get(spec['slug_env'], spec['slug_default'])}"
    meta_path = write_metadata(target, slug, spec)
    print(f"slug        {slug}")
    print(f"metadata    {meta_path.relative_to(REPO_ROOT)}")
    print(f"licence     {LICENCE}")
    print(f"visibility  {'PUBLIC' if public else 'private'}")

    exists = dataset_exists(slug)
    action = "version" if exists else "create"
    print(f"action      {action}  (slug {'already exists' if exists else 'is new'})")

    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    if not assume_yes:
        print(
            "\nThis publishes to a PUBLIC Kaggle page. Public datasets are indexed by\n"
            "search engines and may be forked by others; deleting one later does not\n"
            "un-publish what has already been copied."
        )
        if input(f"Proceed with `kaggle datasets {action}`? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return 130

    msg = message or f"SmartScan {what} {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    if exists:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "version",
               "-p", str(target), "-m", msg, "--dir-mode", "zip"]
    else:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "create",
               "-p", str(target), "--dir-mode", "zip"]
        if public:
            cmd.append("--public")

    print(f"\n$ {' '.join(cmd[2:])}")
    proc = _run(cmd, cwd=REPO_ROOT)
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    if proc.returncode == 0:
        print(f"\nhttps://www.kaggle.com/datasets/{slug}")
    return proc.returncode


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=sorted(TARGETS), default="dataset")
    parser.add_argument("-m", "--message", default=None, help="Version message.")
    parser.add_argument("--dry-run", action="store_true", help="Check everything, upload nothing.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--private", action="store_true", help="Publish privately.")
    args = parser.parse_args()
    return publish(
        what=args.what, message=args.message, dry_run=args.dry_run,
        assume_yes=args.yes, public=not args.private,
    )


if __name__ == "__main__":
    raise SystemExit(main())
