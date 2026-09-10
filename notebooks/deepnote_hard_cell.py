"""HARD tier on the full corpus — the version that survives a machine stop.

Paste into a Deepnote cell and run. ~3-4 hours on an L4.

WHAT WENT WRONG LAST TIME, and what changed here.

The first run reached student epoch 9/12 with a best validation AP of 0.8086
against the shipped model's 0.1021, and then the machine stopped and the model
was lost. The checkpoints were being written to /root, which is local disk that
dies with the machine, and the copy to /work was the LAST step of the cell, so
it never ran. Seven hours produced a number and no artifact.

So: ``run.out_dir`` now points at /work. The trainer already checkpoints as it
goes and keeps the best-validating weights, so with the destination on
persistent storage a machine stop at any point leaves the best model so far
sitting on disk. Nothing depends on reaching the end of the cell any more.

The run is also much shorter, because the first one showed where the value is:

  teacher  0.0337 -> 0.0313 by epoch 3, then 0.0312 at epoch 10.
           Seven epochs bought 0.0001 and about 2.5 hours. Now 3.
  student  ap 0.7978 at epoch 1, 0.8086 at epoch 7. Six epochs bought 1.4%.
           Now 6, which keeps essentially all of it.

No Kaggle credentials needed: the 676 MB archive is on /work already.
"""
import json
import os
import pathlib
import subprocess
import time
import zipfile

T0 = time.time()
ARCHIVE = "/work/kagglehub_cache/datasets/shirishrajgupta/ew-smart-scan-rf-environment/3.archive"
DS = "/root/ds"          # local: /work is s3fs, 9003 files took >15 min there vs 10 s here
REPO = "/root/repo"
OUT = "/work/trainout"   # persistent: this is the whole point


def sh(cmd, check=False):
    """Run a shell command, streaming output into the cell."""
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"  -> exit {rc}", flush=True)
    if check and rc:
        raise SystemExit(f"failed: {cmd}")
    return rc


# ---- 0. what survived from last time -----------------------------------
print("=" * 62 + "\nSTATE\n" + "=" * 62, flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh(f"ls -la {OUT}/checkpoints/ 2>/dev/null || echo '(no previous checkpoints)'")
if not pathlib.Path(ARCHIVE).exists():
    raise SystemExit(
        f"The corpus archive is gone from {ARCHIVE}.\n"
        "Re-download needs KAGGLE_USERNAME and KAGGLE_KEY set as Deepnote\n"
        "environment variables (Integrations -> Environment variables), then\n"
        "run: pip install kagglehub && python -c \"import kagglehub;"
        "kagglehub.dataset_download('shirishrajgupta/ew-smart-scan-rf-environment')\""
    )

# ---- 1. corpus onto local disk -----------------------------------------
sh(f"rm -rf {DS} && mkdir -p {DS}")
print(f"extracting {ARCHIVE} -> {DS}", flush=True)
with zipfile.ZipFile(ARCHIVE) as z:
    z.extractall(DS)
found = list(pathlib.Path(DS).rglob("index.parquet"))
if not found:
    raise SystemExit("no index.parquet — the corpus did not unpack")
DS_ROOT = str(found[0].parent)
print("corpus root:", DS_ROOT, flush=True)

# ---- 2. code and deps --------------------------------------------------
sh(f"rm -rf {REPO} && git clone --depth 1 "
   f"https://github.com/shirish-raj-gupta/SIH26055_Prototype.git {REPO}", check=True)
sh("python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
sh(f"cd {REPO} && python -m pip install -q -e '.[ml,viz]'", check=True)

# Seed the persistent output dir with the shipped model, so the NEW/SHIP
# comparison at the end has something to compare against.
sh(f"mkdir -p {OUT}/checkpoints")
for f in ("predictor_hard.pt", "predictor_hard_history.json"):
    src = pathlib.Path(REPO) / "runs" / "checkpoints" / f
    dst = pathlib.Path(OUT) / "checkpoints" / f.replace(".", "_shipped.", 1)
    if src.exists() and not dst.exists():
        dst.write_bytes(src.read_bytes())
        print(f"kept shipped {f} -> {dst.name}", flush=True)

# ---- 3. prove the corpus is real before spending hours ------------------
if sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage data --dataset-root {DS_ROOT}"):
    raise SystemExit("corpus check failed — refusing to train on regenerated data")

# ---- 4. train, writing checkpoints straight to /work -------------------
# batch 64 OOM'd on the L4 (9.58 GiB wanted, 8.57 GiB reserved-but-unallocated),
# so expandable segments plus a ladder rather than one guess.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
trained = False
for bs in (32, 16, 8):
    print(f"\n{'=' * 62}\npredictor, batch_size={bs}\n{'=' * 62}", flush=True)
    rc = sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"python -m smartscan.cli train --config hard.yaml --what predictor "
            f"--dataset {DS_ROOT} --windows-per-episode 120 --workers 16 --steps 6 "
            f"--set predictor.batch_size={bs} "
            f"--set predictor.distillation.teacher_epochs=3 "
            f"--set run.out_dir={OUT}")
    if rc == 0:
        trained = True
        break
    print(f"batch_size={bs} failed (likely OOM); trying smaller", flush=True)

print(f"\ntrained={trained} after {(time.time() - T0) / 60:.1f} min", flush=True)
sh(f"ls -la {OUT}/checkpoints/")


# ---- 5. did it get better? ---------------------------------------------
def _scores(path):
    """(auc, ap, accuracy, ap-lift) from a predictor history file, or None."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    s = h.get("scores_vs_truth") or {}
    return (s.get("auc"), s.get("average_precision"),
            s.get("accuracy"), h.get("ap_lift_over_base_rate"),
            s.get("positive_rate"), h.get("best_val_ap"))


print("\n" + "=" * 62 + "\nPREDICTOR QUALITY\n" + "=" * 62, flush=True)
for tag, fname in (("NEW ", "predictor_hard_history.json"),
                   ("SHIP", "predictor_hard_history_shipped.json")):
    sc = _scores(pathlib.Path(OUT) / "checkpoints" / fname)
    if sc is None:
        print(f"{tag} (missing {fname})", flush=True)
    else:
        auc, ap, acc, lift, base, bva = (x if x is not None else float("nan") for x in sc)
        print(f"{tag} auc={auc:.4f} ap={ap:.4f} acc={acc:.4f} lift={lift:.2f}x "
              f"base={base:.4f} best_val_ap={bva:.4f}", flush=True)

# ---- 6. the number that actually decides it ----------------------------
# Not the loss and not the AP. On MEDIUM, retraining raised AUC 0.684 -> 0.763
# and made the scheduler WORSE: never-intercepted went 112 -> 126. The bar is
# `sequential` at 138 on HARD, which nothing learned has beaten.
sh(f"cp -f {OUT}/checkpoints/predictor_hard.pt {REPO}/runs/checkpoints/ 2>/dev/null")
sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage evaluate --n-seeds 8")
sh(f"cp -v {REPO}/reports/hard_retrain_eval.json {OUT}/ 2>/dev/null")
sh(f"ls -la {OUT}/")
print(f"\n=== DONE in {(time.time() - T0) / 60:.1f} min ===", flush=True)
print(f"Everything worth keeping is under {OUT} on persistent storage.", flush=True)
