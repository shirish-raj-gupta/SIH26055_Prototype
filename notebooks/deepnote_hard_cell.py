# =====================================================================
# SmartScan — HARD tier on the FULL published corpus.
# Paste this into a Deepnote cell and run it. ~2-3 hours on an L4.
#
# Run it from the Deepnote UI, not from an API session. An interactive
# session expires after 15 idle minutes and takes the machine with it,
# which is how an earlier attempt lost 18 minutes of training with
# nothing saved. A run you start in the UI is not tied to any session.
#
# No Kaggle credentials needed: the 676 MB archive is already sitting on
# /work from the earlier download. If it has been cleared, the fallback
# at the bottom of the extract step uses kagglehub instead, which does
# need KAGGLE_USERNAME / KAGGLE_KEY set as Deepnote environment vars.
#
# What this changes versus the shipped checkpoint: it trains on 557 real
# HARD episodes instead of ~40 regenerated from seeds. The window count
# is deliberately kept close to the shipped recipe so the comparison
# isolates episode diversity rather than confounding it with data volume.
"""Paste-into-Deepnote cell: HARD tier training on the full corpus."""
import json
import os
import pathlib
import subprocess
import time
import zipfile

T0 = time.time()
ARCHIVE = "/work/kagglehub_cache/datasets/shirishrajgupta/ew-smart-scan-rf-environment/3.archive"
DS = "/root/ds"        # local disk: /work is s3fs and unpacking 9003 files there took >15 min
REPO = "/root/repo"    # local disk too, for the same reason


def sh(cmd, check=False):
    """Run a shell command, streaming its output into the cell."""
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"  -> exit {rc}", flush=True)
    if check and rc:
        raise SystemExit(f"failed: {cmd}")
    return rc


# ---- 0. what are we on -------------------------------------------------
sh("nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader")
sh("nproc && free -g | head -2 && df -h / | tail -1")

# ---- 1. corpus onto LOCAL disk ----------------------------------------
sh(f"rm -rf {DS} && mkdir -p {DS}")
if pathlib.Path(ARCHIVE).exists():
    print(f"extracting {ARCHIVE} -> {DS}", flush=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(DS)
else:
    print("archive gone from /work; falling back to kagglehub", flush=True)
    sh("python -m pip install -q kagglehub", check=True)
    import kagglehub
    src = kagglehub.dataset_download("shirishrajgupta/ew-smart-scan-rf-environment")
    sh(f"cp -r {src}/* {DS}/")

found = list(pathlib.Path(DS).rglob("index.parquet"))
print("index.parquet:", found[:2], flush=True)
if not found:
    raise SystemExit("no index.parquet — the corpus did not unpack")
# The dataset root is whatever directory holds index.parquet.
DS_ROOT = str(found[0].parent)
print("corpus root:", DS_ROOT, flush=True)

# ---- 2. code and deps --------------------------------------------------
sh(f"rm -rf {REPO} && git clone --depth 1 "
   f"https://github.com/shirish-raj-gupta/SIH26055_Prototype.git {REPO}", check=True)
sh("python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
sh(f"cd {REPO} && python -m pip install -q -e '.[ml,viz]'", check=True)
sh("python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())\"")

# ---- 3. prove the corpus is real before spending hours on it -----------
# This refuses to continue if the loader silently falls back to
# regenerating episodes from seeds, which is the exact sampling the whole
# exercise exists to avoid.
if sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage data --dataset-root {DS_ROOT}"):
    raise SystemExit("corpus check failed — refusing to train on regenerated data")

# ---- 4. train ----------------------------------------------------------
# batch 64 hit CUDA OOM on the L4 (9.58 GiB requested, 8.57 GiB reserved
# but unallocated — fragmentation), so: expandable segments, and a ladder
# down through batch sizes rather than one guess.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
trained = False
for bs in (32, 16, 8):
    print(f"\n{'=' * 60}\npredictor, batch_size={bs}\n{'=' * 60}", flush=True)
    rc = sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"python -m smartscan.cli train --config hard.yaml --what predictor "
            f"--dataset {DS_ROOT} --windows-per-episode 120 --workers 16 --steps 12 "
            f"--set predictor.batch_size={bs}")
    if rc == 0:
        trained = True
        break
    print(f"batch_size={bs} failed (likely OOM); trying smaller", flush=True)

print(f"\ntrained={trained} after {(time.time() - T0) / 60:.1f} min", flush=True)

# ---- 5. did it actually get better? ------------------------------------
# Read the histories directly rather than shelling out: this cell is already
# Python, and the nested quoting of a `python -c` one-liner is a good way to
# ship a comparison that silently prints nothing.
def _scores(path):
    """(auc, ap, accuracy, ap-lift) from a predictor history file."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    s = h.get("scores_vs_truth") or {}
    return (s.get("auc"), s.get("average_precision"),
            s.get("accuracy"), h.get("ap_lift_over_base_rate"))


print("\n" + "=" * 60 + "\nPREDICTOR QUALITY\n" + "=" * 60, flush=True)
ck = pathlib.Path(REPO) / "runs" / "checkpoints"
for tag, fname in (("NEW ", "predictor_hard_history.json"),
                   ("SHIP", "predictor_hard_history_shipped.json")):
    sc = _scores(ck / fname)
    if sc is None:
        print(f"{tag} (missing {fname})", flush=True)
    else:
        auc, ap, acc, lift = (x if x is not None else float("nan") for x in sc)
        print(f"{tag} auc={auc:.4f}  ap={ap:.4f}  acc={acc:.4f}  lift={lift:.2f}", flush=True)

# The number that decides it. Not the loss, not the training return:
# dqn_hard's return went -570 -> +312 over 3M steps and it still missed
# more emitters than a plain sweep.
sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage evaluate --n-seeds 8")

# ---- 6. keep the results (local disk dies with the machine) ------------
sh("mkdir -p /work/results")
sh(f"cp -v {REPO}/runs/checkpoints/predictor_hard*.pt "
   f"{REPO}/runs/checkpoints/predictor_hard*.json /work/results/ 2>/dev/null")
sh(f"cp -v {REPO}/reports/hard_retrain_eval.json /work/results/ 2>/dev/null")
sh("ls -la /work/results/")
print(f"\n=== DONE in {(time.time() - T0) / 60:.1f} min ===", flush=True)
