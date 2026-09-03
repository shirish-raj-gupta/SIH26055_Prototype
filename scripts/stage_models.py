#!/usr/bin/env python
"""Assemble `build/models` for the second Kaggle dataset.

`publish_kaggle.py --what models` has always known where to look, but nothing
ever put anything there, so the models dataset named in the plan was never
published. This stages what an external consumer actually needs:

* the twelve trained checkpoints, each carrying its own architecture tag so it
  loads without guessing at a config default;
* the ONNX exports, which are the artefacts that leave Python for embedded
  hardware;
* the training histories and logs, so a reader can see the runs rather than
  take the numbers on trust;
* a model card recording what each file is, how it scored, and -- for the three
  that do not work -- why it is shipped anyway.

    python scripts/stage_models.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEST = REPO_ROOT / "build" / "models"
CKPT = REPO_ROOT / "runs" / "checkpoints"
ONNX = REPO_ROOT / "runs" / "onnx"


def _predictor_row(tier: str) -> str:
    """One model-card line for a predictor, or a note that it is absent."""
    h = CKPT / f"predictor_{tier}_history.json"
    if not h.is_file():
        return f"| `predictor_{tier}.pt` | transformer | not trained | - |"
    s = json.loads(h.read_text(encoding="utf-8"))["scores_vs_truth"]
    lift = s["average_precision"] / s["positive_rate"]
    return (
        f"| `predictor_{tier}.pt` | transformer | AUC {s['auc']:.3f}, "
        f"AP {s['average_precision']:.3f} | {lift:.2f}x over a "
        f"{s['positive_rate']:.3f} base rate |"
    )


def _rl_row(kind: str, tier: str) -> str:
    """One model-card line for an RL agent."""
    t = CKPT / f"{kind}_{tier}_trainlog.json"
    if not t.is_file():
        return f"| `{kind}_{tier}.pt` | actor-critic | not trained | - |"
    d = json.loads(t.read_text(encoding="utf-8"))
    ret, ent = d["returns"][-1], d["entropy"][-1]
    note = ""
    if kind == "hybrid" and tier in {"easy", "hard"}:
        note = " **does not schedule -- see below**"
    return (
        f"| `{kind}_{tier}.pt` | actor-critic | return {ret:.1f}, "
        f"final entropy {ent:.2f} | {d['steps'][-1]:,} env steps{note} |"
    )


def build_card() -> str:
    """Write the model card, including the models that do not work."""
    rows = [_predictor_row(t) for t in ("easy", "medium", "hard")]
    rows += [_rl_row(k, t) for k in ("dqn", "ppo", "hybrid") for t in ("easy", "medium", "hard")]
    body = "\n".join(rows)
    return f"""# EW Smart Scan — Trained Scheduler Models

Companion to **ew-smart-scan-rf-environment**. Checkpoints for the learned
schedulers in [SIH PS-26055](https://github.com/shirish-raj-gupta/SIH26055_Prototype),
a closed-loop Electronic Support receiver scheduler.

## Contents

| file | architecture | result | notes |
|---|---|---|---|
{body}

`*_history.json` and `*_trainlog.json` accompany each checkpoint so the runs can
be inspected rather than taken on trust.

## ONNX

`onnx/predictor_<tier>.onnx` — opset 17, single self-contained file, weights
folded in rather than left in a sidecar. Each was verified against its torch
model on a pseudo-random probe before export was accepted; the largest
discrepancy across the three is 2.4e-06.

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("onnx/predictor_medium.onnx")
x = np.zeros((1, 4, 128, 128), dtype=np.float32)   # visit, hit, SNR, staleness
logits = sess.run(None, {{"observation_window": x}})[0]   # (1, 128)
```

## What does not work, and why it is here anyway

**The three hybrids do not learn.** Under the greedy argmax used at evaluation,
`hybrid_easy` and `hybrid_hard` tune to a *single channel* for the whole
episode — coverage 0.000 on EASY. `hybrid_hard` had 3,000,000 steps, so it is
not a budget problem; its policy entropy stays near uniform (4.42 against a
maximum of log 128 = 4.85), so the argmax is decided by a vanishing margin that
lands on the same channel every time. **A hybrid training return is not evidence
that the hybrid learned anything**: 236.0 and 253.8 both describe a policy that
parks. Raising `rl.hybrid.entropy_coef` to 0.03 rescues EASY and fails on HARD.

**`ppo` parks too**, reaching a median worst-case staleness of the full 10-second
episode on MEDIUM.

They are published because a negative result that can be reproduced is worth
more than a gap where a model should be, and because the benchmark labels any
substitution rather than hiding it.

## Reading the predictor scores

The scheduler takes an **argmax** over these probabilities — it ranks channels
and never thresholds — so AUC and AP lift over the base rate are the numbers
that matter. Accuracy is not: at a ~9 % positive rate, predicting "idle"
everywhere scores 0.91.

Run-to-run spread on this task is **sd 0.038 AUC** over four independent draws,
so a single-run difference below about 0.08 carries no information. The gap
between `predictor_easy` (0.911) and `predictor_medium` (0.683) is roughly six
standard deviations and is tier difficulty, not corpus size.

## Licence

CC BY-SA 4.0, matching the environment dataset. The simulator is MIT.
"""


def main() -> int:
    """Stage the directory."""
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "onnx").mkdir(parents=True)

    n = 0
    for pattern in ("*.pt", "*_history.json", "*_trainlog.json"):
        for f in sorted(CKPT.glob(pattern)):
            shutil.copy2(f, DEST / f.name)
            n += 1
    for f in sorted(ONNX.glob("*.onnx")):
        shutil.copy2(f, DEST / "onnx" / f.name)
        n += 1

    (DEST / "model_card.md").write_text(build_card(), encoding="utf-8")
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"staged {n} files ({total / 1e6:.1f} MB) -> {DEST}")
    print("next: python scripts/publish_kaggle.py --what models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
