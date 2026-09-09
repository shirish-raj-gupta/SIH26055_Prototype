"""Measure the predictor's run-to-run spread, and check a candidate recipe.

RESULT: it does not beat it. Four independent draws of 31x400 give AUC 0.685,
0.673, 0.736, 0.644 -- mean 0.684, sd 0.038 -- against the shipped 0.683. A
single earlier run of the same recipe returned 0.767, which is above the maximum
of all four repeats: an upper-tail draw, not an improvement.

The number worth keeping is the SPREAD. sd 0.038 means a single-run difference
between two predictors has to clear roughly 0.08 before it means anything. That
retires the 12,400-vs-16,000 window comparison, which is far inside the noise,
and the diversity sweep with it (31x400 -> 0.767, 62x200 -> 0.696, 97x128 ->
0.733 all came from one run each).

SUPERSEDED IN PART: this script once read that as retiring the "more/better
data" question outright. It does not, and the scope is why -- every variation
measured here holds the window budget near 12,400, so it can only speak about
redistributing a fixed budget, not about enlarging it.

Retraining at a larger (unrecorded) seed-regenerated episode count moved MEDIUM
0.683 -> 0.763, clearing the +0.08 bar this script set, with EASY 0.911 -> 0.957
and HARD 0.673 -> 0.703 moving the same way but staying inside it. The
corroboration is the TEACHER, which improves on all three tiers: the four
repeats above vary only the student's draw, so a moved teacher is outside what
this measurement can produce, and that is what distinguishes the result from the
0.767 upper-tail draw.

MEASURED AT 300x200. Six draws on verified-disjoint blocks give AUC 0.7598,
0.7641, 0.7597, 0.7718, 0.7634, 0.7627 -- mean 0.7636, sd 0.0044. The bar at
that recipe is 2 sd = 0.0089 and the gap over the incumbent is +0.081, so it
clears its own bar ninefold. Set N_EP, WPE = 300, 200 to reproduce.

DISJOINTNESS IS NOT AUTOMATIC. Episodes are range(seed+1000, seed+1000+N_EP),
so seeds must differ by at least N_EP. An earlier pair differing by 101 at 300
episodes shared 199 of 300 and, read as replicates, made the recipe look 51x
tighter than 31x400; the honest figure over disjoint blocks is 9x. The `rep*137`
spacing below is safe only because 137 > N_EP = 31 -- raise N_EP and it is not.

What stays true is the measurement itself: within the 31x400 recipe the spread
really is sd 0.038.

It also confirms which difference IS real: predictor_easy at 0.957 against
medium at 0.763 is far outside the spread, so that is tier difficulty rather than
corpus size -- which is what the easy tier reaching the project's best AUC from
the SMALLEST corpus already suggested.

One run each is what produced the two findings this project has already had to
retract, so the candidate is repeated over DISJOINT episode blocks. The shipped
40x400 recipe cannot be rebuilt at the current memory ceiling (it needs 7.0 GB
free), so its 0.683 stays a single historical draw -- the comparison is a
replicated candidate against an unreplicated incumbent, and is reported as such.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smartscan.agents.predictors import train_predictor  # noqa: E402
from smartscan.config import load_config  # noqa: E402

N_EP, WPE, REPS = 31, 400, 4

def main() -> None:
    """Run the repeats and print the spread."""
    cfg = load_config("medium.yaml")
    aucs, lifts, best = [], [], []
    for rep in range(REPS):
        base = cfg.run.seed + 1000 + rep * 137     # disjoint, non-overlapping
        seeds = list(range(base, base + N_EP))
        _, h = train_predictor(cfg, seeds=seeds, arch="transformer",
                               max_windows_per_episode=WPE, verbose=False)
        s = h["scores_vs_truth"]
        lift = s["average_precision"] / s["positive_rate"]
        aucs.append(s["auc"])
        lifts.append(lift)
        best.append(h["best_epoch"])
        print(f"  rep{rep} (seeds {base}..{base+N_EP-1}): AUC {s['auc']:.3f}  "
              f"AP {s['average_precision']:.3f}  lift {lift:.2f}x  epoch {h['best_epoch']}",
              flush=True)
    a = np.array(aucs)
    print(f"\n  candidate 31x400: AUC mean {a.mean():.3f}  sd {a.std(ddof=1):.3f}  "
          f"min {a.min():.3f}  max {a.max():.3f}")
    print(f"  lift mean {np.mean(lifts):.2f}x")
    print("  shipped 40x400 (single draw): AUC 0.683, lift 3.67x")
    print(f"  every candidate run above shipped? {bool((a > 0.683).all())}")
    out = REPO_ROOT / "reports" / "predictor_replication.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_ep": N_EP, "wpe": WPE, "aucs": aucs, "lifts": lifts,
         "best_epochs": best, "shipped_auc": 0.683}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
