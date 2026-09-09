#!/usr/bin/env python
"""Why the HARD tier resists every scheduler, measured rather than asserted.

No policy in this repository beats the plain sweep on HARD. That reads as a
failure until you ask what the ceiling actually is.

A receiver that sees ``K`` of ``B`` channels visits any given channel a
fraction ``K/B`` of the time. An emitter that is physically catchable in ``n``
slots therefore offers, in expectation, ``n * K/B`` chances across the whole
episode -- for **any** policy whose coverage is uniform, whatever cleverness
decides the order. When that number falls below 1, the emitter is a coin flip
that no amount of scheduling, training or compute can convert into a
certainty.

On HARD that is not a corner case. AgileBeamRadar has a median of about 0.8
expected chances, so a Poisson bound puts its floor miss rate near 44 % before
any scheduler has made a single decision, and it accounts for the largest
share of everyone's misses.

The second half of the argument is why prediction cannot rescue it. Estimating
a scanning emitter's period needs at least two observations of it. An emitter
offering fewer than one expected look is, in the overwhelming majority of
episodes, seen zero times or once -- so the very emitters that get missed are
the ones no predictor can have learned anything about. That is an
information-theoretic wall, not a model-capacity one.

What this script does NOT claim: that HARD is optimal as played. It reports
the classes with genuine headroom too -- ones whose Poisson floor is near zero
while real schedulers still miss them -- because those are where work would
actually pay.

Usage::

    python scripts/hard_tier_ceiling.py
    python scripts/hard_tier_ceiling.py --tiers medium,hard --n-seeds 8
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORTS = REPO_ROOT / "reports"

#: Policies compared. All are closed form or belief-driven; the point of the
#: exercise is that none of them separates from the sweep on HARD.
POLICIES = ("sequential", "coprime_sweep", "whittle", "phase_locked")


def analyse(tier: str, n_seeds: int) -> dict:
    """Expected chances per emitter class, and what each policy actually misses."""
    from smartscan.agents import build_agent
    from smartscan.config import load_config
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.hal.simulated import detection_probability_tensor
    from smartscan.runner import run_episode

    cfg = load_config(f"{tier}.yaml")
    k, b = cfg.receiver.ibw_channels, cfg.spectrum.n_channels
    duty = k / b

    chances: dict[str, list[float]] = collections.defaultdict(list)
    misses: dict[str, collections.Counter] = {p: collections.Counter() for p in POLICIES}
    totals: collections.Counter = collections.Counter()
    n_catchable = 0
    n_real = 0

    for i in range(n_seeds):
        seed = cfg.run.seed + i
        scenario = generate_scenario(seed, config=cfg)
        episode = build_episode(scenario)
        pd_tensor = detection_probability_tensor(episode, cfg)
        interceptable = pd_tensor > 0.01

        real = [t for t in episode.truth if not t.is_interferer]
        n_real += len(real)
        catchable: set[int] = set()
        for truth in real:
            cells = interceptable & (episode.emitter_id == truth.emitter_id)
            n_slots = int(cells.any(axis=0).sum())
            chances[truth.emitter_class].append(n_slots * duty)
            totals[truth.emitter_class] += 1
            if n_slots:
                catchable.add(truth.emitter_id)
        n_catchable += len(catchable)

        for policy in POLICIES:
            result = run_episode(
                cfg, seed, build_agent(policy, cfg, seed, scenario),
                scenario=scenario, episode=episode,
            )
            found = set(result.first_intercept)
            for truth in real:
                if truth.emitter_id in catchable and truth.emitter_id not in found:
                    misses[policy][truth.emitter_class] += 1

    classes = sorted(totals)
    out: dict = {
        "tier": tier, "n_seeds": n_seeds, "k": k, "b": b, "duty": duty,
        "n_real_emitters": n_real, "n_catchable": n_catchable,
        "policies": POLICIES, "classes": {},
    }
    for cls in classes:
        arr = np.asarray(chances[cls], dtype=float)
        median = float(np.median(arr))
        out["classes"][cls] = {
            "n": int(totals[cls]),
            "median_expected_chances": median,
            # Poisson with mean = expected chances: the probability of getting
            # zero looks at an emitter, for any uniform-coverage policy.
            "poisson_floor_miss_rate": float(np.exp(-median)),
            "fraction_below_one_chance": float((arr < 1).mean()),
            "misses": {p: int(misses[p][cls]) for p in POLICIES},
            "miss_rate": {p: float(misses[p][cls] / max(totals[cls], 1)) for p in POLICIES},
        }
    out["total_misses"] = {p: int(sum(misses[p].values())) for p in POLICIES}
    return out


def render(reports: list[dict]) -> str:
    """Render the ceiling analysis as markdown."""
    L: list[str] = []
    A = L.append
    A("# Why the HARD tier resists every scheduler")
    A("")
    A("_Generated by `scripts/hard_tier_ceiling.py`._")
    A("")
    A("A receiver seeing `K` of `B` channels visits any given channel a fraction")
    A("`K/B` of the time, so an emitter physically catchable in `n` slots offers")
    A("`n·K/B` expected chances across the episode — for **any** policy with uniform")
    A("coverage, however clever its ordering. Below one expected chance, catching it")
    A("is a coin flip that no scheduling, training or compute converts into a")
    A("certainty.")
    A("")

    for rep in reports:
        A(f"## `{rep['tier']}` — K/B = {rep['k']}/{rep['b']} = {rep['duty']:.4f}, "
          f"{rep['n_seeds']} seeds")
        A("")
        A("| emitter class | n | median E[chances] | Poisson floor miss | "
          "share with E<1 | " + " | ".join(f"`{p}` misses" for p in rep["policies"]) + " |")
        A("|---|---|---|---|---|" + "|".join("---" for _ in rep["policies"]) + "|")
        for cls, d in sorted(rep["classes"].items(),
                             key=lambda kv: kv[1]["median_expected_chances"]):
            A(f"| {cls} | {d['n']} | {d['median_expected_chances']:.2f} | "
              f"{d['poisson_floor_miss_rate']:.1%} | {d['fraction_below_one_chance']:.0%} | "
              + " | ".join(str(d["misses"][p]) for p in rep["policies"]) + " |")
        A("| **total** | | | | | "
          + " | ".join(f"**{rep['total_misses'][p]}**" for p in rep["policies"]) + " |")
        A("")

        # Keyed on the share of instances below one chance, not the median: a
        # class whose median sits at exactly 1.0 can still have half its
        # members under the wall, and that half is what drives the misses.
        hard_floor = [c for c, d in rep["classes"].items()
                      if d["median_expected_chances"] < 1.0
                      or d["fraction_below_one_chance"] >= 0.25]
        headroom = [
            (c, d) for c, d in rep["classes"].items()
            if d["poisson_floor_miss_rate"] < 0.05
            and min(d["miss_rate"][p] for p in rep["policies"]) > 0.05
        ]
        if hard_floor:
            A(f"**Below the wall:** {', '.join(f'`{c}`' for c in hard_floor)} — fewer than")
            A("one expected look per episode. A predictor cannot help here either: "
              "estimating")
            A("a scan period needs at least two observations, and these emitters are seen")
            A("zero times or once. The missing information was never collected.")
            A("")
        if headroom:
            A("**Where headroom is real:** " + ", ".join(
                f"`{c}` (floor {d['poisson_floor_miss_rate']:.1%}, "
                f"best policy still misses {min(d['miss_rate'][p] for p in rep['policies']):.0%})"
                for c, d in headroom))
            A("")
            A("These are catchable in principle and still missed, so they — not the")
            A("needle-in-haystack classes — are where scheduling work would pay.")
            A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="medium,hard")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--out", default=str(REPORTS / "hard_tier_ceiling"))
    args = ap.parse_args()

    reports = [analyse(t.strip(), args.n_seeds) for t in args.tiers.split(",") if t.strip()]
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(reports, indent=1, default=float), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(reports), encoding="utf-8")
    print(f"wrote {stem.with_suffix('.md')}")
    print(f"wrote {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
