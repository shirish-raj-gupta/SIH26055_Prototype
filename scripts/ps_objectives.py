#!/usr/bin/env python
"""Score every scheduler on the two objectives SIH 26055 actually names.

The problem statement is specific about what the scheduler is for:

    "Development of a robust scheduler using machine learning to **minimize
    intercept time** and **ensure a high interception rate** is the primary
    objective of the strategy."

Two objectives, both named. It does not ask for "emitters never intercepted" --
that is a useful diagnostic, and the tier-ceiling analysis shows uniform
coverage is optimal for it, but optimising it is not what was asked and a
scheduler judged only on it looks bad for the wrong reason.

This reports, per tier, the improvement of every scheduler over the **tuned**
sequential sweep on both named objectives at once, paired seed-for-seed with
bootstrap confidence intervals and a Holm correction across the family of
comparisons. A scheduler only qualifies if it beats the baseline on *both* --
winning interception rate by parking on one loud emitter while intercept time
collapses is not a smart scan strategy, and the ``predictor`` policy does
exactly that on the harder tiers.

Reads the shipped ``reports/metrics_{tier}.json``; writes
``reports/ps_objectives.{md,json}``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPORTS = REPO_ROOT / "reports"
TIERS = ("easy", "medium", "hard")

#: The tuned open-loop sweep. Beating a deliberately weak incumbent would prove
#: nothing, so this is the one whose dwell was swept and set to its best value.
BASELINE = "sequential"

#: (metric key, human name, lower_is_better) for each objective the PS names.
OBJECTIVES = (
    ("ttfi_median_s", "intercept time", True),
    ("twir_rate", "interception rate", False),
)


def _per_seed(rows: list[dict], key: str) -> dict[int, float]:
    """Seed -> value, so every comparison below stays paired."""
    out = {}
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[int(r["seed"])] = float(v)
    return out


def _delta(treat: dict, base: dict, lower_is_better: bool) -> dict:
    """Paired relative improvement of ``treat`` over ``base``, with a CI."""
    from scipy import stats

    from smartscan.analysis.metrics import paired_bootstrap_delta

    seeds = sorted(set(treat) & set(base))
    if len(seeds) < 3:
        return {}
    a = np.array([treat[s] for s in seeds])
    b = np.array([base[s] for s in seeds])

    # paired_bootstrap_delta reports positive when the treatment is LOWER.
    # That is the improvement for intercept time and the reverse for
    # interception rate, so the arguments are swapped rather than the sign
    # flipped afterwards -- the CI bounds stay the right way round.
    ci = (paired_bootstrap_delta(a, b, relative=True) if lower_is_better
          else paired_bootstrap_delta(b, a, relative=True))
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"n": len(seeds), "improvement": float(ci.point),
            "lo": float(ci.lo), "hi": float(ci.hi), "p_raw": p}


def _holm(pvals: dict[tuple, float]) -> dict[tuple, float]:
    """Holm-Bonferroni over the whole family of comparisons in a tier."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)       # enforce monotonicity
        out[k] = running
    return out


def analyse(tier: str) -> dict:
    rows = json.loads((REPORTS / f"metrics_{tier}.json").read_text(encoding="utf-8"))["rows"]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    if BASELINE not in by:
        return {}

    res: dict[str, dict] = {}
    praw: dict[tuple, float] = {}
    for agent, rs in by.items():
        if agent == BASELINE:
            continue
        res[agent] = {}
        for key, name, lower in OBJECTIVES:
            d = _delta(_per_seed(rs, key), _per_seed(by[BASELINE], key), lower)
            if d:
                res[agent][name] = d
                praw[(agent, name)] = d["p_raw"]

    for (agent, name), p in _holm(praw).items():
        res[agent][name]["p_holm"] = p
        res[agent][name]["significant"] = bool(p < 0.05)

    # Two bars, because one alone misreads the harder tiers.
    #
    #   qualifies  -- BOTH named objectives significantly improved. The strict
    #                 reading of the problem statement.
    #   acceptable -- at least one significantly improved and NEITHER
    #                 significantly worsened. On MEDIUM and HARD several
    #                 policies lift interception rate by 50%+ with intercept
    #                 time statistically unchanged; reporting only the strict
    #                 bar would score those as failures when they are a real
    #                 gain at no measured cost.
    for obj in res.values():
        names = [name for _k, name, _l in OBJECTIVES]
        got = [obj.get(n) for n in names]
        obj["qualifies"] = bool(
            obj and all(g and g.get("significant") and g["improvement"] > 0 for g in got))
        improved = any(g and g.get("significant") and g["improvement"] > 0 for g in got)
        harmed = any(g and g.get("significant") and g["improvement"] < 0 for g in got)
        obj["acceptable"] = bool(improved and not harmed)
    return {"tier": tier, "baseline": BASELINE,
            "n_seeds": len(by[BASELINE]), "agents": res}


def render(data: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# The two objectives SIH 26055 names")
    A("")
    A("> _\"Development of a robust scheduler using machine learning to **minimize")
    A("> intercept time** and **ensure a high interception rate** is the primary")
    A("> objective of the strategy.\"_")
    A("")
    A("Improvement over the **tuned** sequential sweep, paired seed-for-seed, with")
    A("95% bootstrap CIs and Holm correction across each tier's family of tests. A")
    A("scheduler **qualifies** only by improving *both* objectives significantly --")
    A("winning interception rate by parking on one loud emitter while intercept time")
    A("collapses is not a smart scan strategy.")
    A("")
    for tier in data["tiers"]:
        d = data["tiers"][tier]
        A(f"## `{tier}` — {d['n_seeds']} paired seeds vs `{d['baseline']}`")
        A("")
        A("| scheduler | intercept time | interception rate | verdict |")
        A("|---|---|---|---|")

        def cell(o, name):
            v = o.get(name)
            if not v:
                return "—"
            mark = "**" if v.get("significant") else ""
            return (f"{mark}{v['improvement']:+.1%}{mark} "
                    f"[{v['lo']:+.0%}, {v['hi']:+.0%}] p={v.get('p_holm', 1):.3g}")

        rank = sorted(
            d["agents"].items(),
            key=lambda kv: (not kv[1].get("qualifies"),
                            -(kv[1].get("interception rate", {}).get("improvement", -9))),
        )
        for agent, o in rank:
            tick = ("**both improved**" if o.get("qualifies")
                    else "improved, no harm" if o.get("acceptable") else "no")
            A(f"| `{agent}` | {cell(o, 'intercept time')} | "
              f"{cell(o, 'interception rate')} | {tick} |")
        A("")
        winners = [a for a, o in d["agents"].items() if o.get("qualifies")]
        if not winners:
            ok = [a for a, o in d["agents"].items() if o.get("acceptable")]
            if ok:
                best = max(ok, key=lambda a: d["agents"][a]["interception rate"]["improvement"])
                o = d["agents"][best]
                A(f"**Recommended for `{tier}`: `{best}`** — interception rate "
                  f"{o['interception rate']['improvement']:+.1%} (significant), intercept "
                  f"time {o['intercept time']['improvement']:+.1%} "
                  f"(p={o['intercept time'].get('p_holm', 1):.3g}, unchanged). No scheduler "
                  f"improves both significantly on this tier, but this one improves the "
                  f"rate at no measured cost in time.")
                A("")
                continue
        if winners:
            best = max(winners,
                       key=lambda a: d["agents"][a]["interception rate"]["improvement"])
            o = d["agents"][best]
            A(f"**Recommended for `{tier}`: `{best}`** — intercept time "
              f"{o['intercept time']['improvement']:+.1%}, interception rate "
              f"{o['interception rate']['improvement']:+.1%}, both significant after "
              f"Holm correction.")
        else:
            A(f"**No scheduler improves both objectives significantly on `{tier}`.**")
        A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPORTS / "ps_objectives"))
    args = ap.parse_args()

    data: dict = {"tiers": {}}
    for tier in TIERS:
        if (REPORTS / f"metrics_{tier}.json").is_file():
            d = analyse(tier)
            if d:
                data["tiers"][tier] = d
    if not data["tiers"]:
        print("no reports/metrics_*.json -- run `make benchmark` first", file=sys.stderr)
        return 1

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(data, indent=1, default=float),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(render(data), encoding="utf-8")
    print(render(data))
    print(f"wrote {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
