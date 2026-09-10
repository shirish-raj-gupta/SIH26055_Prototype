#!/usr/bin/env python
"""Which scheduler to fly on which tier, and why the answer changes.

There is no single best scheduler here, and saying so is not a hedge -- the
optimum moves with emitter density, in a direction the physics predicts.

A receiver seeing ``K`` of ``B`` channels can cover the band on a fixed
revisit period. Whether that is enough depends on how many emitters have to be
found in it. On EASY, five emitters, coverage is nearly free: every sweep and
every bandit reaches zero never-intercepted, so the coverage constraint is
slack and the whole slot budget can be spent on exploitation -- which is why
``ppo`` and ``predictor`` reach TWIR an order of magnitude above the sweep.
On HARD, thirty emitters with five decoys, coverage is the binding constraint,
and every slot the predictor takes is a slot removed from finding an emitter
nobody has seen yet. There, the plain sweep wins and every learned policy
loses in proportion to how much of the schedule it controls.

So the recommendation is a frontier, not a winner. For each tier this reports
the Pareto-optimal policies over (never-intercepted, TWIR): the ones where you
cannot improve one without giving up the other. Which point on that frontier
to fly is a mission decision -- find everything once, or collect as much as
possible from what you have found -- not something a benchmark can settle.

Reads the shipped ``reports/metrics_{tier}.json``; writes
``reports/tier_policy.{md,json}``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

# The report uses arrows and em dashes; a Windows console defaults to cp1252
# and raises UnicodeEncodeError on them, killing the script after the files are
# already written. The terminal's encoding should not decide whether this runs.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
REPORTS = REPO_ROOT / "reports"
TIERS = ("easy", "medium", "hard")


def _aggregate(tier: str) -> dict[str, dict[str, float]]:
    """Per-agent summary across seeds for one tier."""
    rows = json.loads((REPORTS / f"metrics_{tier}.json").read_text(encoding="utf-8"))["rows"]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)

    def med(rs, key):
        xs = [r.get(key) for r in rs
              if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))]
        return float(np.median(xs)) if xs else float("nan")

    def per_seed(rs, key):
        """Seed-aligned values, so comparisons can stay paired."""
        return [
            (r.get("seed"),
             float(r.get(key)) if isinstance(r.get(key), (int, float)) else float("nan"))
            for r in sorted(rs, key=lambda x: x.get("seed", 0))
        ]

    out = {}
    for agent, rs in by.items():
        out[agent] = {
            "n_seeds": len(rs),
            # Kept seed by seed: a summed count and a median hide whether a gap
            # is a real effect or one bad scenario, and every comparison below
            # is paired on seed to remove scenario variance.
            "per_seed_never": per_seed(rs, "n_never_intercepted"),
            "per_seed_twir": per_seed(rs, "twir_rate"),
            # Summed, not averaged: it is a count of emitters, and the total
            # over the grid is what the leaderboard reports.
            "never_intercepted": float(sum(r.get("n_never_intercepted", 0) for r in rs)),
            "twir": med(rs, "twir_rate"),
            "ttfi_s": med(rs, "ttfi_median_s"),
            "coverage": med(rs, "coverage"),
            "reward": med(rs, "reward_total"),
        }
    return out


def _pareto(agents: dict[str, dict[str, float]]) -> list[str]:
    """Agents not dominated on (never-intercepted down, TWIR up)."""
    keep = []
    for a, va in agents.items():
        if not np.isfinite(va["twir"]):
            continue
        dominated = any(
            b != a
            and np.isfinite(vb["twir"])
            and vb["never_intercepted"] <= va["never_intercepted"]
            and vb["twir"] >= va["twir"]
            and (vb["never_intercepted"] < va["never_intercepted"] or vb["twir"] > va["twir"])
            for b, vb in agents.items()
        )
        if not dominated:
            keep.append(a)
    return sorted(keep, key=lambda a: agents[a]["never_intercepted"])


def _paired(agents: dict, a: str, b: str, key: str, lower_is_better: bool) -> dict:
    """Paired bootstrap + Wilcoxon of ``a`` against ``b`` on one metric.

    Paired on seed, so scenario variance cancels: the same 30 worlds are shown
    to both schedulers, and only the policy differs.
    """
    from scipy import stats

    from smartscan.analysis.metrics import paired_bootstrap_delta

    xa = dict(agents[a][key])
    xb = dict(agents[b][key])
    seeds = sorted(set(xa) & set(xb))
    va = np.array([xa[s] for s in seeds], float)
    vb = np.array([xb[s] for s in seeds], float)
    ok = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[ok], vb[ok]
    if va.size < 3:
        return {"n": int(va.size)}

    # paired_bootstrap_delta reports positive when the treatment scores LOWER,
    # which is an improvement only for a lower-is-better metric. Flip the sign
    # for TWIR so "positive means better" holds either way.
    # A relative improvement is undefined when the baseline is zero, and on
    # EASY it is zero on most seeds -- reporting "+0.0% ... significant" is
    # both meaningless and self-contradictory. Absolute is always defined, so
    # it leads, and the relative figure is only quoted when it means something.
    ci = paired_bootstrap_delta(va, vb, relative=False)
    rel = None
    if np.all(vb > 0):
        rel = paired_bootstrap_delta(va, vb, relative=True)
    sign = 1.0 if lower_is_better else -1.0
    try:
        w = stats.wilcoxon(va, vb)
        pval = float(w.pvalue)
    except ValueError:                      # all differences zero
        pval = 1.0
    def flip(c):
        return (sign * float(c.point),
                sign * float(c.hi if sign < 0 else c.lo),
                sign * float(c.lo if sign < 0 else c.hi))

    abs_pt, abs_lo, abs_hi = flip(ci)
    out = {
        "n": int(va.size),
        "median_a": float(np.median(va)),
        "median_b": float(np.median(vb)),
        "abs": abs_pt, "abs_lo": abs_lo, "abs_hi": abs_hi,
        # The per-seed median can be 0 while the test is significant: most
        # seeds tie and a handful differ consistently. Totals make that
        # legible instead of looking like a contradiction.
        "total_a": float(va.sum()), "total_b": float(vb.sum()),
        "p": pval,
        "significant": bool(pval < 0.05),
    }
    if rel is not None:
        r_pt, r_lo, r_hi = flip(rel)
        out.update({"rel": r_pt, "rel_lo": r_lo, "rel_hi": r_hi})
    return out


#: Equivalence margin, emitters per seed. Two schedulers count as
#: interchangeable on coverage only if the 95% CI on their paired difference
#: lies entirely inside +/- this. Per-seed never-intercepted medians run 0
#: (EASY) to 6 (HARD), so half an emitter is a tight bound on the tiers where
#: the choice is live.
EQUIV_MARGIN = 0.5


def _tied_best(agents: dict, usable: dict, margin: float = EQUIV_MARGIN):
    """Best on never-intercepted, resolving genuine ties on TWIR.

    Ranking on the raw count can pick a winner a paired test cannot separate
    from the runner-up -- HARD has `sequential` 138 against `coprime_sweep`
    139 at p=0.977 -- so the count alone is partly noise.

    But the fix is not "choose whatever is not significantly worse". A
    non-significant result is not evidence of equivalence, it is often just
    low power, and selecting on it would systematically prefer whichever
    option has the better secondary metric. On EASY that rule promoted
    `predictor` -- 5 never-intercepted against ten schedulers at 0 -- purely
    because 5 events across 30 seeds fails to reach p<0.05.

    So equivalence is tested directly: the 95% CI on the paired difference
    must lie entirely within +/- ``margin`` emitters per seed. That is a
    two-one-sided-tests bound, and it can fail in two different ways -- worse,
    or simply inconclusive -- which are not the same thing and are not
    conflated here. Only genuinely equivalent policies compete on TWIR.
    """
    ranked = sorted(usable, key=lambda a: usable[a]["never_intercepted"])
    leader = ranked[0]
    tied, inconclusive = [leader], []
    for a in ranked[1:]:
        c = _paired(agents, leader, a, "per_seed_never", lower_is_better=True)
        lo, hi = c.get("abs_lo"), c.get("abs_hi")
        if lo is None or hi is None:
            continue
        # Two conditions, because the CI alone is not enough. On EASY the
        # leader misses 0 while `ppo` misses 8; 8/30 = 0.27 per seed sits
        # inside any margin worth setting, so a CI test alone calls 8
        # equivalent to 0. Nobody needs a hypothesis test to rank 8 against 0.
        # So the totals must also be close in relative terms, with +1 slack so
        # that a single emitter is never decisive.
        total_ok = usable[a]["never_intercepted"] <= (
            usable[leader]["never_intercepted"] * 1.05 + 1.0)
        if abs(lo) <= margin and abs(hi) <= margin and total_ok:
            tied.append(a)                       # equivalent on both readings
        elif not c.get("significant"):
            inconclusive.append(a)               # cannot tell either way
    best = max(tied, key=lambda a: usable[a]["twir"])
    return best, tied, inconclusive


def render(data: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Which scheduler for which tier")
    A("")
    A("_Generated by `scripts/tier_policy.py` from the shipped benchmark._")
    A("")
    A("The optimum moves with emitter density, and it moves in the direction the")
    A("physics predicts. A receiver seeing `K` of `B` channels covers the band on a")
    A("fixed revisit period; whether that suffices depends on how many emitters must")
    A("be found in it.")
    A("")
    A("- **EASY** — 5 emitters. Coverage is nearly free: every sweep and bandit reaches")
    A("  zero never-intercepted, so the constraint is slack and the slot budget can go")
    A("  to exploitation.")
    A("- **MEDIUM** — 15 emitters. The constraint starts to bind; the frontier has a")
    A("  real trade-off along it.")
    A("- **HARD** — 30 emitters, 5 decoys. Coverage is the binding constraint. Every")
    A("  slot a predictor takes is a slot not spent finding an emitter nobody has seen,")
    A("  and emitter channels are drawn uniformly at random, so before first contact")
    A("  there is nothing to predict.")
    A("")
    for tier in data["tiers"]:
        d = data["tiers"][tier]
        A(f"## `{tier}`")
        A("")
        A("| scheduler | never intercepted | per-seed (med, min-max) | TWIR | ttfi (s) |")
        A("|---|---|---|---|---|")
        for a in d["pareto"]:
            v = d["agents"][a]
            xs = np.array([x for _, x in v["per_seed_never"]], float)
            xs = xs[np.isfinite(xs)]
            spread = (f"{np.median(xs):.1f}, {xs.min():.0f}-{xs.max():.0f}"
                      if xs.size else "-")
            A(f"| **`{a}`** | {v['never_intercepted']:.0f} | {spread} | "
              f"{v['twir']:.5f} | {v['ttfi_s']:.4f} |")
        A("")
        A(f"_Pareto-optimal over (never-intercepted, TWIR) from {len(d['agents'])} "
          f"schedulers, {d['n_seeds']} seeds. Everything else is beaten on both axes._")
        A("")
        tied = d.get("tied_group") or [d["best_coverage"]]
        A(f"- **Find everything once** → `{d['best_coverage']}` "
          f"({d['agents'][d['best_coverage']]['never_intercepted']:.0f} never intercepted"
          f", TWIR {d['agents'][d['best_coverage']]['twir']:.5f})")
        if len(tied) > 1:
            others = ", ".join(f"`{a}` ({d['agents'][a]['never_intercepted']:.0f}, "
                               f"TWIR {d['agents'][a]['twir']:.5f})"
                               for a in tied if a != d["best_coverage"])
            A(f"  - equivalent on coverage within ±{EQUIV_MARGIN} emitters/seed "
              f"(95% CI): {others}. Chosen on TWIR.")
        inc = d.get("inconclusive") or []
        if inc:
            A("  - _inconclusive_ (neither separable nor equivalent at 30 seeds): "
              + ", ".join(f"`{a}` ({d['agents'][a]['never_intercepted']:.0f})" for a in inc)
              + ". More seeds would be needed to rank these.")
        A(f"- **Collect the most from what is found** → `{d['best_twir']}` "
          f"(TWIR {d['agents'][d['best_twir']]['twir']:.5f}, "
          f"{d['agents'][d['best_twir']]['never_intercepted']:.0f} never intercepted)")
        cov, twir = d["best_coverage"], d["best_twir"]
        st = d.get("stats") or {}
        for label, key in (("vs the runner-up on coverage", "runner_up"),
                           ("coverage pick vs TWIR pick", "cov_vs_twir")):
            c = st.get(key)
            if not c or "abs" not in c:
                continue
            verdict = "significant" if c["significant"] else "NOT significant"
            rel = (f", {c['rel']:+.1%} relative" if "rel" in c else "")
            A(f"- _{label} ({c['names']}), paired over {c['n']} seeds:_ "
              f"**{c['abs']:+.2f} emitters/seed** never-intercepted "
              f"(95% CI [{c['abs_lo']:+.2f}, {c['abs_hi']:+.2f}]{rel}); "
              f"totals {c['total_a']:.0f} vs {c['total_b']:.0f}, "
              f"Wilcoxon p={c['p']:.3g} — **{verdict}**")
        if cov == twir:
            A(f"- Both at once: `{cov}` dominates on this tier.")
        else:
            dn = d["agents"][twir]["never_intercepted"] - d["agents"][cov]["never_intercepted"]
            ratio = (d["agents"][twir]["twir"] / d["agents"][cov]["twir"]
                     if d["agents"][cov]["twir"] else float("nan"))
            A(f"- The price of the swap: **{dn:+.0f} emitters never intercepted** "
              f"for **{ratio:.1f}x the TWIR**.")
        A("")
    A("## Reading this as one strategy")
    A("")
    A("The pattern across tiers is the actual result: **as emitter density rises, the")
    A("optimal policy shifts from exploitation to coverage.** That is a scan strategy")
    A("that adapts to the environment rather than a fixed schedule, which is what the")
    A("problem statement asks for -- and the switch can be driven online, since the")
    A("count of distinct emitters found in the first second is an observable the")
    A("receiver already has.")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPORTS / "tier_policy"))
    args = ap.parse_args()

    data: dict = {"tiers": {}}
    for tier in TIERS:
        if not (REPORTS / f"metrics_{tier}.json").is_file():
            continue
        agents = _aggregate(tier)
        front = _pareto(agents)
        usable = {a: v for a, v in agents.items() if np.isfinite(v["twir"])}
        data["tiers"][tier] = {
            "agents": agents,
            "pareto": front,
            "n_seeds": max(v["n_seeds"] for v in agents.values()),
            # Tie-break on TWIR. Ten schedulers reach zero never-intercepted on
            # EASY; recommending whichever sorts first would name `sequential`
            # when `epsilon_greedy` matches its zero at 3.7x the TWIR and
            # strictly dominates it. A tie on the primary axis is decided by the
            # secondary one, not by dictionary order.
            "best_coverage": _tied_best(agents, usable)[0],
            "tied_group": _tied_best(agents, usable)[1],
            "inconclusive": _tied_best(agents, usable)[2],
            "stats": {},
            "best_twir": max(usable, key=lambda a: usable[a]["twir"]),
        }
        # Is the recommendation actually distinguishable from the next option,
        # or is a 61-vs-62 gap one seed of luck? Paired, so the same worlds are
        # shown to both and only the policy differs.
        d = data["tiers"][tier]
        cov, tw = d["best_coverage"], d["best_twir"]
        ranked = sorted(usable, key=lambda a: usable[a]["never_intercepted"])
        if len(ranked) > 1:
            second = ranked[1] if ranked[0] == cov else ranked[0]
            c = _paired(agents, cov, second, "per_seed_never", lower_is_better=True)
            c["names"] = f"`{cov}` vs `{second}`"
            d["stats"]["runner_up"] = c
        if cov != tw:
            c = _paired(agents, cov, tw, "per_seed_never", lower_is_better=True)
            c["names"] = f"`{cov}` vs `{tw}`"
            d["stats"]["cov_vs_twir"] = c
    if not data["tiers"]:
        print("no reports/metrics_*.json -- run `make benchmark` first", file=sys.stderr)
        return 1

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(data, indent=1, default=float),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(render(data), encoding="utf-8")
    print(render(data))
    print(f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
