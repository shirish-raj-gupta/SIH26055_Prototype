#!/usr/bin/env python
"""Measure what revisiting a stale channel is actually worth.

Every value-based scheduler here scores a channel as::

    value = P(occupied) + coverage_weight * (time_since_visit / n_slots)

The staleness term is a straight line with a hand-set slope, and
``coverage_weight`` has been 1.0 since it was written. Nothing in the
repository establishes that the value of revisiting grows linearly with the
gap, and the whole HARD result turns on how coverage trades against
prediction -- so it is worth knowing the real shape rather than assuming one.

``observations_combined.csv`` has 8.26 M dwells replayed across every
scheduler and tier, with the per-channel detections each dwell produced. That
is enough to measure the quantity directly: for every (channel, revisit gap)
pair actually flown, did the look land a genuine detection?

Method. ``slots_elapsed`` in the corpus is the gap between consecutive
*dwells*, not the gap since a given channel was last seen, so the staleness
that matters has to be reconstructed. For each (tier, episode, scheduler) the
dwell sequence is expanded to one row per observed channel, sorted by
(channel, time), and differenced within channel -- giving the exact revisit
gap preceding every look, and whether that look hit.

What comes out is ``P(detection | revisit gap)`` per tier: the empirical
version of the term the schedulers approximate with a line.

Usage::

    python scripts/staleness_value.py --csv observations_combined.csv
    python scripts/staleness_value.py --max-rows 2000000     # quick look
"""

from __future__ import annotations

import argparse
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

#: Channels observed per dwell. The corpus carries snr_est_db_0..3, so 4.
K = 4

#: Schedulers whose revisit gap does NOT depend on what they have observed.
#: This is the whole point of the split: for everyone else the gap is CHOSEN
#: using the belief, so a long gap means the policy decided the channel was
#: dead, and P(hit | long gap) is low partly because of that decision rather
#: than because of staleness. That is confounding by indication, and it would
#: make the measurement say whatever the policies already believed.
#:
#: `sequential` sweeps a fixed saw-tooth and `random` draws uniformly; neither
#: reads a hit. Their gaps are exogenous, so their curve identifies the causal
#: shape. `coprime_sweep` is excluded despite being open-loop in spirit --
#: avoid_detected_periods is on, so it nudges its step using estimated periods.
OPEN_LOOP = ("sequential", "random")

#: The instrument the conclusion rests on. Both members of OPEN_LOOP have
#: exogenous gaps, but only `random` has SUPPORT across the gap range: a
#: saw-tooth populates gap 1 and its own revisit period and almost nothing
#: else, so on HARD its 512-767 bin holds 810 looks out of 644,544 -- 0.13% of
#: its data, in a range the pattern should never reach, and plainly an edge
#: artifact. Pooling the two let those 810 looks drag the combined curve to
#: p=1.1e-12 and invent a staleness effect. Uniform draws spread across every
#: bin, so `random` alone identifies the shape without Simpson's paradox.
PRIMARY = "random"

#: Revisit-gap bin edges in slots (1 slot = 1 ms). Log-ish, because the
#: interesting structure is at short gaps and the tail is long.
EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768,
                  1024, 2048, 4096, 10_001], dtype=np.int64)


def _expand(window_lo: np.ndarray, t: np.ndarray, mask: np.ndarray):
    """One row per observed channel: (channel, slot, hit)."""
    ch = (window_lo[:, None] + np.arange(K, dtype=np.int64)).ravel()
    tt = np.repeat(t, K)
    # hit_mask is packbits little-endian over the K channels of the window.
    hit = ((mask[:, None] >> np.arange(K, dtype=np.int64)) & 1).ravel()
    return ch, tt, hit


def _accumulate(group, looks, hits):
    """Fold one (tier, episode, agent) group into the histograms."""
    w = group["window_lo"].to_numpy(np.int64)
    t = group["t"].to_numpy(np.int64)
    m = group["true_hit_mask"].to_numpy(np.int64)
    if w.size < 2:
        return
    ch, tt, hit = _expand(w, t, m)

    # Sort by channel then time so a difference within a channel is the gap
    # since that channel was last observed -- which is the staleness the
    # scheduler's term is standing in for, and is not `slots_elapsed`.
    order = np.lexsort((tt, ch))
    ch, tt, hit = ch[order], tt[order], hit[order]

    same = ch[1:] == ch[:-1]
    gap = tt[1:] - tt[:-1]
    # Pair each gap with the look that ENDED it.
    valid = same & (gap > 0)
    if not np.any(valid):
        return
    idx = np.digitize(gap[valid], EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1)
    np.add.at(looks, idx[ok], 1)
    np.add.at(hits, idx[ok], hit[1:][valid][ok])


def analyse(csv: Path, max_rows: int | None) -> dict:
    """Stream the corpus and build P(hit | revisit gap) per tier."""
    import pandas as pd

    nbin = len(EDGES) - 1
    looks: dict[str, np.ndarray] = {}
    hits: dict[str, np.ndarray] = {}
    carry = None
    seen = 0

    cols = ["difficulty", "episode_id", "agent", "t", "window_lo", "true_hit_mask"]
    reader = pd.read_csv(csv, usecols=cols, chunksize=1_000_000)
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        key = (chunk["difficulty"] + "|" + chunk["episode_id"] + "|" + chunk["agent"])
        # Hold back the final group: it may continue into the next chunk, and
        # splitting a group would invent a revisit gap at the seam.
        last = key.iloc[-1]
        tail = key == last
        carry = chunk[tail].copy()
        body = chunk[~tail]
        for (tier, _ep, ag), g in body.groupby(
                ["difficulty", "episode_id", "agent"], sort=False):
            for key in (f"{tier}|{ag}", f"{tier}|ALL"):
                looks.setdefault(key, np.zeros(nbin, np.int64))
                hits.setdefault(key, np.zeros(nbin, np.int64))
                _accumulate(g, looks[key], hits[key])
        seen += len(body)
        print(f"  {seen:,} rows", flush=True)
        if max_rows and seen >= max_rows:
            carry = None
            break
    if carry is not None and len(carry):
        for (tier, _ep, ag), g in carry.groupby(
                ["difficulty", "episode_id", "agent"], sort=False):
            for key in (f"{tier}|{ag}", f"{tier}|ALL"):
                looks.setdefault(key, np.zeros(nbin, np.int64))
                hits.setdefault(key, np.zeros(nbin, np.int64))
                _accumulate(g, looks[key], hits[key])

    out: dict = {"rows_used": seen, "k": K, "edges": EDGES.tolist(),
                 "open_loop": list(OPEN_LOOP), "cells": {}}
    # Pool the open-loop schedulers per tier: that pooled curve is the causal
    # estimate, everything else is the same quantity contaminated by choice.
    for tier in {k.split("|")[0] for k in looks}:
        nb = len(EDGES) - 1
        lo, hi = np.zeros(nb, np.int64), np.zeros(nb, np.int64)
        for a in OPEN_LOOP:
            k = f"{tier}|{a}"
            if k in looks:
                lo += looks[k]
                hi += hits[k]
        looks[f"{tier}|OPEN_LOOP"] = lo
        hits[f"{tier}|OPEN_LOOP"] = hi
    for key in looks:
        p = np.divide(hits[key], np.maximum(looks[key], 1), dtype=float)
        out["cells"][key] = {"looks": looks[key].tolist(),
                             "hits": hits[key].tolist(), "p_hit": p.tolist()}
    return out


def _homogeneity(cell: dict, min_looks: int = 500) -> dict:
    """Chi-square test that P(hit) is the same in every revisit-gap bin."""
    from scipy import stats

    looks = np.asarray(cell["looks"])
    hits = np.asarray(cell["hits"])
    keep = looks >= min_looks
    if keep.sum() < 3:
        return {}
    lk, ht = looks[keep], hits[keep]
    chi2, p, dof, _ = stats.chi2_contingency(np.array([ht, lk - ht]))
    rates = ht / lk
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "bins": int(keep.sum()), "n": int(lk.sum()),
            "pooled": float(ht.sum() / lk.sum()),
            "min": float(rates.min()), "max": float(rates.max())}


def render(d: dict) -> str:
    L: list[str] = []
    A = L.append
    edges = d["edges"]
    A("# What a revisit is actually worth")
    A("")
    A(f"_Generated by `scripts/staleness_value.py` from {d['rows_used']:,} replayed dwells._")
    A("")
    A("**Provenance.** These are the published corpus episodes (ids like")
    A("`easy_20261558`), not the 30 paired seeds 20260902-20260931 that the")
    A("leaderboard, `ps_objectives.md` and `tier_policy.md` are built from -- the two")
    A("populations are all but disjoint. So this is an independent check on roughly a")
    A("hundred times more episodes, rather than a re-reading of the runs it is used to")
    A("interpret.")
    A("")
    A("Every value-based scheduler scores a channel as "
      "`P(occupied) + coverage_weight x staleness`,")
    A("with the staleness term a straight line and `coverage_weight` fixed at 1.0 since")
    A("it was written. This measures the shape that line approximates.")
    A("")
    A("**The revisit gap has to be exogenous for this to mean anything.** For a")
    A("belief-driven scheduler the gap is *chosen*: a long gap means the policy decided")
    A("the channel was dead, so a low hit rate at long gaps is partly that decision")
    A("rather than staleness. Only `sequential` and `random` pick their gaps without")
    A("reading a hit, so their pooled curve identifies the causal shape and the rest")
    A("are shown beside it to expose how large the selection effect is.")
    A("")
    A("## Is it flat?")
    A("")
    A("Chi-square for homogeneity of `P(hit)` across gap bins, per tier, for each")
    A("open-loop scheduler separately -- separately because pooling them is what")
    A("manufactured a spurious effect the first time this was run.")
    A("")
    A("| tier | scheduler | pooled P(hit) | bins | n | chi2 | p | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for tier in sorted({k.split("|")[0] for k in d["cells"]}):
        for ag in d["open_loop"]:
            cell = d["cells"].get(f"{tier}|{ag}")
            if not cell:
                continue
            h = _homogeneity(cell)
            if not h:
                continue
            flat = "flat" if h["p"] >= 0.05 else "**not flat**"
            star = " (primary)" if ag == PRIMARY else ""
            A(f"| `{tier}` | `{ag}`{star} | {h['pooled']:.4f} | {h['bins']} | "
              f"{h['n']:,} | {h['chi2']:.1f} | {h['p']:.3g} | {flat} |")
    A("")
    A(f"`{PRIMARY}` is the instrument the conclusion rests on. Both schedulers pick")
    A("gaps without reading a hit, but a saw-tooth only ever produces gap 1 and its")
    A("own revisit period, so its long-gap bins hold a few hundred looks out of")
    A("hundreds of thousands and are edge artifacts. Uniform draws populate every bin.")
    A("")
    tiers = sorted({k.split("|")[0] for k in d["cells"]})
    for tier in tiers:
        # The PRIMARY instrument, not the pooled curve. Showing the pooled one
        # here would contradict the test above, which is what ruled it out.
        key = f"{tier}|{PRIMARY}"
        if key not in d["cells"]:
            continue
        v = d["cells"][key]
        A(f"## `{tier}` — `{PRIMARY}` only (exogenous gaps, support in every bin)")
        A("")
        A("| revisit gap (slots) | looks | detections | P(detection) |")
        A("|---|---|---|---|")
        for i, p in enumerate(v["p_hit"]):
            n = v["looks"][i]
            if n < 200:
                continue
            A(f"| {edges[i]}-{edges[i + 1] - 1} | {n:,} | {v['hits'][i]:,} | {p:.4f} |")
        A("")
        others = sorted(k for k in d["cells"]
                        if k.startswith(tier + "|")
                        and k.split("|")[1] not in ("ALL", "OPEN_LOOP", *d["open_loop"]))
        if others:
            A("<details><summary>Belief-driven schedulers on the same tier "
              "(confounded)</summary>")
            A("")
            A("| scheduler | P(hit) at gap 2-3 | at 32-63 | at 512-767 |")
            A("|---|---|---|---|")
            def at(vv, lo_edge):
                i = edges.index(lo_edge)
                return f"{vv['p_hit'][i]:.4f}" if vv["looks"][i] >= 200 else "-"
            for k in others:
                A(f"| `{k.split('|')[1]}` | {at(d['cells'][k], 2)} | "
                  f"{at(d['cells'][k], 32)} | {at(d['cells'][k], 512)} |")
            A("")
            A("</details>")
            A("")
    return chr(10).join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(REPO_ROOT / "observations_combined.csv"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default=str(REPORTS / "staleness_value"))
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.is_file():
        raise SystemExit(f"no corpus at {csv}")
    data = analyse(csv, args.max_rows)
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(data), encoding="utf-8")
    print(render(data))
    print(f"wrote {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
