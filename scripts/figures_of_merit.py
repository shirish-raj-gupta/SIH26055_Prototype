#!/usr/bin/env python
"""Emit the problem statement's figures of merit as one auditable table.

SIH 26055 names seven figures of merit and asks for a scan strategy built on
them. Every one has been implemented since early on -- ``metrics.py`` even
numbers them ``**Metric 1**`` through ``**Metric 10**`` in its docstrings -- but
they were scattered across three report files, a training history and a
function nobody calls from the benchmark. "Percentage of correct predictions"
was the worst case: computed on every training run, persisted nowhere a reader
could find it.

This writes ``reports/figures_of_merit.{md,json}``: each figure of merit, its
value on each tier, and the exact function that produced it. Nothing here is
recomputed from scratch except Pd and Pfa, which need an episode; the rest is
read from the shipped artifacts so the table cannot drift from the benchmark.

Usage::

    python scripts/figures_of_merit.py            # reads shipped reports
    python scripts/figures_of_merit.py --tier medium
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORTS = REPO_ROOT / "reports"
CKPT = REPO_ROOT / "runs" / "checkpoints"
TIERS = ("easy", "medium", "hard")

#: The scheduler the figures are quoted for. Whittle is the headline policy;
#: quoting Pd/Pfa for a different one would not match the leaderboard.
HEADLINE_AGENT = "whittle"


def _median(rows: list[dict], agent: str, key: str) -> float:
    """Median of ``key`` across seeds for one agent, or nan."""
    vals = [r[key] for r in rows if r.get("agent") == agent and r.get(key) is not None]
    vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def _detection_figures(tier: str) -> dict[str, float]:
    """Pd and Pfa, which need an actual episode rather than a summary row."""
    from smartscan.agents import build_agent
    from smartscan.analysis.metrics import empirical_pd, empirical_pfa, sensitivity_db
    from smartscan.config import load_config
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.runner import run_episode

    cfg = load_config(f"{tier}.yaml")
    seed = cfg.run.seed
    scenario = generate_scenario(seed, config=cfg)
    episode = build_episode(scenario)
    result = run_episode(
        cfg, seed, build_agent(HEADLINE_AGENT, cfg, seed, scenario),
        scenario=scenario, episode=episode,
    )

    pd_curve = empirical_pd(episode, result.visit_mask, result.true_hit_mask)
    # A single Pd number is meaningless without an SNR: quote the high-SNR
    # asymptote, over bins with enough trials to mean anything.
    n = np.asarray(pd_curve["n"], dtype=float)
    pd_vals = np.asarray(pd_curve["pd"], dtype=float)
    solid = n >= 30
    pd_high = float(np.nanmax(pd_vals[solid])) if solid.any() else float("nan")

    pfa = empirical_pfa(episode, result.visit_mask, result.hit_mask)
    sens = sensitivity_db(cfg)
    return {
        "pd_high_snr": pd_high,
        "pd_n_bins_used": int(solid.sum()),
        "pfa": float(pfa["pfa"]),
        "pfa_lo": float(pfa["lo"]),
        "pfa_hi": float(pfa["hi"]),
        "pfa_trials": int(pfa["n_trials"]),
        "sensitivity_pulse_db": float(sens["pulse_single_db"]),
        "sensitivity_energy_db": float(sens["energy_db"]),
    }


def _reward_decomposition(tier: str, agents: tuple[str, ...]) -> dict[str, dict]:
    """Split figure of merit 5 into the terms that produced it.

    The total return goes negative on the hard tier and a bare negative number
    invites the wrong conclusion. It is not a broken scheduler and it is not
    the staleness penalty: it is ``w5_interferer_dwell``, which the hard config
    doubles to 2.0, charged on every dwell that lands on a decoy. Splitting the
    sum is the only way to say that with evidence rather than assertion.

    Implemented by swapping the accountant for a tallying subclass, since
    ``run_episode`` constructs its own and returns only the summed reward.
    """
    import smartscan.runner as runner
    from smartscan.agents import build_agent
    from smartscan.config import load_config
    from smartscan.env.rf_environment import build_episode, generate_scenario

    built: list = []

    class _Tally(runner.RewardAccountant):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            built.append(self)
            self.terms = dict.fromkeys(
                ("new", "reconfirm", "retune", "interferer", "staleness"), 0.0)
            self.counts = dict.fromkeys(("new", "reconfirm", "retune", "interferer"), 0)

        def step(self, detected_ids, retuned, interferer_dwell, max_staleness):
            c = self.cfg
            for eid in np.unique(detected_ids):
                eid = int(eid)
                if eid <= 0:
                    continue
                if eid not in self.seen:
                    self.terms["new"] += c.w1_threat_intercept * self.threat.get(eid, 0.5) + c.w2_novelty
                    self.counts["new"] += 1
                elif self.reconfirms.get(eid, 0) < c.reconfirm_cap_per_emitter:
                    self.terms["reconfirm"] += c.w3_reconfirm
                    self.counts["reconfirm"] += 1
            if retuned:
                self.terms["retune"] -= c.w4_retune
                self.counts["retune"] += 1
            if interferer_dwell:
                self.terms["interferer"] -= c.w5_interferer_dwell
                self.counts["interferer"] += 1
            st = c.w6_staleness * (max_staleness / max(self.n_slots, 1))
            if c.normalise_by_episode:
                st /= max(self.n_slots, 1)
            self.terms["staleness"] -= st
            return super().step(detected_ids, retuned, interferer_dwell, max_staleness)

    cfg = load_config(f"{tier}.yaml")
    seed = cfg.run.seed
    scenario = generate_scenario(seed, config=cfg)
    episode = build_episode(scenario)
    out: dict[str, dict] = {
        "_n_interferers": sum(1 for t in episode.truth if t.is_interferer),
        "_n_emitters": len(episode.truth),
        "_w5": float(cfg.reward.w5_interferer_dwell),
        "_seed": int(seed),
    }
    for agent in agents:
        original = runner.RewardAccountant
        runner.RewardAccountant = _Tally
        try:
            built.clear()
            result = runner.run_episode(
                cfg, seed, build_agent(agent, cfg, seed, scenario),
                scenario=scenario, episode=episode,
            )
        finally:
            runner.RewardAccountant = original
        tally = built[-1]
        out[agent] = {
            "total": float(np.sum(result.rewards)),
            "terms": dict(tally.terms),
            "counts": dict(tally.counts),
        }
    return out


def _predictor_figures(tier: str) -> dict[str, float]:
    """Metric 8, straight from the shipped training history."""
    path = CKPT / f"predictor_{tier}_history.json"
    if not path.exists():
        return {}
    hist = json.loads(path.read_text(encoding="utf-8"))
    s = hist.get("scores_vs_truth") or {}
    base = float(s.get("positive_rate", float("nan")))
    acc = float(s.get("accuracy", float("nan")))
    return {
        "accuracy": acc,
        # The number that stops the accuracy being read as a triumph.
        "accuracy_of_always_idle": 1.0 - base if np.isfinite(base) else float("nan"),
        "precision": float(s.get("precision", float("nan"))),
        "recall": float(s.get("recall", float("nan"))),
        "f1": float(s.get("f1", float("nan"))),
        "brier": float(s.get("brier", float("nan"))),
        "auc": float(s.get("auc", float("nan"))),
        "average_precision": float(s.get("average_precision", float("nan"))),
        "positive_rate": base,
        "ap_lift_over_base_rate": float(hist.get("ap_lift_over_base_rate", float("nan"))),
        "threshold": float(s.get("threshold", 0.5)),
    }


def collect() -> dict:
    """Gather every figure of merit from the shipped artifacts."""
    out: dict = {"headline_agent": HEADLINE_AGENT, "tiers": {}}

    scan_path = REPORTS / "scan_on_scan.json"
    scan = json.loads(scan_path.read_text(encoding="utf-8")) if scan_path.exists() else {}
    summary = scan.get("summary", scan)
    out["scan_estimation"] = {
        "median_arrival_time_error_s": summary.get("median_arrival_time_error_s"),
        "median_rel_error_lomb_scargle": summary.get("median_rel_error_lomb_scargle"),
        "median_rel_error_sdif": summary.get("median_rel_error_sdif"),
        "n_emitters": summary.get("n_emitters"),
        "n_resolved": summary.get("n_resolved"),
    }

    for tier in TIERS:
        mpath = REPORTS / f"metrics_{tier}.json"
        if not mpath.exists():
            continue
        rows = json.loads(mpath.read_text(encoding="utf-8"))["rows"]
        a = HEADLINE_AGENT
        tier_out = {
            "intercept_rate_per_s": _median(rows, a, "intercept_rate_per_s"),
            "reward_total": _median(rows, a, "reward_total"),
            "reward_discounted": _median(rows, a, "reward_discounted"),
            "ttfi_median_s": _median(rows, a, "ttfi_median_s"),
            "ttfi_p90_s": _median(rows, a, "ttfi_p90_s"),
            "twir_rate": _median(rows, a, "twir_rate"),
            "interception_ratio_raw": _median(rows, a, "interception_ratio_raw"),
            "coverage": _median(rows, a, "coverage"),
            "coverage_entropy": _median(rows, a, "coverage_entropy"),
            "n_seeds": sum(1 for r in rows if r.get("agent") == a),
        }
        tier_out.update(_detection_figures(tier))
        tier_out["predictor"] = _predictor_figures(tier)
        tier_out["reward_decomposition"] = _reward_decomposition(
            tier, (HEADLINE_AGENT, "sequential"))
        out["tiers"][tier] = tier_out
    return out


def _fmt(v: object, spec: str = ".4g") -> str:
    """Format a number for the table, or an em dash when it is missing."""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if not np.isfinite(float(v)):
            return "∞" if float(v) > 0 else "—"
        if spec.endswith("d"):  # integer specs reject floats
            return f"{int(round(float(v))):{spec}}"
        return f"{float(v):{spec}}"
    return str(v)


def render(data: dict) -> str:
    """Render the compliance table as markdown."""
    tiers = [t for t in TIERS if t in data["tiers"]]
    head = " | ".join(f"`{t}`" for t in tiers)
    sep = " | ".join("---" for _ in tiers)
    L: list[str] = []
    A = L.append

    A("# Figures of merit — SIH 26055")
    A("")
    A("Every figure of merit named in the problem statement, with the function that")
    A(f"computes it. Values are medians across seeds for **`{data['headline_agent']}`**,")
    A("the headline scheduler, so they match `reports/leaderboard.md` directly.")
    A("")
    A("_Generated by `scripts/figures_of_merit.py` from the shipped artifacts._")
    A("")
    A(f"| # | Figure of merit (PS wording) | {head} | Computed by |")
    A(f"|---|---|{sep}|---|")

    def row(num: str, label: str, key: str, fn: str, spec: str = ".4g", sub: str | None = None) -> None:
        vals = " | ".join(
            _fmt((data["tiers"][t]["predictor"] if sub else data["tiers"][t]).get(key), spec)
            for t in tiers
        )
        A(f"| {num} | {label} | {vals} | `{fn}` |")

    row("1", "Probability of detection (high-SNR)", "pd_high_snr", "empirical_pd", ".3f")
    row("2", "Probability of false alarm", "pfa", "empirical_pfa", ".2e")
    row("3", "Sensitivity — single pulse (dB SNR)", "sensitivity_pulse_db", "sensitivity_db", ".1f")
    row("3", "Sensitivity — energy detect (dB SNR)", "sensitivity_energy_db", "sensitivity_db", ".1f")
    row("4", "Avg intercept rate (per s)", "intercept_rate_per_s", "average_intercept_rate", ".1f")
    row("5", "Avg reward / cost function", "reward_total", "average_reward", ".1f")
    row("6", "**% of correct predictions**", "accuracy", "prediction_scores", ".2%", sub="predictor")
    A("| 6 | ↳ _same score for an always-idle model_ | "
      + " | ".join(_fmt(data['tiers'][t]['predictor'].get('accuracy_of_always_idle'), '.2%') for t in tiers)
      + " | base rate |")
    row("6", "↳ average precision (the honest one)", "average_precision", "average_precision", ".3f", sub="predictor")
    row("6", "↳ AP lift over base rate", "ap_lift_over_base_rate", "—", ".2f", sub="predictor")
    row("6", "↳ ROC AUC", "auc", "roc_auc", ".3f", sub="predictor")
    row("6", "↳ Brier score", "brier", "prediction_scores", ".4f", sub="predictor")

    # Repeating one number across three tier columns would imply it was measured
    # per tier. It comes from configs/scan_on_scan.yaml, which is a separate
    # scenario, so it gets one cell and a footnote.
    err = data["scan_estimation"].get("median_arrival_time_error_s")
    span = " | ".join(["—"] * (len(tiers) - 1))
    A(f"| 7 | Avg intercept time error (s) † | {_fmt(err, '.3f')} | {span} | "
      f"`average_intercept_time_error` |")
    A("")
    A("† Measured on `configs/scan_on_scan.yaml`, the dedicated scan-on-scan")
    A("scenario, not per tier — the tier configs carry no rotating-emitter ground")
    A("truth to predict arrival times against.")
    A("")
    A("### Supporting figures")
    A("")
    A(f"| Quantity | {head} | Computed by |")
    A(f"|---|{sep}|---|")
    for label, key, fn, spec in (
        ("Time to first intercept, median (s)", "ttfi_median_s", "time_to_first_intercept", ".4f"),
        ("Time to first intercept, p90 (s)", "ttfi_p90_s", "time_to_first_intercept", ".4f"),
        ("Interception ratio, threat-weighted", "twir_rate", "interception_ratio", ".5f"),
        ("Interception ratio, raw", "interception_ratio_raw", "interception_ratio", ".5f"),
        ("Band coverage", "coverage", "spectrum_coverage", ".3f"),
        ("Coverage entropy", "coverage_entropy", "coverage_entropy", ".4f"),
        ("Seeds", "n_seeds", "—", "d"),
    ):
        vals = " | ".join(_fmt(data["tiers"][t].get(key), spec) for t in tiers)
        A(f"| {label} | {vals} | `{fn}` |")

    se = data["scan_estimation"]
    A("")
    A("### Scan-period estimation (PS: *\"approaches to intercept a periodic scan receiver optimally\"*)")
    A("")
    A(f"- Emitters resolved: **{se.get('n_resolved')} / {se.get('n_emitters')}**")
    A(f"- Median relative period error, Lomb-Scargle: **{_fmt(se.get('median_rel_error_lomb_scargle'), '.4%')}**")
    A(f"- Median relative period error, SDIF: **{_fmt(se.get('median_rel_error_sdif'), '.2%')}**")
    A(f"- Median arrival-time error: **{_fmt(se.get('median_arrival_time_error_s'), '.3f')} s**")
    A("")
    A("### Reading note on figure of merit 5 — why the return is negative on `hard`")
    A("")
    A("A negative total return does not mean the scheduler failed. Splitting the sum")
    A("into the terms that produced it says what it does mean:")
    A("")
    for t in tiers:
        dec = data["tiers"][t].get("reward_decomposition") or {}
        if not dec:
            continue
        ha = dec.get(data["headline_agent"], {})
        terms, counts = ha.get("terms", {}), ha.get("counts", {})
        if not terms:
            continue
        A(f"**`{t}`** — {dec.get('_n_interferers')} of {dec.get('_n_emitters')} emitters are "
          f"interferers, `w5_interferer_dwell` = {_fmt(dec.get('_w5'), '.1f')}, seed {dec.get('_seed')}:")
        A("")
        A("| term | contribution | events |")
        A("|---|---|---|")
        for key in sorted(terms, key=lambda k: -abs(terms[k])):
            cnt = counts.get(key)
            A(f"| {key} | {terms[key]:+.1f} | {cnt if cnt is not None else '—'} |")
        A(f"| **total** | **{ha.get('total', float('nan')):+.1f}** | |")
        A("")

    # The comparison is the point: everyone is negative here, and the open-loop
    # sweep is the one that pays most for it.
    hard = (data["tiers"].get("hard") or {}).get("reward_decomposition") or {}
    ha, base = hard.get(data["headline_agent"]), hard.get("sequential")
    if ha and base:
        n_a = ha["counts"].get("interferer", 0)
        n_b = base["counts"].get("interferer", 0)
        if n_b:
            A(f"On `hard` the penalty is charged {n_a} times against "
              f"`{data['headline_agent']}` and {n_b} times against the `sequential` "
              f"sweep — **{(1 - n_a / n_b):.0%} fewer dwells wasted on decoys** "
              f"({base['total']:+.1f} against {ha['total']:+.1f} total return).")
            A("")
    A("This is the problem statement's own complaint about open-loop scanning —")
    A("*\"may lose time to nonthreatening emitters by not giving time to new or")
    A("threatening ones\"* — measured directly. The `hard` config doubles")
    A("`w5_interferer_dwell` to 2.0 precisely so that decoys cost what they should,")
    A("which makes every policy's return negative and the *differences* between them")
    A("the thing to read. Staleness, by contrast, contributes under a point: it is")
    A("not what drives the sign.")
    A("")
    A("### Reading note on figure of merit 6")
    A("")
    A("The problem statement asks for *\"percentage of correct predictions\"*, and it is")
    A("reported above. It should not be read as the headline: channel occupancy is")
    A("roughly 8 % positive, so a model that predicts \"idle\" everywhere scores the")
    A("*always-idle* row — within a couple of points of the real model — while being")
    A("useless. Average precision collapses to the base rate for that model instead,")
    A("which is why the scheduler is selected on AP and why the lift over base rate is")
    A("quoted beside it. The scheduler ranks channels by predicted occupancy and never")
    A("applies a threshold, so accuracy, precision and recall describe an operating")
    A("point it does not use.")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPORTS / "figures_of_merit"),
                    help="Output stem; .md and .json are written.")
    args = ap.parse_args()

    data = collect()
    if not data["tiers"]:
        print("no reports/metrics_*.json found -- run `make benchmark` first", file=sys.stderr)
        return 1

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(data, indent=1, default=float), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(data), encoding="utf-8")
    print(f"wrote {stem.with_suffix('.md')}")
    print(f"wrote {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
