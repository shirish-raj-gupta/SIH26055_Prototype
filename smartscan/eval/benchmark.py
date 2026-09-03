"""Benchmark harness: {schedulers} x {tiers} x {seeds}, with honest statistics.

Every headline number is a **paired** comparison against
``eval.baseline_agent`` on identical scenarios, with a cluster bootstrap over
seeds and a Wilcoxon signed-rank test corrected for multiple comparisons by
Holm-Bonferroni.

Pairing is what makes 30 seeds enough. Scenario-to-scenario variance dwarfs the
difference between policies, so an unpaired comparison would need hundreds of
seeds to resolve a 25 % effect. Because the environment draws its detection
realisation from the scenario seed (common random numbers, see
:mod:`smartscan.hal.simulated`), two schedulers on the same seed face not only
the same world but the same luck.

Effect sizes are reported alongside p-values, because a p-value on 30 seeds says
only that a difference exists, not that it matters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from smartscan.agents import build_agent
from smartscan.analysis.metrics import evaluate_episode, paired_bootstrap_delta
from smartscan.config import Config, load_config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.runner import run_episode

__all__ = [
    "BenchmarkResult",
    "Comparison",
    "holm_bonferroni",
    "leaderboard_latex",
    "leaderboard_markdown",
    "run_benchmark",
    "run_grid",
    "tidy_frame",
    "write_results_parquet",
]

#: Metrics where a *smaller* value is better. Everything else is
#: higher-is-better, and the sign of the reported improvement is flipped.
MIN_PAIRED_SEEDS = 10
#: Below this many finite paired seeds a comparison is withheld rather than
#: reported: censored metrics can otherwise be 'significant' on a handful of
#: lucky runs. See run_benchmark.

LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "ttfi_median_s", "ttfi_p90_s", "ttfi_mean_s", "ttfi_hard_median_s", "ttfi_hard_p90_s",
        "staleness_max_s", "staleness_mean_s", "revisit_p95_s", "waste_fraction",
        "popup_latency_s", "fa_burden", "n_never_intercepted", "wall_time_s",
        # NOTE: popup_detect_rate is deliberately absent -- higher is better.
    }
)


@dataclass(frozen=True)
class Comparison:
    """Paired comparison of one scheduler against the baseline on one metric.

    Attributes:
        agent: Candidate scheduler key.
        metric: Metric name.
        baseline_median: Baseline median across seeds.
        agent_median: Candidate median across seeds.
        improvement: Fractional improvement, signed so positive is better.
        ci_lo: Lower bound of the 95 % paired bootstrap CI on the improvement.
        ci_hi: Upper bound.
        p_value: Two-sided Wilcoxon signed-rank p-value.
        p_holm: Holm-Bonferroni corrected p-value.
        effect_size: Matched-pairs rank-biserial correlation in ``[-1, 1]``.
        n_seeds: Number of paired seeds.
    """

    agent: str
    metric: str
    baseline_median: float
    agent_median: float
    improvement: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_holm: float
    effect_size: float
    n_seeds: int

    @property
    def significant(self) -> bool:
        """Whether the corrected p-value clears 0.05."""
        return bool(np.isfinite(self.p_holm) and self.p_holm < 0.05)


@dataclass
class BenchmarkResult:
    """Full benchmark output.

    Attributes:
        rows: One dict of metrics per (agent, seed).
        comparisons: Paired comparisons against the baseline.
        config_hash: Hash of the resolved configuration.
        tier: Tier label.
        baseline: Baseline scheduler key.
        withheld: ``(agent, metric, n_finite, n_seeds)`` for comparisons not
            run because too few seeds had finite values on both sides.
    """

    rows: list[dict[str, Any]]
    comparisons: list[Comparison]
    config_hash: str
    tier: str
    baseline: str
    withheld: list[tuple[str, str, int, int]] | None = None

    def per_agent(self, metric: str) -> dict[str, np.ndarray]:
        """Return per-seed values of ``metric``, keyed by agent."""
        out: dict[str, list[float]] = {}
        for r in self.rows:
            out.setdefault(r["agent"], []).append(r.get(metric, float("nan")))
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def to_json(self, path: str | Path) -> Path:
        """Write the result to a JSON file.

        Args:
            path: Destination path.

        Returns:
            The path written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tier": self.tier,
            "baseline": self.baseline,
            "config_hash": self.config_hash,
            "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.rows],
            "comparisons": [asdict(c) for c in self.comparisons],
        }
        p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        return p


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Uniformly more powerful than plain Bonferroni at the same family-wise error
    rate, and the right default when comparing many schedulers on many metrics
    (we run dozens of tests, so uncorrected p-values would manufacture
    significance by volume).

    Args:
        p_values: Raw p-values.

    Returns:
        Corrected p-values in the input order, clipped to ``[0, 1]``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation -- the Wilcoxon effect size."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total) if total else 0.0


def run_benchmark(
    config: Config,
    agents: Iterable[str] | None = None,
    seeds: Sequence[int] | None = None,
    metrics: Iterable[str] | None = None,
    n_jobs: int = 1,
    progress: bool = True,
) -> BenchmarkResult:
    """Run every scheduler over every seed and compute paired statistics.

    Args:
        config: Resolved configuration.
        agents: Scheduler keys; defaults to ``config.eval.agents``.
        seeds: Scenario seeds; defaults to ``seed .. seed + n_seeds - 1``.
        metrics: Metrics to compare; defaults to ``config.eval.metrics``.
        n_jobs: Parallel workers for joblib. ``1`` keeps everything in-process,
            which is what the reproducibility test uses.
        progress: Print a one-line progress note per seed.

    Returns:
        The populated :class:`BenchmarkResult`.
    """
    agent_keys = list(agents or config.eval.agents)
    seed_list = list(seeds or range(config.run.seed, config.run.seed + config.run.n_seeds))
    metric_keys = list(metrics or config.eval.metrics)
    baseline = config.eval.baseline_agent

    def one_seed(seed: int) -> list[dict[str, Any]]:
        # Build the scenario and tensors ONCE per seed and share them across
        # schedulers: that is what makes the comparison paired.
        scenario = generate_scenario(seed, config=config)
        episode = build_episode(scenario)
        out = []
        for key in agent_keys:
            agent = build_agent(key, config, seed, scenario)
            res = run_episode(
                config, seed, agent, scenario=scenario, episode=episode,
            )
            row = evaluate_episode(res, config)
            row.pop("_detail", None)
            row["agent"] = key
            # Carry the agent's own name, not just its registry key. A learned
            # scheduler with no checkpoint substitutes an analytic policy and
            # says so in its name -- "predictor (untrained -> ucb1 fallback)".
            # Keying the row on `key` alone throws that away, and the
            # leaderboard then reports UCB1's numbers under the name of a model
            # that was never trained.
            row["agent_name"] = agent.name
            row["fallback"] = "fallback" in agent.name or "->" in agent.name
            out.append(row)
        return out

    rows: list[dict[str, Any]] = []
    if n_jobs != 1:
        from joblib import Parallel, delayed

        for chunk in Parallel(n_jobs=n_jobs)(delayed(one_seed)(s) for s in seed_list):
            rows.extend(chunk)
    else:
        for i, seed in enumerate(seed_list):
            rows.extend(one_seed(seed))
            if progress:
                print(f"  seed {i + 1}/{len(seed_list)} done", flush=True)

    result = BenchmarkResult(rows, [], config.hash(), config.scenario.difficulty, baseline)

    # -- paired statistics ------------------------------------------------- #
    raw: list[tuple[str, str, dict[str, Any]]] = []
    dropped: list[tuple[str, str, int, int]] = []
    for metric in metric_keys:
        per = result.per_agent(metric)
        if baseline not in per:
            continue
        base = per[baseline]
        lower_better = metric in LOWER_IS_BETTER
        for key in agent_keys:
            if key == baseline or key not in per:
                continue
            cand = per[key]
            ok = np.isfinite(cand) & np.isfinite(base)

            # Dropping non-finite seeds is not neutral. ttfi_hard_median_s is
            # +inf exactly when an agent never intercepted a hard-class emitter,
            # so discarding those seeds throws away the agent's failures and
            # scores it only on the runs where it succeeded -- flattering the
            # worst performers most. On MEDIUM this left as few as 3 of 30 seeds
            # for some agents while every other metric kept all 30.
            #
            # There is no honest paired test on a censored sample this small, so
            # refuse rather than report one. MIN_PAIRED_SEEDS is deliberately
            # high: this project has already retracted two findings that came
            # from small samples, and a 3-seed comparison is not evidence.
            n_ok = int(ok.sum())
            if n_ok < MIN_PAIRED_SEEDS:
                dropped.append((key, metric, n_ok, len(cand)))
                continue
            a, b = cand[ok], base[ok]
            ci = paired_bootstrap_delta(
                a, b, relative=True, n_boot=config.eval.n_bootstrap,
                level=config.eval.ci, seed=config.run.seed,
            )
            sign = 1.0 if lower_better else -1.0
            diff = (b - a) * sign
            try:
                p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
            except ValueError:  # all differences zero
                p = 1.0
            raw.append(
                (key, metric, {
                    "baseline_median": float(np.median(b)),
                    "agent_median": float(np.median(a)),
                    "improvement": sign * ci.point,
                    "ci_lo": sign * (ci.lo if sign > 0 else ci.hi),
                    "ci_hi": sign * (ci.hi if sign > 0 else ci.lo),
                    "p_value": p,
                    "effect_size": _rank_biserial(diff),
                    "n_seeds": int(ok.sum()),
                })
            )

    corrected = holm_bonferroni([r[2]["p_value"] for r in raw])
    result.comparisons = [
        Comparison(agent=k, metric=m, p_holm=float(pc), **d)
        for (k, m, d), pc in zip(raw, corrected, strict=True)
    ]

    # A withheld comparison must not read as a passed one. Say what was not
    # tested and why, or the absence looks like the agent simply did not win.
    result.withheld = dropped
    if dropped and progress:
        print(
            f"  {len(dropped)} comparison(s) withheld: fewer than "
            f"{MIN_PAIRED_SEEDS} seeds where both agent and baseline were finite",
            flush=True,
        )
        for key, metric, n_ok, n_all in sorted(dropped, key=lambda d: d[2])[:8]:
            print(f"    {key:<15}{metric:<24}{n_ok}/{n_all} seeds", flush=True)
    return result


def leaderboard_markdown(result: BenchmarkResult, metrics: Sequence[str] | None = None) -> str:
    """Render the benchmark as a markdown leaderboard.

    Args:
        result: The benchmark output.
        metrics: Metrics to show; a compact default is used if omitted.

    Returns:
        A markdown string.
    """
    metrics = list(
        metrics
        or ["ttfi_hard_median_s", "ttfi_median_s", "twir_rate", "coverage",
            "staleness_max_s", "coverage_entropy", "reward_total"]
    )
    agents = sorted({r["agent"] for r in result.rows}, key=lambda a: (a != result.baseline, a))

    lines = [
        f"### Leaderboard - tier `{result.tier}`, baseline `{result.baseline}`",
        f"_config hash `{result.config_hash[:12]}`, "
        f"{len({r['seed'] for r in result.rows})} seeds, medians across seeds_",
        "",
        "| agent | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(metrics) + 1),
    ]
    for a in agents:
        vals = []
        for m in metrics:
            v = [r.get(m, np.nan) for r in result.rows if r["agent"] == a]
            vals.append(f"{np.nanmedian(v):.4g}" if v else "-")
        mark = " **(baseline)**" if a == result.baseline else ""
        # A row whose agent substituted an analytic policy for missing weights
        # must not read as a trained model's result.
        sub = next((r.get("agent_name", "") for r in result.rows
                    if r["agent"] == a and r.get("fallback")), "")
        if sub:
            mark += f" [NOT TRAINED - ran as `{sub}`]"
        lines.append(f"| `{a}`{mark} | " + " | ".join(vals) + " |")

    lines += ["", "### Paired improvement vs baseline (95 % bootstrap CI, Holm-corrected)", "",
              "| agent | metric | baseline | agent | improvement | 95 % CI | p (Holm) | effect | sig |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in result.comparisons:
        lines.append(
            f"| `{c.agent}` | {c.metric} | {c.baseline_median:.4g} | {c.agent_median:.4g} "
            f"| {100 * c.improvement:+.1f}% | [{100 * c.ci_lo:+.1f}%, {100 * c.ci_hi:+.1f}%] "
            f"| {c.p_holm:.3g} | {c.effect_size:+.2f} | {'yes' if c.significant else 'no'} |"
        )

    fell_back = sorted({r["agent"] for r in result.rows if r.get("fallback")})
    if fell_back:
        names = ", ".join(f"`{a}`" for a in fell_back)
        lines += [
            "",
            f"> **{len(fell_back)} agent(s) had no trained weights and ran as a "
            f"substitute policy: {names}.** Those rows measure the substitute, "
            "not the model named. Train them, or read them as duplicates of the "
            "policy they fell back to.",
        ]
    return "\n".join(lines)


def tidy_frame(result: BenchmarkResult) -> Any:
    """Return the per-(agent, seed, metric) results in **tidy** long form.

    One row per observation rather than one per run, because every downstream
    consumer -- a group-by, a facet plot, a statistical test -- wants it that
    way, and a wide table forces each of them to melt it first.

    Args:
        result: The benchmark output.

    Returns:
        A ``pandas.DataFrame`` with columns ``tier, agent, seed, metric, value,
        is_baseline, config_hash``.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'pandas is required for tidy output; install `pip install "smartscan[viz]"`.'
        ) from exc

    records = []
    for row in result.rows:
        for key, value in row.items():
            if key in {"agent", "seed"} or isinstance(value, (str, dict, list)):
                continue
            records.append(
                {
                    "tier": result.tier,
                    "agent": row["agent"],
                    "seed": int(row["seed"]),
                    "metric": key,
                    "value": float(value),
                    "is_baseline": row["agent"] == result.baseline,
                    "config_hash": result.config_hash,
                }
            )
    return pd.DataFrame.from_records(records)


def write_results_parquet(
    results: BenchmarkResult | Sequence[BenchmarkResult],
    path: str | Path = "reports/results.parquet",
) -> Path:
    """Write the tidy results table, appending across tiers.

    Args:
        results: One benchmark result, or several to concatenate.
        path: Destination parquet path.

    Returns:
        The path written.
    """
    import pandas as pd

    batch = [results] if isinstance(results, BenchmarkResult) else list(results)
    frame = pd.concat([tidy_frame(r) for r in batch], ignore_index=True)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p, index=False, compression="zstd")
    return p


def _latex_escape(text: str) -> str:
    """Escape the LaTeX specials that appear in agent and metric names."""
    for char, repl in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(char, repl)
    return text


def leaderboard_latex(
    result: BenchmarkResult,
    metrics: Sequence[str] | None = None,
    caption: str | None = None,
    label: str = "tab:leaderboard",
) -> str:
    """Render the paired comparison as a LaTeX table for the report.

    Significance is marked with a dagger rather than a colour, so the table
    survives being printed in black and white -- and the confidence interval is
    shown beside every point estimate, because a bare percentage invites the
    reader to over-read it.

    Args:
        result: The benchmark output.
        metrics: Metrics to include; a compact default is used if omitted.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        A ``table`` environment as a string.
    """
    metrics = list(metrics or ["ttfi_hard_median_s", "twir_rate", "coverage"])
    chosen = [c for c in result.comparisons if c.metric in metrics]
    if not chosen:
        return "% no comparisons available\n"

    head = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\begin{tabular}{llrrrr}\n"
        "    \\toprule\n"
        "    Scheduler & Metric & Baseline & Agent & Improvement (95\\% CI) & $p$ \\\\\n"
        "    \\midrule\n"
    )
    body = ""
    for metric in metrics:
        for c in [x for x in chosen if x.metric == metric]:
            mark = "$^{\\dagger}$" if c.significant else ""
            body += (
                f"    \\texttt{{{_latex_escape(c.agent)}}} & "
                f"{_latex_escape(c.metric)} & "
                f"{c.baseline_median:.4g} & {c.agent_median:.4g} & "
                f"{100 * c.improvement:+.1f}\\% "
                f"[{100 * c.ci_lo:+.1f}, {100 * c.ci_hi:+.1f}]{mark} & "
                f"{c.p_holm:.3g} \\\\\n"
            )
        body += "    \\midrule\n"
    body = body.rsplit("    \\midrule\n", 1)[0]

    n_seeds = len({r["seed"] for r in result.rows})
    default_caption = (
        f"Paired comparison against \\texttt{{{_latex_escape(result.baseline)}}} on the "
        f"{result.tier} tier, {n_seeds} seeds. Improvement is signed so positive is "
        f"better; intervals are 95\\% paired bootstrap. $p$ is Wilcoxon signed-rank "
        f"after Holm--Bonferroni correction; $\\dagger$ marks $p < 0.05$."
    )
    tail = (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        f"  \\caption{{{caption or default_caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return head + body + tail


def run_grid(
    tiers: Sequence[str] = ("easy", "medium", "hard"),
    agents: Iterable[str] | None = None,
    n_seeds: int | None = None,
    n_jobs: int = 1,
    out_dir: str | Path = "reports",
    figures: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run the full {schedulers} x {tiers} x {seeds} grid.

    Args:
        tiers: Tier config names (without the ``.yaml``).
        agents: Scheduler keys; defaults to each tier's ``eval.agents``.
        n_seeds: Override the seed count.
        n_jobs: Parallel workers.
        out_dir: Directory for every artefact: ``metrics_<tier>.json``,
            ``results.parquet``, ``leaderboard.md``, ``leaderboard.tex`` and the
            figures.
        figures: Also regenerate the figures.

    Returns:
        Mapping from tier name to :class:`BenchmarkResult`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, BenchmarkResult] = {}
    sections: list[str] = []
    latex: list[str] = []
    last_cfg = None

    for tier in tiers:
        cfg = load_config(f"{tier}.yaml")
        if n_seeds is not None:
            cfg = cfg.with_overrides(run={"n_seeds": n_seeds})
        last_cfg = cfg
        print(
            f"[benchmark] tier={tier} agents={list(agents or cfg.eval.agents)} "
            f"seeds={cfg.run.n_seeds} jobs={n_jobs}"
        )
        res = run_benchmark(cfg, agents=agents, n_jobs=n_jobs)
        res.to_json(out / f"metrics_{tier}.json")
        results[tier] = res
        sections.append(leaderboard_markdown(res))
        latex.append(leaderboard_latex(res, label=f"tab:leaderboard-{tier}"))

    (out / "leaderboard.md").write_text("\n\n".join(sections), encoding="utf-8")
    (out / "leaderboard.tex").write_text("\n".join(latex), encoding="utf-8")
    try:
        write_results_parquet(list(results.values()), out / "results.parquet")
    except ImportError:
        print("[benchmark] pandas missing; skipping results.parquet")

    if figures and last_cfg is not None:
        try:
            _write_figures(results, last_cfg, out, n_seeds or 6)
        except ImportError as exc:
            print(f"[benchmark] figures skipped: {exc}")
    return results


def _write_figures(
    results: dict[str, BenchmarkResult],
    config: Config,
    out: Path,
    n_seeds: int,
) -> None:
    """Regenerate every figure the report needs, from re-run episodes.

    The benchmark keeps only scalar metrics per seed, so the figures need
    trajectories. Re-running a handful of episodes is cheaper and less
    error-prone than carrying every ``(B, T)`` mask through the grid.
    """
    from smartscan.eval import plots

    tier = "medium" if "medium" in results else next(iter(results))
    # Reload per-tier rather than reusing `config`, which is whichever tier ran
    # last; the figures must describe the tier they are labelled with.
    cfg = load_config(f"{tier}.yaml") if tier != config.scenario.difficulty else config
    keys = [a for a in ("sequential", "ucb1", "whittle", "coprime_sweep") if a in
            {r["agent"] for r in results[tier].rows}]
    if not keys:
        keys = ["sequential"]

    per_agent: dict[str, list[Any]] = {k: [] for k in keys}
    for i in range(min(n_seeds, 6)):
        seed = cfg.run.seed + i
        scenario = generate_scenario(seed, config=cfg)
        episode = build_episode(scenario)
        for key in keys:
            per_agent[key].append(
                run_episode(
                    cfg, seed, build_agent(key, cfg, seed, scenario),
                    scenario=scenario, episode=episode,
                )
            )

    written = plots.save_all(per_agent, cfg, out)
    written.append(plots.plot_scan_on_scan(cfg, out / "f6_scan_on_scan.png"))
    ablation = out / "ablation.json"
    if ablation.is_file():
        written.append(plots.plot_ablation_tornado(ablation, out / "f7_ablation_tornado.png"))
    print(f"[benchmark] wrote {len(written)} figures to {out}/")
