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
    "leaderboard_markdown",
    "run_benchmark",
    "run_grid",
]

#: Metrics where a *smaller* value is better. Everything else is
#: higher-is-better, and the sign of the reported improvement is flipped.
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
    """

    rows: list[dict[str, Any]]
    comparisons: list[Comparison]
    config_hash: str
    tier: str
    baseline: str

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
            res = run_episode(
                config, seed, build_agent(key, config, seed, scenario),
                scenario=scenario, episode=episode,
            )
            row = evaluate_episode(res, config)
            row.pop("_detail", None)
            row["agent"] = key
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
            if ok.sum() < 3:
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
    return "\n".join(lines)


def run_grid(
    tiers: Sequence[str] = ("easy", "medium", "hard"),
    agents: Iterable[str] | None = None,
    n_seeds: int | None = None,
    n_jobs: int = 1,
    out_dir: str | Path = "reports",
) -> dict[str, BenchmarkResult]:
    """Run the full {schedulers} x {tiers} x {seeds} grid.

    Args:
        tiers: Tier config names (without the ``.yaml``).
        agents: Scheduler keys; defaults to each tier's ``eval.agents``.
        n_seeds: Override the seed count.
        n_jobs: Parallel workers.
        out_dir: Directory for ``metrics_<tier>.json`` and the leaderboard.

    Returns:
        Mapping from tier name to :class:`BenchmarkResult`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, BenchmarkResult] = {}
    sections: list[str] = []

    for tier in tiers:
        cfg = load_config(f"{tier}.yaml")
        if n_seeds is not None:
            cfg = cfg.with_overrides(run={"n_seeds": n_seeds})
        print(f"[benchmark] tier={tier} agents={list(agents or cfg.eval.agents)} seeds={cfg.run.n_seeds}")
        res = run_benchmark(cfg, agents=agents, n_jobs=n_jobs)
        res.to_json(out / f"metrics_{tier}.json")
        results[tier] = res
        sections.append(leaderboard_markdown(res))

    (out / "leaderboard.md").write_text("\n\n".join(sections), encoding="utf-8")
    return results
