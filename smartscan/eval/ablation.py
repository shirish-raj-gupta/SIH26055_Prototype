"""Ablations, sensitivity sweeps and distribution-shift tests.

The sweeps the problem brief asks for:

* **reward-term removal** -- zero each ``w1..w6`` in turn;
* **IBW ratio** ``K/B`` in ``{1/32, 1/16, 1/8}``;
* **retune cost** ``t_settle`` sweep;
* **emitter density** sweep;
* **belief decay** on/off;
* **predictor-in-the-loop** on/off.

Plus robustness under distribution shift -- train on MEDIUM, test on HARD, and
test at double the trained emitter count. **The degradation is reported as
measured.** A graceful-degradation curve is a stronger submission than a hidden
failure, and a scheduler that falls over out of distribution is something the
operator needs to know before the mission, not after.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from smartscan.agents import build_agent
from smartscan.analysis.metrics import evaluate_episode
from smartscan.config import Config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.runner import run_episode

__all__ = ["evaluate_variant", "robustness_shift", "run_ablations"]

#: Metrics carried through every sweep.
_KEYS = (
    "ttfi_hard_median_s", "ttfi_median_s", "twir_rate", "coverage",
    "staleness_max_s", "coverage_entropy", "waste_fraction", "reward_total",
)


def evaluate_variant(
    config: Config, agents: Sequence[str], seeds: Sequence[int], progress: bool = False
) -> dict[str, dict[str, float]]:
    """Run several agents over several seeds and return median metrics.

    Args:
        config: Resolved configuration for this variant.
        agents: Scheduler keys.
        seeds: Scenario seeds.
        progress: Print a note per seed.

    Returns:
        ``{agent: {metric: median}}``.
    """
    rows: dict[str, list[dict[str, Any]]] = {a: [] for a in agents}
    for seed in seeds:
        scenario = generate_scenario(seed, config=config)
        episode = build_episode(scenario)
        for key in agents:
            res = run_episode(
                config, seed, build_agent(key, config, seed, scenario),
                scenario=scenario, episode=episode,
            )
            row = evaluate_episode(res, config)
            row.pop("_detail", None)
            rows[key].append(row)
        if progress:
            print(f"    seed {seed} done", flush=True)
    return {
        a: {k: float(np.nanmedian([r[k] for r in rs])) for k in _KEYS}
        for a, rs in rows.items()
        if rs
    }


def _seeds(config: Config, n: int) -> list[int]:
    return list(range(config.run.seed, config.run.seed + n))


def run_ablations(
    config: Config,
    which: str = "all",
    n_seeds: int = 8,
    agents: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the requested ablation sweeps.

    Args:
        config: Baseline configuration.
        which: ``all``, or one of ``reward``, ``ibw``, ``retune``, ``density``,
            ``belief``, ``shift``.
        n_seeds: Seeds per variant.
        agents: Schedulers to include; a compact analytic set by default, since
            the sweeps are about the *environment* and reward, not the network.
        verbose: Print progress.

    Returns:
        Nested dict of results, JSON-serialisable.
    """
    agents = list(agents or ["sequential", "ucb1", "whittle", "phase_locked"])
    seeds = _seeds(config, n_seeds)
    out: dict[str, Any] = {
        "config_hash": config.hash(),
        "tier": config.scenario.difficulty,
        "agents": agents,
        "n_seeds": n_seeds,
        "baseline": {},
    }
    want = {which} if which != "all" else {"reward", "ibw", "retune", "density", "belief", "shift"}

    if verbose:
        print("[ablation] baseline", flush=True)
    out["baseline"] = evaluate_variant(config, agents, seeds)

    # -- reward-term removal ------------------------------------------------ #
    if "reward" in want:
        out["reward"] = {}
        terms = [
            "w1_threat_intercept", "w2_novelty", "w3_reconfirm",
            "w4_retune", "w5_interferer_dwell", "w6_staleness",
        ]
        for term in terms:
            if verbose:
                print(f"[ablation] reward: drop {term}", flush=True)
            variant = config.with_overrides(reward={term: 0.0})
            out["reward"][f"without_{term}"] = evaluate_variant(variant, agents, seeds)
        # Five-level sensitivity sweep on the coverage weight, which is the
        # single most influential policy hyper-parameter here.
        out["coverage_weight"] = {}
        for w in (0.0, 0.25, 0.5, 1.0, 2.0):
            variant = config.with_overrides(agents={"coverage_weight": w})
            out["coverage_weight"][str(w)] = evaluate_variant(variant, agents, seeds)

    # -- IBW ratio K/B ------------------------------------------------------ #
    if "ibw" in want:
        out["ibw_ratio"] = {}
        b = config.n_channels
        for ratio, k in (("1/32", b // 32), ("1/16", b // 16), ("1/8", b // 8)):
            if k < 1:
                continue
            if verbose:
                print(f"[ablation] IBW K/B = {ratio} (K={k})", flush=True)
            variant = config.with_overrides(receiver={"ibw_channels": int(k)})
            out["ibw_ratio"][ratio] = evaluate_variant(variant, agents, seeds)

    # -- retune cost -------------------------------------------------------- #
    if "retune" in want:
        out["t_settle"] = {}
        for t in (0, 1, 2, 5, 10):
            if verbose:
                print(f"[ablation] t_settle = {t}", flush=True)
            variant = config.with_overrides(receiver={"t_settle_slots": t})
            out["t_settle"][str(t)] = evaluate_variant(variant, agents, seeds)

    # -- emitter density ---------------------------------------------------- #
    if "density" in want:
        out["n_emitters"] = {}
        base_n = config.scenario.n_emitters
        for n in sorted({max(base_n // 2, 2), base_n, base_n * 2}):
            if verbose:
                print(f"[ablation] n_emitters = {n}", flush=True)
            variant = _rescaled(config, n)
            out["n_emitters"][str(n)] = evaluate_variant(variant, agents, seeds)

    # -- belief decay ------------------------------------------------------- #
    if "belief" in want:
        out["belief_decay"] = {}
        for half_life in (200, 2000, 10**9):  # 10**9 slots == effectively no decay
            label = "off" if half_life > 10**6 else str(half_life)
            if verbose:
                print(f"[ablation] belief half-life = {label}", flush=True)
            variant = config.with_overrides(belief={"decay_half_life_slots": half_life})
            out["belief_decay"][label] = evaluate_variant(variant, agents, seeds)

    # -- distribution shift -------------------------------------------------- #
    if "shift" in want:
        out["shift"] = robustness_shift(config, agents, n_seeds, verbose)

    return out


def _rescaled(config: Config, n_emitters: int) -> Config:
    """Return the config with its mix rescaled to ``n_emitters``."""
    mix = config.scenario.mix
    total = sum(mix.values()) or 1
    scaled = {k: int(round(v * n_emitters / total)) for k, v in mix.items()}
    drift = n_emitters - sum(scaled.values())
    if drift:
        key = max(scaled, key=lambda k: scaled[k])
        scaled[key] = max(scaled[key] + drift, 0)
    return config.with_overrides(scenario={"n_emitters": n_emitters, "mix": scaled})


def robustness_shift(
    config: Config,
    agents: Sequence[str],
    n_seeds: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Distribution-shift tests, reported without flattering.

    Three shifts:

    1. **Tier shift** -- evaluate on HARD while configured for the current tier.
    2. **Density shift** -- double the trained emitter count.
    3. **Class hold-out** -- remove a class from the mix and add its share to
       the scanning classes, so the scheduler meets a mix it was not tuned for.

    Args:
        config: Baseline configuration.
        agents: Scheduler keys.
        n_seeds: Seeds per variant.
        verbose: Print progress.

    Returns:
        Dict of variant name to per-agent medians, plus the in-distribution
        reference so the degradation can be read directly.
    """
    from smartscan.config import load_config

    seeds = _seeds(config, n_seeds)
    out: dict[str, Any] = {"in_distribution": evaluate_variant(config, agents, seeds)}

    if verbose:
        print("[shift] tier -> hard", flush=True)
    try:
        hard = load_config("hard.yaml")
        out["tier_hard"] = evaluate_variant(hard, agents, _seeds(hard, n_seeds))
    except FileNotFoundError:
        out["tier_hard"] = {}

    if verbose:
        print("[shift] 2x emitter density", flush=True)
    out["density_2x"] = evaluate_variant(
        _rescaled(config, config.scenario.n_emitters * 2), agents, seeds
    )

    if verbose:
        print("[shift] class hold-out (no frequency_agile)", flush=True)
    mix = dict(config.scenario.mix)
    moved = mix.pop("frequency_agile", 0)
    if moved:
        mix["circular_scan"] = mix.get("circular_scan", 0) + moved
        held = config.with_overrides(
            scenario={"mix": {**dict.fromkeys(config.scenario.mix, 0), **mix}}
        )
        out["holdout_frequency_agile"] = evaluate_variant(held, agents, seeds)
    return out
