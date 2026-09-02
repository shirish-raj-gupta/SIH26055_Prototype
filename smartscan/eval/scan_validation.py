"""Validate the scan-period estimators against ground truth (acceptance test 4).

The problem brief asks for a 4.0 s scan period recovered to within 2 %. That is
**not achievable inside a 10 s episode**: 2.5 revolutions yields two or three
beam arrivals, and no estimator can turn that into a 2 % figure under jitter and
missed looks. ``configs/scan_on_scan.yaml`` therefore extends the horizon to
120 s (30 revolutions) for estimator validation only; the 10 s tiers are
untouched for scheduler benchmarking. See ``docs/architecture.md`` §17-A.

Reported for each scanning emitter:

* relative period error ``|T_hat - Te| / Te`` for Lomb-Scargle and CDIF/SDIF;
* **average intercept time error** ``mean |t_predicted - t_true|`` over predicted
  beam arrivals, which is the metric the brief actually names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from smartscan.agents import build_agent
from smartscan.analysis.estimators import (
    cluster_arrivals,
    estimate_period_ls,
    estimate_period_sdif,
)
from smartscan.analysis.metrics import average_intercept_time_error
from smartscan.config import Config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.env.types import EpisodeTensors
from smartscan.hal.simulated import detection_probability_tensor
from smartscan.runner import run_episode

__all__ = ["true_beam_arrivals", "validate_estimators"]

#: Emitter classes with a genuine, estimable scan period.
_PERIODIC = {"CircularScanRadar", "SectorScanRadar"}


def true_beam_arrivals(
    episode: EpisodeTensors,
    emitter_id: int,
    pd_tensor: np.ndarray,
    threshold: float = 0.5,
    min_separation_slots: float = 0.0,
) -> np.ndarray:
    """Ground-truth main-beam arrival times for one emitter, in slots.

    An "arrival" is the start of a contiguous run of slots in which the emitter
    is strongly detectable. Using the ``Pd`` tensor rather than the raw gain
    means the arrivals are the ones a receiver could actually have caught.

    Args:
        episode: Ground-truth tensors.
        emitter_id: Emitter to extract arrivals for.
        pd_tensor: Per-cell detection probability.
        threshold: ``Pd`` above which the beam counts as illuminating us.
        min_separation_slots: Two arrivals closer than this belong to the
            same beam pass. Within one pass ``Pd`` dips below threshold as
            the gain rolls off and then recovers, which would otherwise
            split a single pass into dozens of spurious arrivals. Half a
            scan period is the natural separation: a scanner cannot
            illuminate us twice in less than that.

    Returns:
        Int array of arrival slot indices.
    """
    mask = (episode.emitter_id == emitter_id) & (pd_tensor > threshold)
    on = mask.any(axis=0).astype(np.int8)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], on])) == 1).astype(np.float64)
    if min_separation_slots > 0 and starts.size:
        starts = cluster_arrivals(starts, min_separation_slots)
    return starts.astype(np.int64)


def _predicted_arrivals(period_slots: float, first_hit: float, horizon: int) -> np.ndarray:
    """Extrapolate arrivals from an estimated period and one observed arrival."""
    if period_slots <= 0:
        return np.zeros(0)
    n = int((horizon - first_hit) / period_slots) + 1
    return first_hit + period_slots * np.arange(max(n, 0))


def validate_estimators(
    config: Config,
    n_seeds: int = 10,
    agent: str = "coprime_sweep",
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the estimator validation sweep.

    Args:
        config: Resolved configuration (normally ``configs/scan_on_scan.yaml``).
        n_seeds: Number of scenarios.
        agent: Scheduler used to collect the intercepts. A low-discrepancy sweep
            is the default because its own spectrum is broad, which is the
            friendliest sampling schedule for a periodogram.
        seeds: Explicit seeds, overriding ``n_seeds``.

    Returns:
        Dict with ``summary`` (aggregate errors and the acceptance verdict) and
        ``rows`` (one record per scanning emitter).
    """
    seed_list = list(seeds or range(config.run.seed, config.run.seed + n_seeds))
    dt = config.time.dt_s
    rows: list[dict[str, Any]] = []

    for seed in seed_list:
        scenario = generate_scenario(seed, config=config)
        episode = build_episode(scenario)
        pd = detection_probability_tensor(episode, config)
        result = run_episode(
            config, seed, build_agent(agent, config, seed, scenario),
            scenario=scenario, episode=episode,
        )
        belief = result.belief
        assert belief is not None

        for truth in episode.truth:
            if truth.emitter_class not in _PERIODIC or not np.isfinite(truth.scan_period_s):
                continue
            ch = truth.home_channel
            hits = belief.hit_times(ch)
            visits = belief.visit_times(ch)
            if hits.size < 4:
                rows.append({
                    "seed": seed, "emitter_id": truth.emitter_id,
                    "emitter_class": truth.emitter_class,
                    "true_period_s": truth.scan_period_s, "n_hits": int(hits.size),
                    "ls_period_s": float("nan"), "ls_rel_error": float("nan"),
                    "sdif_period_s": float("nan"), "sdif_rel_error": float("nan"),
                    "ls_confidence": 0.0, "sdif_confidence": 0.0,
                    "best_method": "none", "best_rel_error": float("nan"),
                    "mean_arrival_error_s": float("nan"), "n_true_arrivals": 0,
                })
                continue

            lo = config.analysis.estimators.period_grid_s.lo / dt
            hi = config.analysis.estimators.period_grid_s.hi / dt
            ls = estimate_period_ls(
                hits, visits, period_min=lo, period_max=hi,
                n_bins=config.analysis.estimators.n_period_bins,
                deconvolve_window=config.analysis.estimators.deconvolve_window,
                peak_snr_threshold=config.analysis.estimators.ls_peak_snr_threshold,
            )
            sdif = estimate_period_sdif(
                hits, period_min=lo, period_max=hi,
                threshold_k=config.analysis.estimators.sdif_threshold_k,
                subharmonic_check=config.analysis.estimators.sdif_subharmonic_check,
            )

            true_p = truth.scan_period_s
            best = ls if ls.confidence >= sdif.confidence else sdif
            arrivals_true = (
                true_beam_arrivals(
                    episode, truth.emitter_id, pd, min_separation_slots=0.5 * true_p / dt
                )
                * dt
            )
            arrivals_pred = _predicted_arrivals(best.period, float(hits[0]), episode.n_slots) * dt
            err = average_intercept_time_error(arrivals_pred, arrivals_true)

            rows.append({
                "seed": seed,
                "emitter_id": truth.emitter_id,
                "emitter_class": truth.emitter_class,
                "true_period_s": true_p,
                "n_hits": int(hits.size),
                "ls_period_s": ls.period * dt,
                "ls_rel_error": abs(ls.period * dt - true_p) / true_p if ls.valid else float("nan"),
                "ls_confidence": ls.confidence,
                "sdif_period_s": sdif.period * dt,
                "sdif_rel_error": abs(sdif.period * dt - true_p) / true_p if sdif.valid else float("nan"),
                "sdif_confidence": sdif.confidence,
                "best_method": best.method,
                "best_rel_error": abs(best.period * dt - true_p) / true_p if best.valid else float("nan"),
                "mean_arrival_error_s": err["mean_abs_error_s"],
                "n_true_arrivals": int(arrivals_true.size),
            })

    def agg(key: str) -> float:
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    ls_err, sdif_err, best_err = agg("ls_rel_error"), agg("sdif_rel_error"), agg("best_rel_error")
    # Acceptance test 4 targets a 4.0 s period specifically.
    near_four = [
        r for r in rows
        if abs(r["true_period_s"] - 4.0) < 0.25 and np.isfinite(r.get("best_rel_error", np.nan))
    ]
    four_err = float(np.median([r["best_rel_error"] for r in near_four])) if near_four else float("nan")

    return {
        "summary": {
            "n_emitters": len(rows),
            "n_resolved": int(sum(np.isfinite(r.get("best_rel_error", np.nan)) for r in rows)),
            "episode_s": config.time.episode_s,
            "median_rel_error_lomb_scargle": ls_err,
            "median_rel_error_sdif": sdif_err,
            "median_rel_error_best": best_err,
            "median_rel_error_at_4s": four_err,
            "median_arrival_time_error_s": agg("mean_arrival_error_s"),
            "acceptance_4s_within_2pct": bool(np.isfinite(four_err) and four_err <= 0.02),
            "config_hash": config.hash(),
        },
        "rows": rows,
    }
