"""The ten figures of merit named in the problem statement.

Every metric is a pure function with its formula in the docstring, so a reviewer
can check the definition without reading the implementation.

Two statistical points are load-bearing:

**Censoring.** Some emitters are never intercepted. Dropping them from the TTFI
average is the standard error in this literature and it flatters every scheduler
that gives up on hard targets. We keep them as right-censored observations and
report a **Kaplan-Meier** survival curve, whose median is well defined even when
a third of the sample never fails.

**Interceptability.** An emitter below tangential sensitivity is present but
physically undetectable. Counting those slots in the denominator of the
interception ratio would punish a scheduler for not doing the impossible, so the
denominator is *interceptable* slots (``Pd > 0``), not merely occupied ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from smartscan.env.propagation import SNR_FLOOR_DB, min_snr_for_pd
from smartscan.env.types import EpisodeTensors

if TYPE_CHECKING:
    # Annotation-only. Importing Config at runtime would be circular, because
    # config.py validates eval.metrics against METRIC_KEYS defined below.
    from smartscan.config import Config

#: Emitter classes whose interception is genuinely a scheduling problem. The
#: always-on classes (CW, pulsed, interferer) are found on the receiver's first
#: pass by ANY policy, so a median TTFI over all emitters is saturated and cannot
#: discriminate between schedulers. The headline TTFI is reported over these.
HARD_CLASSES: frozenset[str] = frozenset(
    {"CircularScanRadar", "SectorScanRadar", "AgileBeamRadar"}
)

#: Every scalar key :func:`evaluate_episode` emits. ``eval.metrics`` in the
#: config is validated against this set, because a metric name that does not
#: match a produced key is silently skipped by the benchmark -- which is exactly
#: how a comparison goes missing without anyone noticing.
METRIC_KEYS: frozenset[str] = frozenset(
    {
        "ttfi_median_s", "ttfi_p90_s", "ttfi_mean_s", "ttfi_hard_median_s",
        "ttfi_hard_p90_s", "n_hard_emitters", "n_never_intercepted",
        "twir_rate", "twir_coverage", "interception_ratio_raw", "coverage",
        "intercept_rate_per_s", "staleness_max_s", "staleness_mean_s",
        "revisit_p95_s", "fraction_visited", "coverage_entropy",
        "waste_fraction", "popup_latency_s", "popup_detect_rate",
        "n_popup_interceptable", "n_popup_found", "discovery_auc", "fa_burden",
        "pfa_empirical", "reward_total", "reward_discounted", "n_retunes",
        "settle_slots_lost", "n_steps", "wall_time_s",
    }
)

__all__ = [
    "HARD_CLASSES",
    "METRIC_KEYS",
    "BootstrapCI",
    "SurvivalCurve",
    "average_intercept_rate",
    "average_intercept_time_error",
    "average_reward",
    "bootstrap_ci",
    "clopper_pearson",
    "coverage_entropy",
    "empirical_pd",
    "empirical_pfa",
    "evaluate_episode",
    "interception_ratio",
    "kaplan_meier",
    "paired_bootstrap_delta",
    "prediction_scores",
    "roc_auc",
    "sensitivity_db",
    "spectrum_coverage",
    "time_to_first_intercept",
]


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        point: The statistic on the observed sample.
        lo: Lower confidence bound.
        hi: Upper confidence bound.
        level: Confidence level, e.g. 0.95.
        n: Sample size.
    """

    point: float
    lo: float
    hi: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.lo:.4g}, {self.hi:.4g}]"


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval (Clopper-Pearson).

    Exact rather than normal-approximate because ``Pd`` and ``Pfa`` estimates
    routinely sit near 0 or 1, where a Wald interval can extend outside ``[0, 1]``
    and badly under-cover.

    Args:
        k: Number of successes.
        n: Number of trials.
        level: Confidence level.

    Returns:
        ``(lo, hi)`` bounds; ``(0, 0)`` when ``n == 0``.
    """
    if n <= 0:
        return 0.0, 0.0
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def bootstrap_ci(
    values: np.ndarray,
    statistic: Any = np.median,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for a statistic.

    Args:
        values: Sample, typically one value per seed (cluster bootstrap).
        statistic: Callable applied to each resample.
        n_boot: Number of resamples.
        level: Confidence level.
        seed: Seed for the resampling, drawn from the ``eval_bootstrap`` stream.

    Returns:
        A :class:`BootstrapCI`.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), level, 0)
    if v.size == 1:
        return BootstrapCI(float(v[0]), float(v[0]), float(v[0]), level, 1)

    from smartscan.seeding import SeedTree

    rng = SeedTree(seed).rng("eval_bootstrap")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    draws = statistic(v[idx], axis=1)
    a = (1.0 - level) / 2.0
    return BootstrapCI(
        float(statistic(v)), float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)),
        level, int(v.size),
    )


def paired_bootstrap_delta(
    treatment: np.ndarray,
    baseline: np.ndarray,
    relative: bool = True,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Paired bootstrap of the improvement of ``treatment`` over ``baseline``.

    Both arrays must be aligned seed-for-seed. Resampling *seeds* rather than
    observations keeps the pairing intact, which removes scenario variance --
    the reason 30 seeds suffice to support a 25 % claim.

    Args:
        treatment: Per-seed metric for the candidate scheduler.
        baseline: Per-seed metric for the reference scheduler.
        relative: Report a fractional improvement rather than an absolute delta.
        n_boot: Number of resamples.
        level: Confidence level.
        seed: Resampling seed.

    Returns:
        A :class:`BootstrapCI` on the (relative) improvement. Positive means the
        treatment scored *lower* on the metric, which is the improvement
        direction for time-like metrics; callers flip the sign for
        higher-is-better metrics.

    Raises:
        ValueError: If the arrays have different lengths.
    """
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), level, 0)

    from smartscan.seeding import SeedTree

    rng = SeedTree(seed).rng("eval_bootstrap")
    idx = rng.integers(0, a.size, size=(n_boot, a.size))

    def stat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mx, my = np.median(x, axis=-1), np.median(y, axis=-1)
        return (my - mx) / np.maximum(my, 1e-12) if relative else (my - mx)

    draws = stat(a[idx], b[idx])
    q = (1.0 - level) / 2.0
    return BootstrapCI(
        float(stat(a[None, :], b[None, :])[0]),
        float(np.quantile(draws, q)),
        float(np.quantile(draws, 1 - q)),
        level,
        int(a.size),
    )


# --------------------------------------------------------------------------- #
# 1-3. Detection performance
# --------------------------------------------------------------------------- #
def empirical_pd(
    episode: EpisodeTensors,
    visit_mask: np.ndarray,
    true_hit_mask: np.ndarray,
    snr_bins: np.ndarray | None = None,
    level: float = 0.95,
) -> dict[str, np.ndarray]:
    """**Metric 1.** Empirical probability of detection per SNR bin.

    ``Pd(bin) = (visited & occupied & detected) / (visited & occupied)`` over
    cells whose true SNR falls in the bin, with exact Clopper-Pearson intervals.

    Args:
        episode: Ground-truth tensors.
        visit_mask: ``(B, T)`` bool, channel observed at that slot.
        true_hit_mask: ``(B, T)`` bool, genuine detection declared.
        snr_bins: Bin edges in dB; a sensible default is used if omitted.
        level: Confidence level for the intervals.

    Returns:
        Dict with ``snr_centre``, ``pd``, ``lo``, ``hi``, ``n`` arrays.
    """
    if snr_bins is None:
        snr_bins = np.arange(-20.0, 45.0, 2.5)
    occupied = (episode.occupancy > 0) & (episode.snr_db > SNR_FLOOR_DB)
    sel = visit_mask & occupied
    snr = episode.snr_db[sel]
    hit = true_hit_mask[sel]

    idx = np.digitize(snr, snr_bins) - 1
    n_bins = snr_bins.size - 1
    pd = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        m = idx == i
        n = int(m.sum())
        counts[i] = n
        if n:
            k = int(hit[m].sum())
            pd[i] = k / n
            lo[i], hi[i] = clopper_pearson(k, n, level)
    return {
        "snr_centre": 0.5 * (snr_bins[:-1] + snr_bins[1:]),
        "pd": pd, "lo": lo, "hi": hi, "n": counts,
    }


def empirical_pfa(
    episode: EpisodeTensors, visit_mask: np.ndarray, hit_mask: np.ndarray, level: float = 0.95
) -> dict[str, float]:
    """**Metric 2.** Empirical probability of false alarm on noise-only channels.

    ``Pfa = declared hits on unoccupied, visited cells / visited unoccupied cells``.

    Args:
        episode: Ground-truth tensors.
        visit_mask: ``(B, T)`` bool.
        hit_mask: ``(B, T)`` bool of *declared* hits (true and false alike).
        level: Confidence level.

    Returns:
        Dict with ``pfa``, ``lo``, ``hi``, ``n_trials``, ``n_false``.
    """
    noise_only = visit_mask & (episode.occupancy == 0)
    n = int(noise_only.sum())
    k = int((hit_mask & noise_only).sum())
    lo, hi = clopper_pearson(k, n, level)
    return {"pfa": k / n if n else 0.0, "lo": lo, "hi": hi, "n_trials": n, "n_false": k}


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, computed by the rank (Mann-Whitney) identity.

    Args:
        scores: Decision statistic, higher meaning "more likely a signal".
        labels: Binary ground truth.

    Returns:
        AUC in ``[0, 1]``; ``nan`` if either class is empty.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sensitivity_db(config: Config, pd_target: float = 0.9, pfa: float = 1e-3) -> dict[str, float]:
    """**Metric 3.** Minimum SNR at which ``Pd >= pd_target`` at fixed ``Pfa``.

    Reported as a single number per configuration, for both detection regimes.
    The detection curve is monotone in SNR, so this is found by a bracketed root
    solve rather than by fitting -- exact to machine precision.

    Args:
        config: Resolved configuration.
        pd_target: Required detection probability, default 0.9.
        pfa: Fixed false-alarm probability, default 1e-3.

    Returns:
        Dict with ``pulse_single_db`` (one pulse) and ``energy_db``
        (full dwell integration), both in dB.
    """
    det = config.receiver.detector
    grid = config.grid()
    n_energy = int(
        np.clip(
            round(config.time.dt_s * float(grid.widths_hz[0]) / det.fft_size),
            1, det.n_integrate_max,
        )
    )
    return {
        "pulse_single_db": min_snr_for_pd(pd_target, pfa, 1, det.swerling),
        "energy_db": min_snr_for_pd(pd_target, pfa, n_energy, det.swerling),
        "n_integrate_energy": float(n_energy),
    }


# --------------------------------------------------------------------------- #
# 4-5. Intercept performance
# --------------------------------------------------------------------------- #
def _interceptable_mask(episode: EpisodeTensors, pd_tensor: np.ndarray | None = None) -> np.ndarray:
    """Cells where an intercept is physically possible (``Pd > 0``)."""
    if pd_tensor is not None:
        return pd_tensor > 0.0
    return (episode.occupancy > 0) & (episode.snr_db > SNR_FLOOR_DB)


def average_intercept_rate(
    episode: EpisodeTensors, true_hit_mask: np.ndarray, by_class: bool = True
) -> dict[str, float]:
    """**Metric 4.** Intercepts per second, overall and per emitter class.

    Args:
        episode: Ground-truth tensors.
        true_hit_mask: ``(B, T)`` bool of genuine detections.
        by_class: Also break the rate down by emitter class.

    Returns:
        Dict with ``overall`` and, when requested, one key per class.
    """
    duration_s = episode.n_slots * episode.dt_s
    out = {"overall": float(true_hit_mask.sum() / duration_s)}
    if not by_class:
        return out
    cls_of = {t.emitter_id: t.emitter_class for t in episode.truth}
    ids = episode.emitter_id[true_hit_mask]
    for eid, count in zip(*np.unique(ids, return_counts=True), strict=True):
        if eid <= 0:
            continue
        key = cls_of.get(int(eid), "unknown")
        out[key] = out.get(key, 0.0) + float(count / duration_s)
    return out


def interception_ratio(
    episode: EpisodeTensors, true_hit_mask: np.ndarray, pd_tensor: np.ndarray | None = None
) -> dict[str, float]:
    """**Metric 5.** Slots intercepted over slots interceptable, raw and threat-weighted.

    ``ratio_e = detected_slots_e / interceptable_slots_e``; the threat-weighted
    figure is ``sum_e pi_e * ratio_e / sum_e pi_e``.

    The denominator counts *interceptable* slots, not merely occupied ones: an
    emitter below tangential sensitivity cannot be detected at all, and charging
    a scheduler for that would measure the link budget rather than the policy.

    Args:
        episode: Ground-truth tensors.
        true_hit_mask: ``(B, T)`` bool of genuine detections.
        pd_tensor: Optional ``Pd`` tensor defining interceptability.

    Returns:
        Dict with ``raw``, ``threat_weighted``, ``coverage`` (fraction of
        emitters ever intercepted) and ``threat_weighted_coverage``.
    """
    interceptable = _interceptable_mask(episode, pd_tensor)
    eid = episode.emitter_id
    per_emitter: dict[int, tuple[int, int]] = {}
    for t in episode.truth:
        m = interceptable & (eid == t.emitter_id)
        n_avail = int(m.sum())
        n_hit = int((true_hit_mask & m).sum())
        per_emitter[t.emitter_id] = (n_hit, n_avail)

    ratios, weights, ever = [], [], []
    for t in episode.truth:
        n_hit, n_avail = per_emitter[t.emitter_id]
        if n_avail == 0:
            continue
        ratios.append(n_hit / n_avail)
        weights.append(t.threat_priority)
        ever.append(1.0 if n_hit > 0 else 0.0)
    if not ratios:
        return {"raw": 0.0, "threat_weighted": 0.0, "coverage": 0.0, "threat_weighted_coverage": 0.0}

    r = np.asarray(ratios)
    w = np.asarray(weights)
    e = np.asarray(ever)
    return {
        "raw": float(r.mean()),
        "threat_weighted": float((w * r).sum() / max(w.sum(), 1e-12)),
        "coverage": float(e.mean()),
        "threat_weighted_coverage": float((w * e).sum() / max(w.sum(), 1e-12)),
    }


# --------------------------------------------------------------------------- #
# 6. Time to first intercept, with censoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SurvivalCurve:
    """A Kaplan-Meier survival estimate.

    Attributes:
        times: Event times, ascending.
        survival: ``S(t)``, the fraction not yet intercepted.
        median: Median survival time, ``inf`` if ``S`` never reaches 0.5.
        p90: Time by which 90 % have been intercepted, ``inf`` if never.
        n_censored: Number of right-censored observations.
    """

    times: np.ndarray
    survival: np.ndarray
    median: float
    p90: float
    n_censored: int


def kaplan_meier(durations: np.ndarray, observed: np.ndarray) -> SurvivalCurve:
    """Kaplan-Meier estimator with right censoring.

    ``S(t) = prod_{t_i <= t} (1 - d_i / n_i)`` where ``d_i`` is the number of
    events at ``t_i`` and ``n_i`` the number still at risk.

    Emitters never intercepted are *censored at the episode horizon*, not
    dropped. Dropping them is the standard error in this literature: it makes a
    scheduler that abandons hard targets look better than one that keeps trying.

    Args:
        durations: Time to intercept, or horizon for censored observations.
        observed: True where an intercept actually occurred.

    Returns:
        A :class:`SurvivalCurve`.
    """
    d = np.asarray(durations, dtype=np.float64)
    o = np.asarray(observed).astype(bool)
    if d.size == 0:
        return SurvivalCurve(np.zeros(0), np.zeros(0), float("inf"), float("inf"), 0)

    order = np.argsort(d)
    d, o = d[order], o[order]
    times = np.unique(d[o])
    surv, s = [], 1.0
    for t in times:
        at_risk = int((d >= t).sum())
        events = int(((d == t) & o).sum())
        if at_risk > 0:
            s *= 1.0 - events / at_risk
        surv.append(s)
    surv_arr = np.asarray(surv)

    def crossing(level: float) -> float:
        below = np.flatnonzero(surv_arr <= level)
        return float(times[below[0]]) if below.size else float("inf")

    return SurvivalCurve(times, surv_arr, crossing(0.5), crossing(0.1), int((~o).sum()), )


def time_to_first_intercept(
    episode: EpisodeTensors,
    first_intercept: dict[int, int],
    pd_tensor: np.ndarray | None = None,
    only_interceptable: bool = True,
    classes: frozenset[str] | None = None,
) -> dict[str, Any]:
    """**Metric 6.** Time to first intercept per emitter: median, p90, survival curve.

    Time is measured from the emitter's ``t_first_active``, not from ``t = 0``,
    so a pop-up that begins at 6 s is not credited with a 6 s head start.

    Args:
        episode: Ground-truth tensors.
        first_intercept: Slot of first genuine detection per emitter id.
        pd_tensor: Optional ``Pd`` tensor defining interceptability.
        only_interceptable: Exclude emitters that were never detectable at all.
        classes: Restrict to these emitter classes; ``None`` means all.

    Returns:
        Dict with ``median_s``, ``p90_s``, ``mean_s``, ``curve``
        (:class:`SurvivalCurve`), ``durations_s``, ``observed`` and
        ``n_emitters``.
    """
    interceptable = _interceptable_mask(episode, pd_tensor)
    horizon = episode.n_slots
    durations, observed = [], []
    for t in episode.truth:
        if classes is not None and t.emitter_class not in classes:
            continue
        avail = interceptable & (episode.emitter_id == t.emitter_id)
        if only_interceptable and not avail.any():
            continue
        # Clock starts when the emitter first becomes detectable.
        slots = np.flatnonzero(avail.any(axis=0))
        t0 = int(slots[0]) if slots.size else t.t_first_active
        hit = first_intercept.get(t.emitter_id)
        if hit is None:
            durations.append(float(horizon - t0))
            observed.append(False)
        else:
            durations.append(float(max(hit - t0, 0)))
            observed.append(True)

    d = np.asarray(durations, dtype=np.float64)
    o = np.asarray(observed, dtype=bool)
    curve = kaplan_meier(d, o)
    dt = episode.dt_s
    return {
        "median_s": curve.median * dt,
        "p90_s": curve.p90 * dt,
        "mean_s": float(d[o].mean() * dt) if o.any() else float("inf"),
        "curve": curve,
        "durations_s": d * dt,
        "observed": o,
        "n_emitters": int(d.size),
        "n_never": int((~o).sum()),
    }


# --------------------------------------------------------------------------- #
# 7. Reward
# --------------------------------------------------------------------------- #
def average_reward(rewards: np.ndarray, gamma: float = 0.99) -> dict[str, float]:
    """**Metric 7.** Cumulative discounted and undiscounted return.

    Args:
        rewards: Per-decision reward sequence.
        gamma: Discount factor.

    Returns:
        Dict with ``total``, ``mean_per_step`` and ``discounted``.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return {"total": 0.0, "mean_per_step": 0.0, "discounted": 0.0}
    g = np.power(gamma, np.arange(r.size))
    return {
        "total": float(r.sum()),
        "mean_per_step": float(r.mean()),
        "discounted": float((r * g).sum()),
    }


# --------------------------------------------------------------------------- #
# 8. Predictor quality
# --------------------------------------------------------------------------- #
def prediction_scores(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """**Metric 8.** Accuracy, precision, recall, F1 and Brier score.

    Macro-averaged over channels, because a micro average is dominated by the
    handful of always-busy channels and would hide total failure on the rest.

    Args:
        y_true: Binary ground truth, shape ``(B, T)`` or ``(N,)``.
        y_prob: Predicted probabilities, same shape.
        threshold: Decision threshold for the hard metrics.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1``, ``brier`` and
        ``auc``.
    """
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_prob, dtype=np.float64)
    if yt.ndim == 1:
        yt, yp = yt[None, :], yp[None, :]

    acc, prec, rec, f1 = [], [], [], []
    for row_t, row_p in zip(yt, yp, strict=True):
        pred = row_p >= threshold
        tp = float((pred & row_t).sum())
        fp = float((pred & ~row_t).sum())
        fn = float((~pred & row_t).sum())
        tn = float((~pred & ~row_t).sum())
        acc.append((tp + tn) / max(tp + tn + fp + fn, 1))
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        prec.append(p)
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p and r and np.isfinite(p) and np.isfinite(r) else float("nan"))

    return {
        "accuracy": float(np.nanmean(acc)),
        "precision": float(np.nanmean(prec)),
        "recall": float(np.nanmean(rec)),
        "f1": float(np.nanmean(f1)),
        "brier": float(np.mean((yp - yt.astype(float)) ** 2)),
        "auc": roc_auc(yp.ravel(), yt.ravel()),
    }


# --------------------------------------------------------------------------- #
# 9. Arrival prediction
# --------------------------------------------------------------------------- #
def average_intercept_time_error(
    predicted_arrivals_s: np.ndarray, true_arrivals_s: np.ndarray
) -> dict[str, float]:
    """**Metric 9.** Mean ``|t_predicted - t_actual|`` arrival time error.

    Each prediction is matched to its nearest true arrival, which is the right
    pairing when predictions and truth may differ in count (a missed revolution
    should cost accuracy, not alignment).

    Args:
        predicted_arrivals_s: Predicted beam arrival times, seconds.
        true_arrivals_s: True beam arrival times, seconds.

    Returns:
        Dict with ``mean_abs_error_s``, ``median_abs_error_s``, ``n``.
    """
    p = np.asarray(predicted_arrivals_s, dtype=np.float64)
    t = np.asarray(true_arrivals_s, dtype=np.float64)
    if p.size == 0 or t.size == 0:
        return {"mean_abs_error_s": float("nan"), "median_abs_error_s": float("nan"), "n": 0}
    err = np.abs(p[:, None] - t[None, :]).min(axis=1)
    return {
        "mean_abs_error_s": float(err.mean()),
        "median_abs_error_s": float(np.median(err)),
        "n": int(p.size),
    }


# --------------------------------------------------------------------------- #
# 10. Spectrum coverage
# --------------------------------------------------------------------------- #
def coverage_entropy(visit_counts: np.ndarray) -> float:
    """Normalised entropy of the attention distribution across channels.

    ``H = -sum p log p / log B`` with ``p`` the visit share. ``1.0`` means
    attention was spread perfectly evenly; ``0`` means the receiver parked on one
    channel. Reported alongside interception ratio because a scheduler can score
    well on one while failing badly on the other, and the pair together says what
    a single number cannot.

    Args:
        visit_counts: Visits per channel, shape ``(B,)``.

    Returns:
        Normalised entropy in ``[0, 1]``.
    """
    c = np.asarray(visit_counts, dtype=np.float64)
    total = c.sum()
    if total <= 0 or c.size < 2:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(c.size))


def spectrum_coverage(visit_mask: np.ndarray, dt_s: float) -> dict[str, float]:
    """**Metric 10.** Coverage fraction, revisit-interval distribution and entropy.

    Args:
        visit_mask: ``(B, T)`` bool of observed cells.
        dt_s: Slot duration in seconds.

    Returns:
        Dict with ``fraction_visited``, ``coverage_rate_per_s``,
        ``revisit_mean_s``, ``revisit_p95_s``, ``revisit_max_s``,
        ``staleness_max_s`` and ``coverage_entropy``.
    """
    b, t = visit_mask.shape
    counts = visit_mask.sum(axis=1)
    duration = t * dt_s

    gaps: list[float] = []
    worst = 0.0
    for ch in range(b):
        idx = np.flatnonzero(visit_mask[ch])
        if idx.size < 2:
            worst = max(worst, float(t))
            continue
        d = np.diff(idx).astype(np.float64)
        gaps.append(d)
        worst = max(worst, float(max(d.max(), idx[0], t - 1 - idx[-1])))
    all_gaps = np.concatenate(gaps) if gaps else np.zeros(1)

    return {
        "fraction_visited": float((counts > 0).mean()),
        "coverage_rate_per_s": float((counts > 0).sum() / duration),
        "revisit_mean_s": float(all_gaps.mean() * dt_s),
        "revisit_p95_s": float(np.quantile(all_gaps, 0.95) * dt_s),
        "revisit_max_s": float(all_gaps.max() * dt_s),
        "staleness_max_s": float(worst * dt_s),
        "coverage_entropy": coverage_entropy(counts),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def evaluate_episode(result: Any, config: Config, pd_tensor: np.ndarray | None = None) -> dict[str, Any]:
    """Compute every figure of merit for one episode.

    Args:
        result: An :class:`~smartscan.runner.EpisodeResult` with ground truth
            retained.
        config: Resolved configuration.
        pd_tensor: Optional ``Pd`` tensor defining interceptability.

    Returns:
        Flat dict of scalar metrics plus nested detail under ``_detail``.

    Raises:
        ValueError: If the result does not carry its episode tensors.
    """
    ep = result.episode
    if ep is None:
        raise ValueError("evaluate_episode needs result.episode; run with keep_episode=True")

    ttfi = time_to_first_intercept(ep, result.first_intercept, pd_tensor)
    ttfi_hard = time_to_first_intercept(ep, result.first_intercept, pd_tensor, classes=HARD_CLASSES)
    ratio = interception_ratio(ep, result.true_hit_mask, pd_tensor)
    cover = spectrum_coverage(result.visit_mask, ep.dt_s)
    rate = average_intercept_rate(ep, result.true_hit_mask)
    pfa = empirical_pfa(ep, result.visit_mask, result.hit_mask)
    reward = average_reward(result.rewards, config.rl.gamma)

    # Pop-up performance: acceptance test 5.
    #
    # Detection RATE and LATENCY are reported separately, because folding a
    # never-detected pop-up into the latency as "the whole episode" conflates
    # two different failures and produces a metric that saturates: measured on
    # HARD, every scheduler scored ~5.0 s, which is just the arithmetic of one
    # found and one missed. Pop-ups that are never physically interceptable
    # (a scanner whose beam does not come round again before the horizon) are
    # excluded from the denominator -- that is a scenario property, not a
    # scheduling failure.
    interceptable_cells = _interceptable_mask(ep, pd_tensor)
    popups = [t for t in ep.truth if t.t_first_active > 0]
    popup_lat: list[float] = []
    n_popup_interceptable = 0
    n_popup_found = 0
    for t in popups:
        if not (interceptable_cells & (ep.emitter_id == t.emitter_id)).any():
            continue
        n_popup_interceptable += 1
        hit = result.first_intercept.get(t.emitter_id)
        if hit is not None:
            n_popup_found += 1
            popup_lat.append((hit - t.t_first_active) * ep.dt_s)

    # Discovery curve area: how quickly the order of battle is built up.
    n_truth = max(len(ep.truth), 1)
    found = np.zeros(ep.n_slots)
    for slot in result.first_intercept.values():
        found[slot:] += 1
    discovery_auc = float(found.mean() / n_truth)

    declared = int(result.hit_mask.sum())
    n_false = int((result.hit_mask & (ep.occupancy == 0)).sum())

    return {
        "agent": result.agent,
        "seed": result.seed,
        "ttfi_median_s": ttfi["median_s"],
        "ttfi_p90_s": ttfi["p90_s"],
        "ttfi_hard_median_s": ttfi_hard["median_s"],
        "ttfi_hard_p90_s": ttfi_hard["p90_s"],
        "n_hard_emitters": ttfi_hard["n_emitters"],
        "ttfi_mean_s": ttfi["mean_s"],
        "n_never_intercepted": ttfi["n_never"],
        "twir_rate": ratio["threat_weighted"],
        "twir_coverage": ratio["threat_weighted_coverage"],
        "interception_ratio_raw": ratio["raw"],
        "coverage": ratio["coverage"],
        "intercept_rate_per_s": rate["overall"],
        "staleness_max_s": cover["staleness_max_s"],
        "staleness_mean_s": cover["revisit_mean_s"],
        "revisit_p95_s": cover["revisit_p95_s"],
        "fraction_visited": cover["fraction_visited"],
        "coverage_entropy": cover["coverage_entropy"],
        "waste_fraction": result.interferer_dwells / max(result.n_steps, 1),
        "popup_latency_s": float(np.mean(popup_lat)) if popup_lat else float("nan"),
        "popup_detect_rate": (
            n_popup_found / n_popup_interceptable if n_popup_interceptable else float("nan")
        ),
        "n_popup_interceptable": n_popup_interceptable,
        "n_popup_found": n_popup_found,
        "discovery_auc": discovery_auc,
        "fa_burden": n_false / max(declared, 1),
        "pfa_empirical": pfa["pfa"],
        "reward_total": reward["total"],
        "reward_discounted": reward["discounted"],
        "n_retunes": result.n_retunes,
        "settle_slots_lost": result.settle_slots_lost,
        "n_steps": result.n_steps,
        "wall_time_s": result.wall_time_s,
        "_detail": {"ttfi": ttfi, "ttfi_hard": ttfi_hard, "coverage": cover, "rate": rate, "pfa": pfa},
    }
