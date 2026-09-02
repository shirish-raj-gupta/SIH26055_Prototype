"""Online scan-period estimation from sparse, irregularly sampled intercepts.

Two estimators, implemented so they can be compared head to head:

**Lomb-Scargle** -- the correct periodogram for unevenly sampled data (Lomb 1976;
Scargle 1982). Our samples *are* uneven: we only learn a channel's state when we
happen to be tuned there.

    Critical correction: the hit sequence is the product of emitter activity and
    **our own visit schedule**. A raw periodogram of hits therefore peaks at the
    receiver's sweep period -- the estimator confidently reports its own tail. We
    compute the spectral window ``W(f)`` of the visit times as well and score
    candidate periods by signal power *not explained by* the window. This is the
    single easiest thing in the module to get wrong, and it has its own test.

**CDIF / SDIF** -- cumulative and sequential difference histograms with a
decreasing detection threshold, the classical PRI-deinterleaving technique
(Mardia 1989; Milojevic & Popovic 1992) applied at scan-period timescale. Robust
to missing arrivals via subharmonic checking, weaker under heavy jitter.

All periods here are in **slots**, not seconds; the caller converts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sp_signal

__all__ = [
    "PeriodEstimate",
    "cluster_arrivals",
    "estimate_period",
    "estimate_period_ls",
    "estimate_period_sdif",
    "spectral_window",
]


@dataclass(frozen=True)
class PeriodEstimate:
    """Result of a period estimate.

    Attributes:
        period: Estimated period in slots; ``0.0`` if no estimate was made.
        confidence: Heuristic confidence in ``[0, 1]``.
        method: Which estimator produced it.
        sigma: Standard error of the period estimate, in slots (``inf`` if unknown).
        n_samples: Number of arrival times used.
    """

    period: float
    confidence: float
    method: str
    sigma: float = float("inf")
    n_samples: int = 0

    @property
    def valid(self) -> bool:
        """Whether the estimate carries any information."""
        return self.period > 0.0 and np.isfinite(self.period)


_NULL = PeriodEstimate(0.0, 0.0, "none")


def cluster_arrivals(times: np.ndarray, min_gap: float) -> np.ndarray:
    """Collapse a burst of hits into the single beam arrival that produced it.

    A scanning radar illuminates us for a contiguous stretch of tens to hundreds
    of slots, so the raw hit series is a **pulse train**, not a point process.
    Its Fourier spectrum therefore carries strong energy at every harmonic
    ``Te/k`` up to ``1/we``, and a periodogram argmax lands on whichever harmonic
    happens to be largest -- observed in practice as confident estimates of
    ``Te/4`` and ``Te/2``.

    Clustering first removes the intra-pass structure that creates those
    harmonics, leaving a sparse point process whose fundamental is unambiguous.
    It is also what a real ES system does when it forms scan marks from pulse
    groups, so this is the physically correct pre-processing step rather than a
    numerical convenience.

    Args:
        times: Hit timestamps in slots.
        min_gap: Two hits closer than this belong to the same arrival.

    Returns:
        Float64 array of arrival times (the first hit of each cluster).
    """
    t = np.unique(np.asarray(times, dtype=np.float64))
    if t.size == 0:
        return t
    breaks = np.flatnonzero(np.diff(t) > max(min_gap, 1e-9))
    starts = np.concatenate([[0], breaks + 1])
    return t[starts]


def _fundamental_from_harmonic(
    periods: np.ndarray,
    power: np.ndarray,
    peak_index: int,
    max_harmonic: int = 6,
    ratio: float = 0.45,
) -> int:
    """Prefer the fundamental over a harmonic of the same pulse train.

    A narrow periodic pulse train has comparable power at ``Te``, ``Te/2``,
    ``Te/3`` ..., so an argmax can legitimately land on a harmonic. If ``k * P``
    still carries at least ``ratio`` of the peak power then ``k * P`` is the
    better explanation: a genuine period always implies power at its harmonics,
    but a harmonic does not imply power at the longer period.

    Args:
        periods: Candidate period grid, ascending.
        power: Periodogram power on that grid.
        peak_index: Index of the raw argmax.
        max_harmonic: Largest multiple tested.
        ratio: Fraction of peak power a multiple must retain to be preferred.

    Returns:
        Index of the chosen fundamental.
    """
    best = peak_index
    peak_p = periods[peak_index]
    threshold = ratio * power[peak_index]
    for k in range(2, max_harmonic + 1):
        target = peak_p * k
        if target > periods[-1]:
            break
        j = int(np.argmin(np.abs(periods - target)))
        # The grid is geometric, so an exact multiple rarely lands on a point.
        lo, hi = max(j - 2, 0), min(j + 3, periods.size)
        local = lo + int(np.argmax(power[lo:hi]))
        if power[local] >= threshold:
            best = local
            threshold = ratio * power[local]
    return best


def _period_grid(period_min: float, period_max: float, n_bins: int) -> np.ndarray:
    """Return a geometric period grid.

    Geometric rather than linear: period resolution should be a constant
    *fractional* accuracy, and a linear grid wastes most of its bins on long
    periods where the periodogram is smooth anyway.
    """
    lo = max(period_min, 1e-6)
    hi = max(period_max, lo * 1.001)
    return np.geomspace(lo, hi, int(n_bins))


def spectral_window(sample_times: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Normalised spectral window of a sampling schedule.

    ``W(f) = |sum_j exp(-2*pi*i*f*t_j)|^2 / N^2`` in ``[0, 1]``. A value near 1
    means the *sampling* itself is periodic at that period, so any apparent
    signal power there is suspect.

    Args:
        sample_times: Times at which the channel was observed, in slots.
        periods: Candidate periods, in slots.

    Returns:
        Float64 array of the same shape as ``periods``.
    """
    t = np.asarray(sample_times, dtype=np.float64)
    if t.size == 0:
        return np.zeros_like(periods)
    phase = 2.0 * np.pi * np.outer(1.0 / periods, t)
    return (np.square(np.cos(phase).sum(axis=1)) + np.square(np.sin(phase).sum(axis=1))) / (t.size**2)


def estimate_period_ls(
    hit_times: np.ndarray,
    visit_times: np.ndarray | None = None,
    *,
    period_min: float = 100.0,
    period_max: float = 15000.0,
    n_bins: int = 4000,
    deconvolve_window: bool = True,
    peak_snr_threshold: float = 6.0,
    cluster_gap: float | None = None,
) -> PeriodEstimate:
    """Estimate a scan period by Lomb-Scargle periodogram.

    The observation series is built on the **visit** times (value 1 where a hit
    was declared, 0 where the channel was looked at and found empty), which is
    what makes the sampling genuinely uneven-but-known. When ``visit_times`` is
    not supplied the hit times alone are used as a point process, which is
    weaker because the zeros carry real information.

    Args:
        hit_times: Slot indices at which a detection was declared.
        visit_times: Slot indices at which the channel was observed at all.
        period_min: Shortest candidate period, in slots.
        period_max: Longest candidate period, in slots.
        n_bins: Number of candidate periods.
        deconvolve_window: Divide out the sampling schedule's own spectrum.
        peak_snr_threshold: Peak-to-median power ratio below which the estimate
            is rejected as noise.
        cluster_gap: Hits closer than this (slots) merge into one beam arrival.
            Defaults to ``period_min / 4``; pass ``0`` to disable.

    Returns:
        A :class:`PeriodEstimate`; ``period == 0`` when nothing is resolvable.
    """
    gap = period_min / 4.0 if cluster_gap is None else float(cluster_gap)
    hits = (
        cluster_arrivals(hit_times, gap)
        if gap > 0
        else np.unique(np.asarray(hit_times, dtype=np.float64))
    )
    if hits.size < 4:
        return _NULL

    periods = _period_grid(period_min, period_max, n_bins)
    # Only periods shorter than the observed span can be resolved at all.
    span = float(hits[-1] - hits[0]) if visit_times is None else float(
        np.max(visit_times) - np.min(visit_times)
    )
    resolvable = periods <= max(span / 2.0, period_min)
    if not np.any(resolvable):
        return _NULL
    periods = periods[resolvable]
    ang_freqs = 2.0 * np.pi / periods

    if visit_times is not None and np.size(visit_times) >= hits.size:
        t = np.asarray(visit_times, dtype=np.float64)
        y = np.isin(t, hits).astype(np.float64)
        if y.sum() < 4:  # clustering left too few labels on the visit grid
            t, y = hits, np.ones_like(hits)
    else:
        t, y = hits, np.ones_like(hits)

    y_centred = y - y.mean()
    if not np.any(np.abs(y_centred) > 0):
        return _NULL

    # scipy's lombscargle wants angular frequencies and a zero-mean series.
    power = sp_signal.lombscargle(t, y_centred, ang_freqs, normalize=True)

    if deconvolve_window:
        w = spectral_window(t, periods)
        # Suppress candidates whose power is explained by the sampling schedule.
        # Without this the periodogram reports OUR sweep period with confidence.
        power = power * np.clip(1.0 - w, 0.0, 1.0)

    med = float(np.median(power)) or 1e-12
    k = _fundamental_from_harmonic(periods, power, int(np.argmax(power)))
    peak = float(power[k])
    if peak / med < peak_snr_threshold:
        return PeriodEstimate(0.0, 0.0, "lomb_scargle", n_samples=hits.size)

    period = float(periods[k])
    # Parabolic interpolation on the log-period grid sharpens the peak beyond
    # the grid spacing, which matters because the grid is coarse by design.
    if 0 < k < power.size - 1:
        y0, y1, y2 = power[k - 1], power[k], power[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-18:
            shift = 0.5 * (y0 - y2) / denom
            log_p = np.log(periods)
            step = log_p[k + 1] - log_p[k]
            period = float(np.exp(log_p[k] + shift * step))

    # Cramer-Rao-flavoured error: resolution improves with the number of cycles
    # spanned and with the periodogram peak sharpness.
    n_cycles = max(span / period, 1.0)
    sigma = period / (2.0 * np.pi * n_cycles * np.sqrt(max(hits.size, 1)))
    confidence = float(np.clip(1.0 - np.exp(-(peak / med - peak_snr_threshold) / 6.0), 0.0, 1.0))
    return PeriodEstimate(period, confidence, "lomb_scargle", sigma, hits.size)


def estimate_period_sdif(
    hit_times: np.ndarray,
    *,
    period_min: float = 100.0,
    period_max: float = 15000.0,
    n_bins: int = 400,
    max_order: int = 6,
    threshold_k: float = 3.0,
    subharmonic_check: bool = True,
    cluster_gap: float | None = None,
    min_count: int = 3,
) -> PeriodEstimate:
    """Estimate a scan period by sequential difference histogram (SDIF).

    For each difference order ``k`` the histogram of ``t[i+k] - t[i]`` is
    compared against the Milojevic-Popovic decreasing threshold
    ``E(tau) = c * (N - k) * exp(-tau / (k * span))``. The first order whose
    histogram clears the threshold supplies the period.

    Args:
        hit_times: Slot indices at which a detection was declared.
        period_min: Shortest candidate period, in slots.
        period_max: Longest candidate period, in slots.
        n_bins: Histogram bins across the candidate range.
        max_order: Highest difference order to test.
        threshold_k: Detection threshold in Poisson standard deviations above
            the uniform-arrival expectation; lower is more permissive.
        subharmonic_check: Reject a candidate that is an integer multiple of a
            shorter candidate with comparable support (the classic SDIF failure
            mode when arrivals are missed).
        cluster_gap: Hits closer than this (slots) merge into one arrival.
            Without it the first-order differences are dominated by the
            one-slot gaps *inside* a beam pass and nothing clears threshold.
        min_count: Absolute minimum bin count, so a single coincidence in a
            near-empty histogram cannot be declared a period.

    Returns:
        A :class:`PeriodEstimate`.
    """
    gap = period_min / 4.0 if cluster_gap is None else float(cluster_gap)
    t = (
        cluster_arrivals(hit_times, gap)
        if gap > 0
        else np.unique(np.asarray(hit_times, dtype=np.float64))
    )
    n = t.size
    if n < 4:
        return _NULL
    span = float(t[-1] - t[0])
    if span <= 0:
        return _NULL

    edges = np.linspace(period_min, min(period_max, span), int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    best: tuple[float, float, int] | None = None

    for k in range(1, min(max_order, n - 1) + 1):
        diffs = t[k:] - t[:-k]
        counts, _ = np.histogram(diffs, bins=edges)
        if counts.sum() == 0:
            continue
        # Threshold against a UNIFORM-ARRIVAL null, not against a fixed
        # fraction of the difference count. The classical Milojevic-Popovic
        # form ``c*(N-k)*exp(-tau/(k*span))`` is calibrated for PRI
        # deinterleaving, where N is thousands of pulses; with ~30 scan
        # arrivals it evaluates to ~17 and nothing ever clears it, so SDIF
        # silently returned "no period" on every episode. Here the expected
        # per-bin count under uniform arrivals is tiny, and a Poisson upper
        # tail is the scale-free test that works in both regimes.
        bin_w = float(edges[1] - edges[0])
        expected = (n - k) * bin_w / span * np.exp(-centres / (k * span))
        thresh = np.maximum(min_count, expected + threshold_k * np.sqrt(np.maximum(expected, 1.0)))
        above = counts > thresh
        if not np.any(above):
            continue
        j = int(np.flatnonzero(above)[np.argmax(counts[above])])
        cand = float(centres[j] / k)  # order-k difference spans k periods
        score = float(counts[j] / max(thresh[j], 1e-9))
        if best is None or score > best[1]:
            best = (cand, score, k)

    if best is None:
        return PeriodEstimate(0.0, 0.0, "sdif", n_samples=n)

    period, score, _order = best
    if subharmonic_check:
        # If half (or a third of) the candidate also has strong support, the
        # candidate is a subharmonic produced by missed arrivals.
        for m in (2, 3):
            sub = period / m
            if sub < period_min:
                continue
            d1 = t[1:] - t[:-1]
            near = np.abs(d1 - sub) < max(0.1 * sub, 1.0)
            if near.sum() >= max(2, 0.3 * d1.size):
                period = sub
                break

    d1 = t[1:] - t[:-1]
    near = np.abs(d1 - period) < max(0.15 * period, 1.0)
    sigma = float(np.std(d1[near])) if near.sum() >= 2 else float(edges[1] - edges[0])
    confidence = float(np.clip((score - 1.0) / 4.0, 0.0, 1.0))
    return PeriodEstimate(period, confidence, "sdif", sigma, n)


def estimate_period(
    hit_times: np.ndarray,
    visit_times: np.ndarray | None = None,
    *,
    method: str = "lomb_scargle",
    period_min: float = 100.0,
    period_max: float = 15000.0,
    config: Any | None = None,
    t_now: int | None = None,  # noqa: ARG001 - accepted for caller symmetry
    n_bins_override: int | None = None,
) -> PeriodEstimate:
    """Dispatch to the configured period estimator.

    Args:
        hit_times: Slot indices at which a detection was declared.
        visit_times: Slot indices at which the channel was observed.
        method: ``"lomb_scargle"``, ``"sdif"``, ``"both"`` or ``"none"``.
        period_min: Shortest candidate period, in slots.
        period_max: Longest candidate period, in slots.
        config: Optional ``EstimatorsConfig`` supplying bin counts and thresholds.
        t_now: Current slot (unused; accepted so callers can pass context).
        n_bins_override: Period-grid resolution, overriding the config.

    Returns:
        A :class:`PeriodEstimate`. For ``"both"``, the higher-confidence result.
    """
    if method == "none":
        return _NULL
    n_bins = int(n_bins_override or getattr(config, "n_period_bins", 4000))
    deconv = getattr(config, "deconvolve_window", True)
    ls_thresh = getattr(config, "ls_peak_snr_threshold", 6.0)
    sdif_k = getattr(config, "sdif_threshold_k", 0.6)
    sdif_sub = getattr(config, "sdif_subharmonic_check", True)

    if method == "lomb_scargle":
        return estimate_period_ls(
            hit_times, visit_times, period_min=period_min, period_max=period_max,
            n_bins=n_bins, deconvolve_window=deconv, peak_snr_threshold=ls_thresh,
        )
    if method == "sdif":
        return estimate_period_sdif(
            hit_times, period_min=period_min, period_max=period_max,
            threshold_k=sdif_k, subharmonic_check=sdif_sub,
        )
    if method == "both":
        a = estimate_period_ls(
            hit_times, visit_times, period_min=period_min, period_max=period_max,
            n_bins=n_bins, deconvolve_window=deconv, peak_snr_threshold=ls_thresh,
        )
        b = estimate_period_sdif(
            hit_times, period_min=period_min, period_max=period_max,
            threshold_k=sdif_k, subharmonic_check=sdif_sub,
        )
        return a if a.confidence >= b.confidence else b
    raise ValueError(f"unknown period estimation method {method!r}")
