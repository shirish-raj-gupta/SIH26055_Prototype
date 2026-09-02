"""Scan-on-scan theory, period estimators, metrics and the Whittle index."""

from __future__ import annotations

import numpy as np
import pytest

from smartscan.agents.whittle import GilbertElliott, whittle_index_curve
from smartscan.analysis.estimators import (
    estimate_period_ls,
    estimate_period_sdif,
    spectral_window,
)
from smartscan.analysis.metrics import (
    average_intercept_time_error,
    bootstrap_ci,
    clopper_pearson,
    coverage_entropy,
    kaplan_meier,
    paired_bootstrap_delta,
    prediction_scores,
    roc_auc,
    sensitivity_db,
)
from smartscan.analysis.scan_on_scan import (
    GOLDEN,
    analyse_coincidence,
    beam_dwell_s,
    expected_time_to_intercept,
    poi_exponential,
    probability_of_intercept,
    three_distance_gaps,
)
from smartscan.config import load_config
from smartscan.eval.benchmark import holm_bonferroni


# --------------------------------------------------------------------------- #
# Scan-on-scan coincidence theory
# --------------------------------------------------------------------------- #
def test_beam_dwell_formula():
    """(beamwidth / 360) * Ts -- 11 ms for a 1 deg beam on a 4 s scan."""
    assert beam_dwell_s(1.0, 4.0) == pytest.approx(4.0 / 360.0)
    assert beam_dwell_s(1.0, 4.0) * 1000 == pytest.approx(11.1, abs=0.1)


def test_commensurate_sweep_can_be_permanently_blind():
    """The synchronism pathology: POI = 0 forever, for most initial phases.

    This is the failure mode the whole module exists to expose, and the one the
    exponential model cannot represent at all.
    """
    te, we, wr = 4.0, beam_dwell_s(2.0, 4.0), 1e-3
    blind = analyse_coincidence(2.0, te, wr, we, horizon_s=600.0)
    assert blind.commensurate
    assert blind.rational == (1, 2)
    assert blind.blind_fraction > 0.9, "a 1/2-commensurate sweep must lock out"

    # The exponential approximation claims certain intercept for the same case.
    assert float(poi_exponential(2.0, te, wr, we, 600.0)) > 0.5


def test_incommensurate_sweep_always_intercepts():
    te, we, wr = 4.0, beam_dwell_s(3.0, 4.0), 3e-3
    r = analyse_coincidence(0.096 * GOLDEN, te, wr, we, horizon_s=600.0)
    assert r.blind_fraction < 0.05
    assert np.isfinite(r.mean_tti_s)


def test_poi_is_monotone_and_bounded():
    te, we, wr = 4.0, beam_dwell_s(2.0, 4.0), 1e-3
    t = np.linspace(0.1, 60.0, 60)
    poi = probability_of_intercept(0.096, te, wr, we, t)
    assert np.all((poi >= 0) & (poi <= 1))
    assert np.all(np.diff(poi) >= -1e-12)


def test_closed_form_tti():
    assert expected_time_to_intercept(0.1, 4.0, 1e-3, 0.02) == pytest.approx(0.1 * 4.0 / 0.021)


def test_three_distance_theorem():
    """At most three distinct gap lengths; the golden ratio minimises the largest."""
    for alpha in (GOLDEN % 1, np.pi % 1, np.sqrt(2) % 1):
        assert len(three_distance_gaps(alpha, 60)) <= 3
    golden_gap = three_distance_gaps(GOLDEN % 1, 60).max()
    for alpha in (0.5, 0.25, 0.1, 0.2):
        assert three_distance_gaps(alpha, 60).max() >= golden_gap


# --------------------------------------------------------------------------- #
# Period estimators
# --------------------------------------------------------------------------- #
def test_lomb_scargle_recovers_a_clean_period():
    period = 400.0
    hits = np.arange(0, 8000, period) + np.array([0, 1, -1, 0, 2, -2, 1, 0, -1, 1, 0, 0, 1, -1, 0, 2, 0, 1, 0, 0])[
        : len(np.arange(0, 8000, period))
    ]
    visits = np.sort(np.unique(np.concatenate([hits, np.arange(0, 8000, 7)])))
    est = estimate_period_ls(hits, visits, period_min=50, period_max=3000, n_bins=3000)
    assert est.valid
    assert abs(est.period - period) / period < 0.05


def test_lomb_scargle_deconvolves_the_sampling_window():
    """Without window deconvolution the periodogram reports OUR sweep period.

    This is the single easiest thing in the module to get wrong, so it gets its
    own test: hits driven ONLY by the visit schedule must not be reported as a
    confident emitter period.
    """
    sweep = 96.0
    visits = np.arange(0, 9000, sweep)
    # Every third visit "hits" -- structure that comes entirely from sampling.
    hits = visits[::3]
    with_deconv = estimate_period_ls(hits, visits, period_min=50, period_max=3000, deconvolve_window=True)
    without = estimate_period_ls(hits, visits, period_min=50, period_max=3000, deconvolve_window=False)
    assert with_deconv.confidence <= without.confidence


def test_spectral_window_peaks_at_the_sampling_period():
    """A periodic sampler's window is a comb: it peaks at the period AND its
    sub-multiples, so the check is "near 1 at 100" rather than "argmax is 100".
    """
    visits = np.arange(0, 4000, 100.0)
    periods = np.linspace(50, 400, 351)
    w = spectral_window(visits, periods)
    assert np.all((w >= 0) & (w <= 1.0 + 1e-9))
    at_period = w[int(np.argmin(np.abs(periods - 100.0)))]
    at_offbeat = w[int(np.argmin(np.abs(periods - 137.0)))]
    assert at_period > 0.9, "the window must peak at the sampling period"
    assert at_offbeat < 0.1, "and be small away from it and its harmonics"
    # Sub-multiples are genuine comb teeth, not spurious.
    assert w[int(np.argmin(np.abs(periods - 50.0)))] > 0.9


def test_sdif_recovers_a_clean_period():
    period = 500.0
    hits = np.arange(0, 9000, period)
    est = estimate_period_sdif(hits, period_min=50, period_max=3000)
    assert est.valid
    assert abs(est.period - period) / period < 0.1


def test_estimators_fail_gracefully_on_aperiodic_arrivals():
    """An agile-beam emitter has no period; the estimator must not invent one."""
    rng = np.random.default_rng(0)
    hits = np.cumsum(rng.gamma(1.2, 400, size=25))
    hits = hits[hits < 9000]
    for est in (
        estimate_period_ls(hits, period_min=50, period_max=3000),
        estimate_period_sdif(hits, period_min=50, period_max=3000),
    ):
        assert est.confidence < 0.9


def test_estimators_return_null_on_too_few_hits():
    assert not estimate_period_ls(np.array([10.0, 20.0])).valid
    assert not estimate_period_sdif(np.array([10.0, 20.0])).valid


# --------------------------------------------------------------------------- #
# Whittle index
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p01,p11",
    [(0.02, 0.9), (0.3, 0.35), (0.6, 0.2), (0.001, 0.999), (0.5, 0.5)],
)
def test_whittle_index_is_indexable_and_monotone(p01: float, p11: float):
    """Indexability is VERIFIED numerically, not assumed."""
    _, index, indexable = whittle_index_curve(GilbertElliott(p01, p11))
    assert indexable, f"indexability failed for p01={p01}, p11={p11}"
    assert np.all(np.diff(index) >= -1e-12)
    assert index[0] <= index[-1]


def test_whittle_matches_myopic_for_identical_positively_correlated_channels():
    """Liu & Zhao (2010): myopic is optimal for identical positively correlated arms.

    The index must therefore be monotone increasing in the belief, so ranking by
    index and ranking by P(busy) give the same order.
    """
    _, index, _ = whittle_index_curve(GilbertElliott(0.05, 0.85))
    omega = np.linspace(0, 1, index.size)
    order_index = np.argsort(index, kind="stable")
    order_myopic = np.argsort(omega, kind="stable")
    assert np.array_equal(np.sort(index[order_index]), index[order_myopic])


def test_gilbert_elliott_stationary_probability():
    ge = GilbertElliott(0.1, 0.8)
    assert ge.stationary == pytest.approx(0.1 / (1 + 0.1 - 0.8))
    assert ge.positively_correlated


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_clopper_pearson_brackets_the_estimate():
    lo, hi = clopper_pearson(5, 100)
    assert lo < 0.05 < hi
    assert clopper_pearson(0, 100)[0] == 0.0
    assert clopper_pearson(100, 100)[1] == 1.0
    assert clopper_pearson(0, 0) == (0.0, 0.0)


def test_kaplan_meier_keeps_censored_observations():
    """Dropping never-intercepted emitters is the standard error here."""
    durations = np.array([1.0, 2.0, 3.0, 10.0, 10.0])
    observed = np.array([True, True, True, False, False])
    curve = kaplan_meier(durations, observed)
    assert curve.n_censored == 2
    # S(1)=0.8, S(2)=0.6, S(3)=0.4 -> the median is the first crossing of 0.5.
    assert curve.median == pytest.approx(3.0)
    assert np.all(np.diff(curve.survival) <= 1e-12)
    assert curve.survival[-1] > 0.0

    # THE POINT: naively dropping the two censored emitters gives a median of
    # 2.0 -- a 33 % optimistic bias, and the standard error in this literature.
    naive = kaplan_meier(durations[observed], observed[observed])
    assert naive.median == pytest.approx(2.0)
    assert curve.median > naive.median


def test_kaplan_meier_all_censored():
    curve = kaplan_meier(np.array([5.0, 5.0]), np.array([False, False]))
    assert not np.isfinite(curve.median)


def test_roc_auc():
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_prediction_scores():
    y = np.array([[0, 1, 1, 0]])
    p = np.array([[0.1, 0.9, 0.8, 0.2]])
    s = prediction_scores(y, p)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(1.0)
    assert s["brier"] < 0.05


def test_coverage_entropy():
    assert coverage_entropy(np.ones(64)) == pytest.approx(1.0)
    one_hot = np.zeros(64)
    one_hot[0] = 100
    assert coverage_entropy(one_hot) == pytest.approx(0.0)


def test_average_intercept_time_error():
    err = average_intercept_time_error(np.array([1.0, 2.1, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert err["mean_abs_error_s"] == pytest.approx(0.1 / 3, abs=1e-9)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    ci = bootstrap_ci(rng.normal(5.0, 1.0, size=40), n_boot=2000, seed=1)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.n == 40


def test_paired_bootstrap_detects_a_real_improvement():
    base = np.linspace(10, 20, 30)
    better = base * 0.7  # a uniform 30 % improvement on a lower-is-better metric
    d = paired_bootstrap_delta(better, base, relative=True, n_boot=2000, seed=1)
    assert d.point == pytest.approx(0.3, abs=0.02)
    assert d.lo > 0.2


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap_delta(np.zeros(4), np.zeros(5))


def test_holm_bonferroni_is_monotone_and_conservative():
    raw = [0.001, 0.01, 0.04, 0.5]
    adj = holm_bonferroni(raw)
    assert np.all(adj >= np.asarray(raw))
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj <= 1.0)
    assert holm_bonferroni([]).size == 0


def test_sensitivity_is_a_single_number_per_config():
    """PS metric 3: minimum SNR for Pd >= 0.9 at Pfa = 1e-3."""
    s = sensitivity_db(load_config("medium.yaml"))
    assert np.isfinite(s["pulse_single_db"]) and np.isfinite(s["energy_db"])
    # Integration gain must make the energy regime the more sensitive one.
    assert s["energy_db"] < s["pulse_single_db"]
