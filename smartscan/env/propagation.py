"""Link budget and probabilistic detection for a square-law envelope detector.

Nothing here hard-codes ``Pd = 1``. Detection is a genuine statistical event, and
that is what makes dwell time a resource worth scheduling: a weak emitter seen
briefly is *probably* missed, so revisit strategy matters.

Receiver model
--------------
A **channelised** ES receiver. Within the tuned IBW each channel is processed by
an ``fft_size``-point FFT whose periodograms are non-coherently averaged over the
dwell. Two detection regimes follow, and which one applies is a property of the
*emitter*, not the receiver (``docs/architecture.md`` §6):

``energy``
    Continuous or bursty signals (CW, comms, broadcast interferers). Signal power
    concentrates in one FFT bin, giving ``10*log10(fft_size)`` dB of processing
    gain against a noise bandwidth of ``channel_width / fft_size``. ``N``
    periodograms are non-coherently integrated over the dwell.

``pulse``
    Radar pulses. An ES receiver has no matched filter -- it does not know the
    waveform -- so it detects on a wide video bandwidth of roughly ``1 / PW``.
    Each pulse is an independent single-sample (``N = 1``) detection opportunity,
    and the dwell succeeds if any pulse is detected:
    ``Pd_dwell = 1 - (1 - Pd_pulse) ** n_pulses``.

    Integrating a 1 us pulse over a 1 ms dwell would bury it by 30 dB, which is
    precisely why real ES receivers use fast log-video detection. Modelling this
    makes "dwell longer" a real trade-off against "hop more often" rather than a
    free win.

Statistics
----------
Let ``W = Z / (2 * sigma^2)`` be the normalised detector output, ``Z`` the sum of
``N`` squared-envelope samples. Under H0, ``W ~ Gamma(N, 1)`` (Erlang), so the
threshold for a required false-alarm probability is
``T = gammaincinv(N, 1 - Pfa)``.

* **Swerling 0** (non-fluctuating): ``2W ~ ncx2(2N, 2*N*chi)`` exactly, hence
  ``Pd = ncx2.sf(2T, 2N, 2*N*chi)``. This is exact, not an approximation.
* **Swerling I** (Rayleigh amplitude, constant across the dwell -- the right
  default for a scanning radar seen through one beam pass) uses the standard
  closed form; for ``N = 1`` it collapses to ``Pd = Pfa ** (1 / (1 + chi))``.

Reference: M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
McGraw-Hill 2014, ch. 6. Albersheim's equation is provided for cross-checking in
tests only (validity ``1e-7 <= Pfa <= 1e-3``, ``0.1 <= Pd <= 0.9``,
``1 <= N <= 8096``); it never appears in the hot path.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "BOLTZMANN_DBM_HZ",
    "SNR_FLOOR_DB",
    "albersheim_snr_db",
    "atmospheric_loss_db",
    "detection_threshold",
    "free_space_path_loss_db",
    "link_budget_dbm",
    "min_snr_for_pd",
    "noise_power_dbm",
    "p_detect",
    "p_detect_dwell",
]

#: kTB at 290 K, in dBm/Hz.
BOLTZMANN_DBM_HZ: float = -173.975

#: Sentinel used in ``SNR[b, t]`` where no emitter is present. Chosen finite so
#: the tensors stay float32-clean and hashable; -200 dB is unreachable physically.
SNR_FLOOR_DB: float = -200.0

#: Above this many integrations the exact Swerling forms are replaced by a
#: moment-matched Gaussian, which is their asymptotic limit and avoids
#: ``gammaincinv`` conditioning problems at very large shape parameters.
_N_EXACT_MAX: int = 4096


# --------------------------------------------------------------------------- #
# Link budget
# --------------------------------------------------------------------------- #
def free_space_path_loss_db(f_hz: np.ndarray | float, range_km: np.ndarray | float) -> np.ndarray:
    """Free-space path loss.

    ``FSPL = 32.44 + 20*log10(f_MHz) + 20*log10(R_km)`` dB.

    Args:
        f_hz: Frequency in Hz.
        range_km: Range in km.

    Returns:
        Loss in dB, broadcast over the inputs.
    """
    f_mhz = np.asarray(f_hz, dtype=np.float64) / 1e6
    r_km = np.asarray(range_km, dtype=np.float64)
    return 32.44 + 20.0 * np.log10(f_mhz) + 20.0 * np.log10(np.maximum(r_km, 1e-6))


def atmospheric_loss_db(f_hz: np.ndarray | float, range_km: np.ndarray | float) -> np.ndarray:
    """Clear-air atmospheric attenuation.

    A linear-in-frequency fit to the ITU-R P.676 sea-level specific attenuation
    over 0.5-18 GHz: ~0.005 dB/km at 1 GHz rising to ~0.03 dB/km at 18 GHz. The
    60 GHz oxygen complex is far out of band, so the linear fit is adequate here
    and is flagged as an approximation, as required.

    Args:
        f_hz: Frequency in Hz.
        range_km: Range in km.

    Returns:
        Loss in dB.
    """
    f_ghz = np.asarray(f_hz, dtype=np.float64) / 1e9
    specific_db_km = 0.004 + 0.0015 * f_ghz
    return specific_db_km * np.asarray(range_km, dtype=np.float64)


def noise_power_dbm(bw_hz: np.ndarray | float, noise_figure_db: float) -> np.ndarray:
    """Thermal noise power in a given bandwidth.

    ``N = -174 dBm/Hz + NF + 10*log10(BW)``.

    Args:
        bw_hz: Noise bandwidth in Hz.
        noise_figure_db: Receiver noise figure in dB.

    Returns:
        Noise power in dBm.
    """
    return BOLTZMANN_DBM_HZ + noise_figure_db + 10.0 * np.log10(np.asarray(bw_hz, dtype=np.float64))


def link_budget_dbm(
    eirp_dbm: np.ndarray | float,
    gain_tx_db: np.ndarray | float,
    gain_rx_dbi: float,
    f_hz: np.ndarray | float,
    range_km: np.ndarray | float,
    misc_loss_db: float = 0.0,
) -> np.ndarray:
    """Received power at the receiver input.

    ``P_rx = EIRP + G_tx(theta) + G_rx - FSPL - L_atm - L_misc``.

    ``gain_tx_db`` is the emitter's *instantaneous* antenna gain toward us
    relative to its main lobe, i.e. 0 dB in the main beam and -30 dB or so in a
    sidelobe. Keeping it separate from EIRP is what lets one propagation path
    serve scanning and non-scanning emitters alike.

    Args:
        eirp_dbm: Emitter EIRP in the main lobe, dBm.
        gain_tx_db: Emitter antenna gain toward the receiver, dB relative to main lobe.
        gain_rx_dbi: Receiver antenna gain, dBi.
        f_hz: Frequency in Hz.
        range_km: Range in km.
        misc_loss_db: Polarisation, radome and implementation losses, dB.

    Returns:
        Received power in dBm.
    """
    return (
        np.asarray(eirp_dbm, dtype=np.float64)
        + np.asarray(gain_tx_db, dtype=np.float64)
        + gain_rx_dbi
        - free_space_path_loss_db(f_hz, range_km)
        - atmospheric_loss_db(f_hz, range_km)
        - misc_loss_db
    )


# --------------------------------------------------------------------------- #
# Detection statistics
# --------------------------------------------------------------------------- #
def detection_threshold(pfa: float, n_integrate: int) -> float:
    """Normalised detector threshold for a required false-alarm probability.

    Under H0 the normalised statistic ``W = Z / (2*sigma^2)`` is ``Gamma(N, 1)``,
    so ``T = gammaincinv(N, 1 - Pfa)``.

    Args:
        pfa: Required probability of false alarm, in ``(0, 1)``.
        n_integrate: Number of non-coherently integrated samples ``N >= 1``.

    Returns:
        Threshold in normalised (Erlang) units.
    """
    n = max(n_integrate, 1)
    if n > _N_EXACT_MAX:
        # Gaussian limit of Gamma(N, 1): mean N, variance N.
        return float(n + np.sqrt(n) * stats.norm.isf(pfa))
    return float(special.gammaincinv(n, 1.0 - pfa))


def _pd_swerling0(snr_lin: np.ndarray, n: int, thresh: float) -> np.ndarray:
    """Exact non-fluctuating detection probability via the non-central chi-square."""
    if n > _N_EXACT_MAX:
        # Moment-matched Gaussian: mean N(1+chi), variance N(1+2chi).
        mean = n * (1.0 + snr_lin)
        var = n * (1.0 + 2.0 * snr_lin)
        return stats.norm.sf((thresh - mean) / np.sqrt(var))
    return stats.ncx2.sf(2.0 * thresh, df=2 * n, nc=2.0 * n * snr_lin)


def _pd_swerling1(snr_lin: np.ndarray, n: int, thresh: float, pfa: float) -> np.ndarray:
    """Swerling I detection probability (Rayleigh amplitude, constant over the dwell).

    ``N = 1`` reduces to the familiar ``Pd = Pfa ** (1 / (1 + chi))``. For
    ``N > 1`` we use the standard closed form, evaluated in log space so the
    ``(1 + 1/(N*chi)) ** (N-1)`` factor cannot overflow.
    """
    chi = np.maximum(snr_lin, 1e-300)
    if n == 1:
        return np.power(pfa, 1.0 / (1.0 + chi))
    if n > _N_EXACT_MAX:
        # For large N the conditional statistic concentrates, so detection is
        # governed by the Rayleigh amplitude draw alone: exp(-(T/N - 1)/chi).
        # That asymptote ignores the residual noise fluctuation, so it tends to
        # ZERO rather than to Pfa as the signal vanishes. Combining the two
        # mutually exclusive routes to a threshold crossing -- signal-driven, or
        # noise alone -- restores the correct Pfa floor.
        signal = np.exp(-np.maximum(thresh / n - 1.0, 0.0) / chi)
        return np.clip(pfa + (1.0 - pfa) * signal, 0.0, 1.0)
    # Evaluated entirely in log space. The naive product is an inf * 0
    # indeterminate form as chi -> 0: the (1 + 1/(N*chi))**(N-1) factor
    # overflows float64 while the incomplete gamma underflows to exactly zero,
    # yielding NaN precisely at the SNR floor where Pd should simply tend to Pfa.
    n_chi = n * chi
    scale = 1.0 + 1.0 / n_chi
    z = thresh / scale
    gi = special.gammainc(n - 1, z)
    # Series form P(a, x) ~ x**a * exp(-x) / Gamma(a+1) rescues log(gi) wherever
    # gi has underflowed.
    log_gi = np.where(
        gi > 1e-300,
        np.log(np.maximum(gi, 1e-320)),
        (n - 1) * np.log(np.maximum(z, 1e-320)) - z - special.gammaln(n),
    )
    log_term = (n - 1) * np.log1p(1.0 / n_chi) + log_gi - thresh / (1.0 + n_chi)
    term = np.exp(np.minimum(log_term, 0.0))
    return np.clip(1.0 - special.gammainc(n - 1, thresh) + term, 0.0, 1.0)


def p_detect(
    snr_db: np.ndarray | float,
    integration_time_s: float | None = None,
    *,
    n_integrate: int | None = None,
    pfa: float = 1e-4,
    swerling: int = 1,
    bandwidth_hz: float | None = None,
) -> np.ndarray:
    """Probability of detection for a square-law envelope detector.

    Either give ``n_integrate`` directly, or give ``integration_time_s`` together
    with ``bandwidth_hz`` and the sample count is taken as their product.

    Args:
        snr_db: Post-processing SNR in dB (scalar or array).
        integration_time_s: Dwell time in seconds. Used with ``bandwidth_hz`` to
            derive ``N`` when ``n_integrate`` is not given.
        n_integrate: Explicit number of non-coherently integrated samples.
        pfa: Required probability of false alarm.
        swerling: ``0`` for non-fluctuating (exact non-central chi-square) or
            ``1`` for Rayleigh scan-to-scan fluctuation (closed form).
        bandwidth_hz: Detection noise bandwidth, used only to derive ``N``.

    Returns:
        Detection probability, same shape as ``snr_db``, clipped to ``[0, 1]``.

    Raises:
        ValueError: If neither ``n_integrate`` nor
            ``integration_time_s`` + ``bandwidth_hz`` is supplied, or if
            ``swerling`` is not 0 or 1.
    """
    if n_integrate is None:
        if integration_time_s is None or bandwidth_hz is None:
            raise ValueError("supply n_integrate, or integration_time_s together with bandwidth_hz")
        n_integrate = max(int(round(integration_time_s * bandwidth_hz)), 1)
    n = max(n_integrate, 1)
    thresh = detection_threshold(pfa, n)
    snr_lin = np.power(10.0, np.asarray(snr_db, dtype=np.float64) / 10.0)

    if swerling == 0:
        pd = _pd_swerling0(snr_lin, n, thresh)
    elif swerling == 1:
        pd = _pd_swerling1(snr_lin, n, thresh, pfa)
    else:
        raise ValueError(f"swerling must be 0 or 1, got {swerling}")
    return np.clip(pd, 0.0, 1.0)


def p_detect_dwell(
    snr_db: np.ndarray | float,
    n_pulses: np.ndarray | int,
    *,
    pfa: float = 1e-4,
    swerling: int = 1,
) -> np.ndarray:
    """Dwell-level detection probability for pulsed emitters.

    Each pulse is an independent single-sample detection opportunity, so the
    dwell succeeds if *any* pulse is detected. With per-pulse ``Pd``:

    ``Pd_dwell = 1 - (1 - Pd_pulse) ** n_pulses``

    This is the ``pulse`` regime described in the module docstring. It is a
    deliberate simplification of true m-of-n binary integration (which would
    trade a little Pd for a large Pfa reduction); 1-of-n is the correct model for
    a receiver that declares an intercept on a single pulse, which is what an ES
    system does when building a first cut of the emitter picture.

    Args:
        snr_db: Peak single-pulse SNR in dB.
        n_pulses: Pulses landing inside the dwell.
        pfa: Per-pulse false-alarm probability.
        swerling: Fluctuation model, 0 or 1.

    Returns:
        Dwell detection probability, clipped to ``[0, 1]``.
    """
    pd_single = p_detect(snr_db, n_integrate=1, pfa=pfa, swerling=swerling)
    n = np.maximum(np.asarray(n_pulses, dtype=np.float64), 0.0)
    return np.clip(1.0 - np.power(1.0 - pd_single, n), 0.0, 1.0)


def albersheim_snr_db(pd: float, pfa: float, n_integrate: int = 1) -> float:
    """SNR required for a target ``Pd`` at a given ``Pfa`` (Albersheim's equation).

    An empirical fit for a non-fluctuating target with a linear detector, valid
    for ``1e-7 <= Pfa <= 1e-3``, ``0.1 <= Pd <= 0.9``, ``1 <= N <= 8096``. It is
    used **only** as an independent cross-check on :func:`p_detect` in the test
    suite; it never appears in the simulation hot path.

    Args:
        pd: Target probability of detection.
        pfa: Probability of false alarm.
        n_integrate: Number of integrated pulses.

    Returns:
        Required single-pulse SNR in dB.
    """
    a = np.log(0.62 / pfa)
    b = np.log(pd / (1.0 - pd))
    z = 6.2 + (4.54 / np.sqrt(n_integrate + 0.44))
    return float(-5.0 * np.log10(n_integrate) + z * np.log10(a + (0.12 * a * b) + (1.7 * b)))


def min_snr_for_pd(
    pd_target: float = 0.9,
    pfa: float = 1e-3,
    n_integrate: int = 1,
    swerling: int = 1,
    bracket_db: tuple[float, float] = (-40.0, 80.0),
) -> float:
    """Minimum SNR at which ``Pd >= pd_target`` -- the receiver **sensitivity**.

    This is figure of merit 3 in the problem statement. The detection curve is
    monotone in SNR, so a bracketed root find is exact to machine precision and
    needs no curve fitting.

    Args:
        pd_target: Required detection probability, default 0.9.
        pfa: Fixed false-alarm probability, default 1e-3.
        n_integrate: Number of integrated samples.
        swerling: Fluctuation model, 0 or 1.
        bracket_db: SNR bracket to search, in dB.

    Returns:
        Minimum SNR in dB, or ``inf`` if ``pd_target`` is unreachable in the
        bracket (which happens when ``Pfa`` is so small that the threshold cannot
        be beaten at any SNR in range).
    """

    def gap(snr_db: float) -> float:
        return float(p_detect(snr_db, n_integrate=n_integrate, pfa=pfa, swerling=swerling)) - pd_target

    lo, hi = bracket_db
    if gap(hi) < 0.0:
        return float("inf")
    if gap(lo) > 0.0:
        return lo
    return float(optimize.brentq(gap, lo, hi, xtol=1e-4))
