"""Scenario calibration: author in SNR space, back-solve the geometry.

A pure ``(EIRP, range)`` prior is physically honest but operationally
uncontrollable. Sampling EIRP over 30 dB and range over 23 dB gives a 53 dB
spread in received power, which means a *loud, close* scanning radar is
detectable through its **sidelobes** for half the episode. At that point it is
not a scan-on-scan target at all -- it is a continuous emitter, and the entire
problem the brief poses has been calibrated away.

So scenarios are authored in **SNR space**: the generator samples the main-lobe
SNR the receiver should see, and back-solves the range that produces it. The link
budget itself is untouched and still fully physical -- we are choosing where in
its domain the scenario sits, exactly as an exercise designer chooses engagement
geometry. Inverting free-space path loss for range is closed form::

    FSPL_allowed = EIRP + G_rx - L_misc - L_atm - (N + SNR_target)
    R_km = 10 ** ((FSPL_allowed - 32.44 - 20*log10(f_MHz)) / 20)

Atmospheric loss depends on the range we are solving for, so the solve is
iterated twice; over 0.5-18 GHz the correction is under 0.5 dB and converges
immediately.
"""

from __future__ import annotations

import numpy as np

from smartscan.env.propagation import atmospheric_loss_db, noise_power_dbm

__all__ = ["detection_bandwidth_hz", "range_for_snr_km"]


def detection_bandwidth_hz(
    detection_mode: str, channel_bw_hz: float, pulse_width_s: float, fft_size: int
) -> float:
    """Noise bandwidth the detector actually sees.

    Args:
        detection_mode: ``"pulse"`` or ``"energy"``.
        channel_bw_hz: Channel width.
        pulse_width_s: Pulse width, used in the pulse regime.
        fft_size: Channeliser FFT length, used in the energy regime.

    Returns:
        Detection noise bandwidth in Hz.
    """
    if detection_mode == "pulse":
        # An ES receiver has no matched filter; video bandwidth ~ 1/PW.
        return float(min(channel_bw_hz, 1.0 / max(pulse_width_s, 1e-12)))
    return float(channel_bw_hz / max(fft_size, 1))


def range_for_snr_km(
    snr_target_db: float,
    eirp_dbm: float,
    f_hz: float,
    detection_bw_hz: float,
    noise_figure_db: float,
    antenna_gain_dbi: float,
    misc_loss_db: float,
    min_km: float = 1.0,
    max_km: float = 2000.0,
) -> float:
    """Range at which an emitter presents a given main-lobe SNR.

    Args:
        snr_target_db: Desired main-lobe SNR at the receiver, dB.
        eirp_dbm: Emitter main-lobe EIRP, dBm.
        f_hz: Carrier frequency, Hz.
        detection_bw_hz: Detector noise bandwidth, Hz.
        noise_figure_db: Receiver noise figure, dB.
        antenna_gain_dbi: Receiver antenna gain, dBi.
        misc_loss_db: Polarisation/radome/implementation losses, dB.
        min_km: Lower clamp on the solved range.
        max_km: Upper clamp on the solved range.

    Returns:
        Range in km, clamped to ``[min_km, max_km]``.
    """
    noise_dbm = float(noise_power_dbm(detection_bw_hz, noise_figure_db))
    p_rx_needed = noise_dbm + snr_target_db
    f_mhz = f_hz / 1e6

    r_km = 100.0
    for _ in range(2):  # atmospheric loss depends on the range being solved for
        l_atm = float(atmospheric_loss_db(f_hz, r_km))
        fspl_allowed = eirp_dbm + antenna_gain_dbi - misc_loss_db - l_atm - p_rx_needed
        r_km = float(10.0 ** ((fspl_allowed - 32.44 - 20.0 * np.log10(f_mhz)) / 20.0))
        r_km = float(np.clip(r_km, min_km, max_km))
    return r_km
