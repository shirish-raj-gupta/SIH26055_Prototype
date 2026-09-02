"""SoapySDR backend skeleton -- **non-functional by design**.

This file exists to make the hardware path concrete and reviewable. It carries
the *identical* signature to :class:`~smartscan.hal.simulated.SimulatedBackend`,
so dropping in a real radio requires **no change to any scheduler, belief, or
evaluation code** -- only ``receiver.backend: soapy`` in the config.

Every method raises :class:`NotImplementedError` with the exact call it would
make. Nothing here silently returns plausible-looking fake data: a stub that
pretends to work is worse than one that refuses to.

Supported devices, once implemented (see ``docs/hardware_roadmap.md``):

======================  ========  ==========  ==========  ==============
Device                  IBW       Tune range  Settle      Indicative cost
======================  ========  ==========  ==========  ==============
RTL-SDR v4              2.4 MHz   24-1766 MHz ~0.3 ms     very low
HackRF One              20 MHz    1 MHz-6 GHz ~1 ms       low
ADALM-Pluto (retuned)   20 MHz    70 MHz-6 GHz ~0.5 ms    low
USRP B210               56 MHz    70 MHz-6 GHz ~0.2 ms    medium
======================  ========  ==========  ==========  ==============

None reaches 18 GHz directly; a block downconverter is required for the upper
band, and its LO settling time dominates ``t_settle``. That is a real constraint
on the deployed system and is stated rather than glossed over.
"""

from __future__ import annotations

from typing import Any

from smartscan.env.types import CaptureHandle, Detection
from smartscan.hal.backend import ReceiverBackend

__all__ = ["SoapySDRBackend"]

_MSG = (
    "SoapySDRBackend is a non-functional stub. Implement the TODO(hardware) "
    "blocks in smartscan/hal/soapy_stub.py and install SoapySDR + a device "
    "driver. See docs/hardware_roadmap.md."
)


class SoapySDRBackend(ReceiverBackend):
    """Real-SDR backend via SoapySDR. Not implemented.

    Args:
        device_args: SoapySDR device key/value pairs, e.g. ``{"driver": "hackrf"}``.
        sample_rate_hz: Requested sample rate; sets the usable IBW.
        gain_db: Front-end gain.
        fft_size: Channeliser FFT length for the detection chain.
        pfa: Design false-alarm probability for the CFAR stage.
    """

    def __init__(
        self,
        device_args: dict[str, str] | None = None,
        sample_rate_hz: float = 20e6,
        gain_db: float = 40.0,
        fft_size: int = 256,
        pfa: float = 1e-4,
    ) -> None:
        self.device_args = device_args or {"driver": "hackrf"}
        self.sample_rate_hz = sample_rate_hz
        self.gain_db = gain_db
        self.fft_size = fft_size
        self.pfa = pfa
        self._device: Any = None
        self._stream: Any = None
        # TODO(hardware): construct the device and stream.
        #   import SoapySDR
        #   from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
        #   self._device = SoapySDR.Device(self.device_args)
        #   self._device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate_hz)
        #   self._device.setGain(SOAPY_SDR_RX, 0, self.gain_db)
        #   self._stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        #   self._device.activateStream(self._stream)
        raise NotImplementedError(_MSG)

    # -- capabilities ------------------------------------------------------ #
    @property
    def ibw_hz(self) -> float:
        """Usable instantaneous bandwidth.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware): ~0.8 * sample_rate_hz after anti-alias roll-off; measure it.
        raise NotImplementedError(_MSG)

    @property
    def tune_range_hz(self) -> tuple[float, float]:
        """Tunable centre-frequency range.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware): self._device.getFrequencyRange(SOAPY_SDR_RX, 0)
        raise NotImplementedError(_MSG)

    @property
    def settle_time_s(self) -> float:
        """Measured LO settling time.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware): MEASURE this, do not assume. Tune between two widely
        # separated frequencies with a CW source present and time how long the
        # magnitude takes to stabilise. It sets the scheduler's retune cost, and
        # a wrong value invalidates every scheduling comparison.
        raise NotImplementedError(_MSG)

    @property
    def noise_figure_db(self) -> float:
        """Measured noise figure.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware): Y-factor measurement with a calibrated noise source.
        raise NotImplementedError(_MSG)

    # -- operation --------------------------------------------------------- #
    def tune(self, center_hz: float) -> None:
        """Retune the radio.

        Args:
            center_hz: Requested centre frequency in Hz.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware):
        #   self._device.setFrequency(SOAPY_SDR_RX, 0, center_hz)
        #   then block for settle_time_s (or poll the LO lock sensor if present:
        #   self._device.readSensor("lo_locked")).
        raise NotImplementedError(_MSG)

    def capture(self, duration_s: float) -> CaptureHandle:
        """Acquire IQ for a dwell.

        Args:
            duration_s: Dwell duration in seconds.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware):
        #   n = int(duration_s * self.sample_rate_hz)
        #   buf = np.empty(n, np.complex64)
        #   self._device.readStream(self._stream, [buf], n, timeoutUs=int(2e6 * duration_s))
        #   return CaptureHandle(..., payload=buf)
        raise NotImplementedError(_MSG)

    def get_detections(self, capture: CaptureHandle) -> list[Detection]:
        """Run the real detection chain over an IQ capture.

        This is where the simulator's analytic ``Pd`` is replaced by actual
        signal processing:

        1. Welch PSD over ``fft_size`` bins (Hann window, 50 % overlap).
        2. Per-bin **CA-CFAR** with guard cells, threshold set from
           :attr:`pfa` -- the same ``Pfa`` the simulator was calibrated to, so
           the scheduling comparison remains apples-to-apples.
        3. Cluster adjacent bins above threshold into single detections.
        4. Convert bin index to absolute Hz using the tuned centre frequency.

        Args:
            capture: Handle from :meth:`capture`.

        Raises:
            NotImplementedError: Always.
        """
        # TODO(hardware): implement steps 1-4 above.
        raise NotImplementedError(_MSG)

    def close(self) -> None:
        """Release the stream and device."""
        # TODO(hardware):
        #   self._device.deactivateStream(self._stream)
        #   self._device.closeStream(self._stream)
        return None
