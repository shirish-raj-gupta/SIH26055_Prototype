"""Receiver hardware abstraction layer.

Everything above this line consumes :class:`~smartscan.env.types.Observation`
and knows nothing about how detections were produced. Everything below it deals
in Hz and seconds. Swapping :class:`~smartscan.hal.simulated.SimulatedBackend`
for a real SDR therefore requires **no change to any scheduler**, which is the
whole point (``docs/architecture.md`` §9, ``docs/hardware_roadmap.md``).

The interface is deliberately the smallest one a real radio can honour:

* :meth:`ReceiverBackend.tune` -- set centre frequency, blocking until settled.
* :meth:`ReceiverBackend.capture` -- acquire for a duration, return a handle.
* :meth:`ReceiverBackend.get_detections` -- run the detection chain on a capture.

A real backend adds a CFAR stage where the simulator has an analytic ``Pd``; the
contract they share is *a list of* :class:`~smartscan.env.types.Detection`
*records in physical units*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from smartscan.env.types import CaptureHandle, Detection

__all__ = ["ReceiverBackend"]


class ReceiverBackend(ABC):
    """Abstract wideband receiver.

    Implementations must be safe to call in the order
    ``tune -> capture -> get_detections`` repeatedly, and must report their own
    settling time honestly: the scheduler's cost model depends on it.
    """

    # -- capability reporting --------------------------------------------- #
    @property
    @abstractmethod
    def ibw_hz(self) -> float:
        """Instantaneous bandwidth of one capture, in Hz."""

    @property
    @abstractmethod
    def tune_range_hz(self) -> tuple[float, float]:
        """Inclusive ``(min, max)`` tunable centre frequency, in Hz."""

    @property
    @abstractmethod
    def settle_time_s(self) -> float:
        """Time lost to LO settling on every retune, in seconds.

        This is measured per device, not assumed: an RTL-SDR settles in a few
        hundred microseconds, a synthesiser-limited USRP can take milliseconds,
        and the scheduler's retune cost is only meaningful if this is real.
        """

    @property
    @abstractmethod
    def noise_figure_db(self) -> float:
        """Receiver noise figure in dB."""

    # -- operation --------------------------------------------------------- #
    @abstractmethod
    def tune(self, center_hz: float) -> None:
        """Retune the receiver.

        Args:
            center_hz: Requested centre frequency in Hz.

        Raises:
            ValueError: If ``center_hz`` is outside :attr:`tune_range_hz`.
        """

    @abstractmethod
    def capture(self, duration_s: float) -> CaptureHandle:
        """Acquire for ``duration_s`` seconds at the current centre frequency.

        Args:
            duration_s: Dwell duration in seconds.

        Returns:
            An opaque handle consumed by :meth:`get_detections`.
        """

    @abstractmethod
    def get_detections(self, capture: CaptureHandle) -> list[Detection]:
        """Run the detection chain over a capture.

        Args:
            capture: Handle returned by :meth:`capture`.

        Returns:
            Declared detections in physical units. False alarms are included and
            are **not** distinguishable from true detections -- that is the whole
            difficulty of the problem.
        """

    def close(self) -> None:
        """Release any hardware resources. No-op by default."""
        return None

    def __enter__(self) -> ReceiverBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
