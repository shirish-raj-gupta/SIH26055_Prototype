"""The eight emitter classes.

Every emitter implements ``activity(n_slots) -> Activity``, **vectorised over the
whole time axis**. That single design choice is what keeps the simulator fast
enough for CPU RL training: a per-slot Python loop over 30 emitters would be
3e5 calls per episode (``docs/architecture.md`` §15).

An :class:`~smartscan.env.types.Activity` is a sparse triplet
``(slots, channels, duty, gain_db, n_pulses)``. It is sparse rather than a dense
``(B, T)`` block because a frequency-agile emitter hopping at 10 kHz visits ten
different channels inside a single 1 ms slot -- something a dense per-slot
representation cannot express at all.

Antenna model
-------------
Scanning emitters use the ITU-R parabolic main-lobe approximation,
``G(theta) = -12 * (theta / theta_3dB) ** 2`` dB, floored at the configured
sidelobe level (and at the backlobe level beyond +-90 deg). The receiver is the
angular origin: ``theta`` is the beam's offset from *us*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from smartscan.env.types import Activity, DetectionMode, EmitterTruth

__all__ = [
    "EMITTER_REGISTRY",
    "AgileBeamRadar",
    "BaseEmitter",
    "CircularScanRadar",
    "CommsBurst",
    "FixedCW",
    "FrequencyAgile",
    "Interferer",
    "PulsedRadar",
    "SectorScanRadar",
    "beam_gain_db",
]


def beam_gain_db(
    offset_deg: np.ndarray,
    beamwidth_deg: float,
    sidelobe_db: float,
    backlobe_db: float = -45.0,
) -> np.ndarray:
    """Antenna gain toward the receiver as a function of beam pointing offset.

    Uses the ITU-R parabolic main-lobe approximation
    ``G(theta) = -12 * (theta / theta_3dB) ** 2`` dB, floored at ``sidelobe_db``
    within +-90 deg of boresight and at ``backlobe_db`` beyond it. Cited as an
    approximation: it is accurate to a few dB inside the main lobe and first
    sidelobe, which is the region that determines intercept opportunity.

    Args:
        offset_deg: Beam offset from the receiver, degrees; any shape.
        beamwidth_deg: 3 dB beamwidth, degrees.
        sidelobe_db: Sidelobe floor relative to main-lobe peak, dB (negative).
        backlobe_db: Backlobe floor relative to main-lobe peak, dB (negative).

    Returns:
        Gain in dB relative to the main-lobe peak, same shape as ``offset_deg``.
    """
    theta = np.abs(((np.asarray(offset_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0)
    main = -12.0 * np.square(theta / max(beamwidth_deg, 1e-6))
    floor = np.where(theta <= 90.0, sidelobe_db, backlobe_db)
    return np.maximum(main, floor)


def _pulse_train(
    n_slots: int,
    dt_s: float,
    pri_s: float,
    pulse_width_s: float,
    *,
    mode: str = "constant",
    jitter_frac: float = 0.0,
    stagger_ratios: tuple[float, ...] = (1.0,),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a pulse train into per-slot pulse counts and duty fractions.

    Args:
        n_slots: Episode length ``T``.
        dt_s: Slot duration.
        pri_s: Nominal pulse repetition interval.
        pulse_width_s: Pulse width.
        mode: ``"constant"``, ``"jittered"`` or ``"stagger"``.
        jitter_frac: Fractional PRI jitter, used when ``mode == "jittered"``.
        stagger_ratios: PRI multipliers cycled through when ``mode == "stagger"``.
        rng: Generator for jitter draws; required when ``mode == "jittered"``.

    Returns:
        Tuple ``(n_pulses, duty)``, each shape ``(T,)``. ``n_pulses`` is int32
        pulse count per slot; ``duty`` is the float32 fraction of each slot
        filled with RF (``n_pulses * PW / dt``, clipped to 1).
    """
    horizon = n_slots * dt_s
    n_nominal = int(np.ceil(horizon / max(pri_s, 1e-12))) + 2

    if mode == "jittered" and rng is not None:
        intervals = pri_s * (1.0 + jitter_frac * rng.uniform(-1.0, 1.0, size=n_nominal))
    elif mode == "stagger":
        ratios = np.asarray(stagger_ratios, dtype=np.float64)
        intervals = pri_s * np.resize(ratios, n_nominal)
    else:
        intervals = np.full(n_nominal, pri_s, dtype=np.float64)

    times = np.cumsum(intervals) - intervals[0]
    times = times[times < horizon]
    slots = np.floor(times / dt_s).astype(np.int64)
    n_pulses = np.bincount(slots, minlength=n_slots)[:n_slots].astype(np.int32)
    duty = np.clip(n_pulses * (pulse_width_s / dt_s), 0.0, 1.0).astype(np.float32)
    return n_pulses, duty


def _dense_to_activity(
    channel: int,
    duty: np.ndarray,
    gain_db: np.ndarray,
    n_pulses: np.ndarray,
    t_first: int,
) -> Activity:
    """Pack per-slot dense arrays for a single channel into a sparse Activity.

    Slots before ``t_first`` are dropped (pop-up emitters are silent until then),
    as are slots with zero duty.
    """
    t = np.arange(duty.size, dtype=np.int32)
    keep = (t >= t_first) & (duty > 0.0)
    idx = np.flatnonzero(keep).astype(np.int32)
    return Activity(
        slots=idx,
        channels=np.full(idx.size, channel, dtype=np.int32),
        duty=duty[idx].astype(np.float32),
        gain_db=gain_db[idx].astype(np.float32) if gain_db.size == duty.size else np.zeros(idx.size, np.float32),
        n_pulses=n_pulses[idx].astype(np.int32),
    )


@dataclass
class BaseEmitter(ABC):
    """Common state and interface for every emitter.

    Attributes:
        emitter_id: Unique id, ``>= 1``.
        home_channel: Channel index the emitter nominally occupies.
        f_center_hz: Nominal centre frequency.
        eirp_dbm: Main-lobe EIRP.
        range_km: Slant range to the receiver.
        threat_priority: Operator value in ``[0, 1]``.
        misc_loss_db: Polarisation/radome/implementation losses.
        t_first_active: First slot at which the emitter may transmit.
        is_novel: Unknown to pre-mission intelligence.
        n_slots: Episode length.
        dt_s: Slot duration.
    """

    emitter_id: int
    home_channel: int
    f_center_hz: float
    eirp_dbm: float
    range_km: float
    threat_priority: float
    misc_loss_db: float
    t_first_active: int
    is_novel: bool
    n_slots: int
    dt_s: float

    #: Detection regime; see ``propagation`` module docstring.
    detection_mode: DetectionMode = "energy"
    #: Low-value, high-duty distractor.
    is_interferer: bool = False

    @abstractmethod
    def activity(self) -> Activity:
        """Return this emitter's sparse emission over the whole episode."""

    @property
    def scan_period_s(self) -> float:
        """Scan period in seconds, or ``nan`` if the emitter is not scan-periodic."""
        return float("nan")

    def params(self) -> dict[str, Any]:
        """Class-specific parameters, carried into the dataset manifest."""
        return {}

    def truth(self) -> EmitterTruth:
        """Build the ground-truth record for evaluation and the dataset manifest."""
        return EmitterTruth(
            emitter_id=self.emitter_id,
            emitter_class=type(self).__name__,
            f_center_hz=float(self.f_center_hz),
            home_channel=int(self.home_channel),
            threat_priority=float(self.threat_priority),
            is_novel=bool(self.is_novel),
            is_interferer=bool(self.is_interferer),
            t_first_active=int(self.t_first_active),
            detection_mode=self.detection_mode,
            scan_period_s=float(self.scan_period_s),
            params=self.params(),
        )


# --------------------------------------------------------------------------- #
# 1. FixedCW
# --------------------------------------------------------------------------- #
@dataclass
class FixedCW(BaseEmitter):
    """Continuous wave on one channel at static SNR.

    The floor case: any scheduler that cannot find this is broken.
    """

    detection_mode: DetectionMode = "energy"

    def activity(self) -> Activity:
        """Return continuous unity-duty emission from ``t_first_active`` onward."""
        duty = np.ones(self.n_slots, dtype=np.float32)
        return _dense_to_activity(
            self.home_channel, duty, np.zeros(self.n_slots, np.float32),
            np.zeros(self.n_slots, np.int32), self.t_first_active,
        )


# --------------------------------------------------------------------------- #
# 2. PulsedRadar
# --------------------------------------------------------------------------- #
@dataclass
class PulsedRadar(BaseEmitter):
    """Non-scanning pulsed radar: constant, jittered or staggered PRI.

    Attributes:
        pri_s: Nominal pulse repetition interval.
        pulse_width_s: Pulse width.
        pri_mode: ``"constant"``, ``"jittered"`` or ``"stagger"``.
        pri_jitter_frac: Fractional jitter for ``"jittered"``.
        stagger_ratios: PRI multipliers for ``"stagger"``.
    """

    pri_s: float = 1.0e-3
    pulse_width_s: float = 1.0e-6
    pri_mode: str = "constant"
    pri_jitter_frac: float = 0.05
    stagger_ratios: tuple[float, ...] = (1.0,)
    rng: np.random.Generator | None = None
    detection_mode: DetectionMode = "pulse"

    def activity(self) -> Activity:
        """Return the binned pulse train on the home channel."""
        n_pulses, duty = _pulse_train(
            self.n_slots, self.dt_s, self.pri_s, self.pulse_width_s,
            mode=self.pri_mode, jitter_frac=self.pri_jitter_frac,
            stagger_ratios=self.stagger_ratios, rng=self.rng,
        )
        return _dense_to_activity(
            self.home_channel, duty, np.zeros(self.n_slots, np.float32), n_pulses, self.t_first_active
        )

    def params(self) -> dict[str, Any]:
        """Return PRI/PW parameters for the manifest."""
        return {
            "pri_s": self.pri_s,
            "pulse_width_s": self.pulse_width_s,
            "pri_mode": self.pri_mode,
        }


# --------------------------------------------------------------------------- #
# 3. CircularScanRadar
# --------------------------------------------------------------------------- #
@dataclass
class CircularScanRadar(BaseEmitter):
    """Mechanically scanning radar; illuminates us once per revolution.

    Beam dwell is ``(beamwidth / 360) * scan_period``: for a 1 deg beam on a 4 s
    scan that is **11 ms out of every 4000 ms**. This is the needle the whole
    scan-on-scan module exists to find. Sidelobes are retained so that weak
    off-beam intercepts remain possible, as the problem statement requires.

    Attributes:
        scan_period_s_: Revolution period ``Ts``.
        beamwidth_deg: 3 dB beamwidth.
        sidelobe_db: Sidelobe floor, dB below main lobe.
        backlobe_db: Backlobe floor, dB below main lobe.
        scan_phase_frac: Initial beam phase in ``[0, 1)`` of a revolution.
        pri_s: Pulse repetition interval.
        pulse_width_s: Pulse width.
    """

    scan_period_s_: float = 4.0
    beamwidth_deg: float = 2.0
    sidelobe_db: float = -30.0
    backlobe_db: float = -45.0
    scan_phase_frac: float = 0.0
    pri_s: float = 1.0e-3
    pulse_width_s: float = 1.0e-6
    detection_mode: DetectionMode = "pulse"

    @property
    def scan_period_s(self) -> float:
        """Revolution period in seconds."""
        return float(self.scan_period_s_)

    def _gain(self) -> np.ndarray:
        """Per-slot antenna gain toward the receiver."""
        t_s = np.arange(self.n_slots, dtype=np.float64) * self.dt_s
        # Beam bearing relative to us; we sit at 0 deg by construction.
        offset = 360.0 * ((t_s / self.scan_period_s_ + self.scan_phase_frac) % 1.0)
        return beam_gain_db(offset, self.beamwidth_deg, self.sidelobe_db, self.backlobe_db)

    def activity(self) -> Activity:
        """Return the pulse train modulated by the rotating antenna pattern."""
        n_pulses, duty = _pulse_train(self.n_slots, self.dt_s, self.pri_s, self.pulse_width_s)
        return _dense_to_activity(self.home_channel, duty, self._gain(), n_pulses, self.t_first_active)

    def params(self) -> dict[str, Any]:
        """Return scan parameters for the manifest."""
        return {
            "scan_period_s": self.scan_period_s_,
            "beamwidth_deg": self.beamwidth_deg,
            "sidelobe_db": self.sidelobe_db,
            "scan_phase_frac": self.scan_phase_frac,
            "pri_s": self.pri_s,
            "beam_dwell_s": self.beamwidth_deg / 360.0 * self.scan_period_s_,
        }


# --------------------------------------------------------------------------- #
# 4. SectorScanRadar
# --------------------------------------------------------------------------- #
@dataclass
class SectorScanRadar(BaseEmitter):
    """Raster/sector scan, bidirectional, with turnaround dwell at each end.

    Deliberately *not* a clean sinusoid: within one frame we are illuminated
    twice (once outbound, once inbound) at unequal spacing, and the turnaround
    dwell breaks the symmetry further. A naive periodogram sees two competing
    periods, which is exactly the stress case the estimator comparison needs.

    Attributes:
        sector_deg: Angular extent of the swept sector.
        scan_rate_deg_s: Angular rate within the sector.
        beamwidth_deg: 3 dB beamwidth.
        turnaround_dwell_s: Dead time at each sector end.
        bidirectional: Sweep back as well as forth.
        sidelobe_db: Sidelobe floor.
        bearing_offset_deg: Where we sit within the sector, relative to centre.
    """

    sector_deg: float = 90.0
    scan_rate_deg_s: float = 60.0
    beamwidth_deg: float = 3.0
    turnaround_dwell_s: float = 0.1
    bidirectional: bool = True
    sidelobe_db: float = -28.0
    backlobe_db: float = -45.0
    bearing_offset_deg: float = 0.0
    pri_s: float = 1.0e-3
    pulse_width_s: float = 1.0e-6
    detection_mode: DetectionMode = "pulse"

    @property
    def _sweep_s(self) -> float:
        """Time for one end-to-end sweep of the sector."""
        return self.sector_deg / max(self.scan_rate_deg_s, 1e-9)

    @property
    def scan_period_s(self) -> float:
        """Full frame period: out, dwell, back, dwell (or out + dwell if unidirectional)."""
        if self.bidirectional:
            return 2.0 * self._sweep_s + 2.0 * self.turnaround_dwell_s
        return self._sweep_s + self.turnaround_dwell_s

    def _gain(self) -> np.ndarray:
        """Per-slot antenna gain toward the receiver."""
        t_s = np.arange(self.n_slots, dtype=np.float64) * self.dt_s
        frame = self.scan_period_s
        phase = np.mod(t_s, frame)
        sweep, dwell = self._sweep_s, self.turnaround_dwell_s
        half = self.sector_deg / 2.0

        # Piecewise beam bearing across one frame.
        angle = np.full_like(phase, -half)
        seg1 = phase < sweep  # outbound
        angle[seg1] = -half + self.scan_rate_deg_s * phase[seg1]
        if self.bidirectional:
            seg2 = (phase >= sweep) & (phase < sweep + dwell)  # park at +half
            angle[seg2] = half
            seg3 = (phase >= sweep + dwell) & (phase < 2 * sweep + dwell)  # inbound
            angle[seg3] = half - self.scan_rate_deg_s * (phase[seg3] - sweep - dwell)
            seg4 = phase >= 2 * sweep + dwell  # park at -half
            angle[seg4] = -half
        else:
            angle[phase >= sweep] = -half

        return beam_gain_db(
            angle - self.bearing_offset_deg, self.beamwidth_deg, self.sidelobe_db, self.backlobe_db
        )

    def activity(self) -> Activity:
        """Return the pulse train modulated by the raster antenna pattern."""
        n_pulses, duty = _pulse_train(self.n_slots, self.dt_s, self.pri_s, self.pulse_width_s)
        return _dense_to_activity(self.home_channel, duty, self._gain(), n_pulses, self.t_first_active)

    def params(self) -> dict[str, Any]:
        """Return sector-scan parameters for the manifest."""
        return {
            "sector_deg": self.sector_deg,
            "scan_rate_deg_s": self.scan_rate_deg_s,
            "beamwidth_deg": self.beamwidth_deg,
            "turnaround_dwell_s": self.turnaround_dwell_s,
            "bidirectional": int(self.bidirectional),
            "frame_period_s": self.scan_period_s,
        }


# --------------------------------------------------------------------------- #
# 5. FrequencyAgile
# --------------------------------------------------------------------------- #
@dataclass
class FrequencyAgile(BaseEmitter):
    """Pseudo-random frequency hopper over a hop set.

    At hop rates above ``1 / dt`` the emitter changes channel **inside** a single
    slot -- up to ten times at 10 kHz with a 1 ms slot. The sub-slot geometry is
    resolved exactly by intersecting the hop interval grid with the slot grid
    (a union of breakpoints), so a slot can legitimately carry fractional duty in
    several channels at once. No dense per-slot representation can express that,
    which is why :class:`Activity` is a sparse triplet.

    Attributes:
        hop_set: Channel indices the emitter hops over.
        hop_rate_hz: Hops per second.
        hop_sequence: Index into ``hop_set`` for each hop.
        pri_s: Pulse repetition interval while dwelling on a channel.
        pulse_width_s: Pulse width.
    """

    hop_set: tuple[int, ...] = ()
    hop_rate_hz: float = 1000.0
    hop_sequence: np.ndarray | None = None
    pri_s: float = 5.0e-4
    pulse_width_s: float = 1.0e-6
    detection_mode: DetectionMode = "pulse"

    def activity(self) -> Activity:
        """Return per-(slot, channel) duty resolved at sub-slot precision."""
        dt, horizon = self.dt_s, self.n_slots * self.dt_s
        hop_dur = 1.0 / self.hop_rate_hz
        n_hops = int(np.ceil(horizon / hop_dur))
        hop_edges = np.arange(n_hops + 1, dtype=np.float64) * hop_dur
        slot_edges = np.arange(self.n_slots + 1, dtype=np.float64) * dt

        # Breakpoint union: every segment lies inside exactly one slot and one hop.
        bounds = np.union1d(slot_edges, hop_edges[hop_edges <= horizon])
        bounds = np.clip(bounds, 0.0, horizon)
        seg_dur = np.diff(bounds)
        keep = seg_dur > 1e-15
        mid = (0.5 * (bounds[:-1] + bounds[1:]))[keep]
        seg_dur = seg_dur[keep]

        slot_idx = np.minimum((mid / dt).astype(np.int64), self.n_slots - 1)
        hop_idx = np.minimum((mid / hop_dur).astype(np.int64), n_hops - 1)
        seq = self.hop_sequence if self.hop_sequence is not None else np.zeros(n_hops, np.int64)
        chan = np.asarray(self.hop_set, dtype=np.int32)[seq[hop_idx] % len(self.hop_set)]

        active = slot_idx >= self.t_first_active
        slot_idx, chan, seg_dur = slot_idx[active], chan[active], seg_dur[active]

        # Aggregate duplicate (slot, channel) pairs -- a slot can revisit a channel.
        n_ch = int(chan.max()) + 1 if chan.size else 1
        key = slot_idx * n_ch + chan
        uniq, inv = np.unique(key, return_inverse=True)
        occupied_s = np.bincount(inv, weights=seg_dur, minlength=uniq.size)
        out_slot = (uniq // n_ch).astype(np.int32)
        out_chan = (uniq % n_ch).astype(np.int32)

        # Pulse count scales with the time actually spent on that channel.
        n_pulses = np.floor(occupied_s / self.pri_s).astype(np.int32)
        duty = np.clip(n_pulses * self.pulse_width_s / dt, 0.0, 1.0).astype(np.float32)
        nonzero = duty > 0.0
        return Activity(
            slots=out_slot[nonzero],
            channels=out_chan[nonzero],
            duty=duty[nonzero],
            gain_db=np.zeros(int(nonzero.sum()), dtype=np.float32),
            n_pulses=n_pulses[nonzero],
        )

    def params(self) -> dict[str, Any]:
        """Return hop-set parameters for the manifest."""
        return {
            "hop_rate_hz": self.hop_rate_hz,
            "hop_set_size": len(self.hop_set),
            "hop_set": tuple(int(c) for c in self.hop_set),
            "pri_s": self.pri_s,
        }


# --------------------------------------------------------------------------- #
# 6. AgileBeamRadar
# --------------------------------------------------------------------------- #
@dataclass
class AgileBeamRadar(BaseEmitter):
    """Electronically steered AESA with non-deterministic revisit -- the hardest case.

    Looks arrive with Gamma-distributed gaps (shape ``k < 2`` gives a heavy tail),
    so there is **no** scan period to estimate. Period estimators must fail
    gracefully here and the scheduler must fall back to coverage; demonstrating
    that graceful failure is more valuable than pretending every emitter is
    periodic.

    Sidelobes are omitted between looks: an AESA pencil beam at -35 dB into a
    wideband ES receiver is effectively undetectable, and modelling it would add
    dense near-zero cells to every tensor for no informational gain.

    Attributes:
        look_times_s: Start time of each look at the receiver.
        look_duration_s: Duration of one look.
        pri_s: Pulse repetition interval during a look.
        pulse_width_s: Pulse width.
    """

    look_times_s: np.ndarray | None = None
    look_duration_s: float = 0.01
    pri_s: float = 5.0e-4
    pulse_width_s: float = 1.0e-6
    revisit_mean_s: float = 2.0
    detection_mode: DetectionMode = "pulse"

    def activity(self) -> Activity:
        """Return emission confined to the sampled look intervals."""
        looks = self.look_times_s if self.look_times_s is not None else np.zeros(0)
        gain = np.full(self.n_slots, -300.0, dtype=np.float32)
        illuminated = np.zeros(self.n_slots, dtype=bool)
        for t0 in looks:
            lo = int(np.floor(t0 / self.dt_s))
            hi = int(np.ceil((t0 + self.look_duration_s) / self.dt_s))
            lo, hi = max(lo, 0), min(hi, self.n_slots)
            if hi > lo:
                illuminated[lo:hi] = True
                gain[lo:hi] = 0.0

        n_pulses, duty = _pulse_train(self.n_slots, self.dt_s, self.pri_s, self.pulse_width_s)
        duty = np.where(illuminated, duty, 0.0).astype(np.float32)
        n_pulses = np.where(illuminated, n_pulses, 0).astype(np.int32)
        return _dense_to_activity(self.home_channel, duty, gain, n_pulses, self.t_first_active)

    def params(self) -> dict[str, Any]:
        """Return look-scheduling parameters for the manifest."""
        return {
            "revisit_mean_s": self.revisit_mean_s,
            "look_duration_s": self.look_duration_s,
            "n_looks": int(self.look_times_s.size) if self.look_times_s is not None else 0,
            "pri_s": self.pri_s,
        }


# --------------------------------------------------------------------------- #
# 7. CommsBurst
# --------------------------------------------------------------------------- #
@dataclass
class CommsBurst(BaseEmitter):
    """Push-to-talk comms: 2-state Markov ON/OFF with long-tailed dwell times.

    Dwell durations are lognormal (or Pareto) rather than geometric. The long OFF
    tail is the point: it punishes any scheduler whose belief collapses to
    "confirmed idle, never revisit", because a channel silent for 3 s may well
    burst at 3.1 s.

    Attributes:
        on_off_intervals: Alternating (start_slot, stop_slot) ON intervals.
    """

    on_intervals: np.ndarray | None = None
    mean_on_slots: float = 250.0
    mean_off_slots: float = 1250.0
    detection_mode: DetectionMode = "energy"

    def activity(self) -> Activity:
        """Return unity duty inside each ON interval."""
        duty = np.zeros(self.n_slots, dtype=np.float32)
        if self.on_intervals is not None:
            for lo, hi in self.on_intervals:
                duty[int(lo) : int(hi)] = 1.0
        return _dense_to_activity(
            self.home_channel, duty, np.zeros(self.n_slots, np.float32),
            np.zeros(self.n_slots, np.int32), self.t_first_active,
        )

    def params(self) -> dict[str, Any]:
        """Return Markov dwell parameters for the manifest."""
        return {
            "mean_on_slots": self.mean_on_slots,
            "mean_off_slots": self.mean_off_slots,
            "n_bursts": len(self.on_intervals) if self.on_intervals is not None else 0,
        }


# --------------------------------------------------------------------------- #
# 8. Interferer
# --------------------------------------------------------------------------- #
@dataclass
class Interferer(BaseEmitter):
    """High-duty, low-value emitter (broadcast, datalink) -- **the trap**.

    Loud, close and almost always on, so it is the single most tempting target
    for any policy maximising raw detection count -- while carrying a
    ``threat_priority`` near 0.02. Its whole purpose is to punish schedulers that
    confuse "easy to detect" with "worth detecting".
    """

    duty_frac: float = 0.9
    detection_mode: DetectionMode = "energy"
    is_interferer: bool = True

    def activity(self) -> Activity:
        """Return continuous emission at the configured duty fraction."""
        duty = np.full(self.n_slots, self.duty_frac, dtype=np.float32)
        return _dense_to_activity(
            self.home_channel, duty, np.zeros(self.n_slots, np.float32),
            np.zeros(self.n_slots, np.int32), self.t_first_active,
        )

    def params(self) -> dict[str, Any]:
        """Return duty parameters for the manifest."""
        return {"duty_frac": self.duty_frac}


#: Maps config class keys to implementations. The scenario generator uses this,
#: so adding an emitter class is a one-line registry change plus the class.
EMITTER_REGISTRY: dict[str, type[BaseEmitter]] = {
    "fixed_cw": FixedCW,
    "pulsed_radar": PulsedRadar,
    "circular_scan": CircularScanRadar,
    "sector_scan": SectorScanRadar,
    "frequency_agile": FrequencyAgile,
    "agile_beam": AgileBeamRadar,
    "comms_burst": CommsBurst,
    "interferer": Interferer,
}
