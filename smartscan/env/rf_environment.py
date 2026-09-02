"""Scenario generation and ground-truth tensor assembly.

The load-bearing performance decision lives here: ground truth is **precomputed
once per episode, vectorised over the whole time axis**, after which stepping the
environment is an ``O(K)`` array slice. A per-slot Python loop over emitters
would be 3e5 calls per episode and would make CPU RL training impossible
(``docs/architecture.md`` §15).

Emitter randomness is drawn from ``SeedTree("emitter", i)`` substreams, so adding
a 16th emitter does not perturb the first 15 and changing the scheduler does not
change the world it is measured in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smartscan.config import Config
from smartscan.env import emitters as em
from smartscan.env.calibration import detection_bandwidth_hz, range_for_snr_km
from smartscan.env.propagation import SNR_FLOOR_DB, link_budget_dbm, noise_power_dbm
from smartscan.env.types import EpisodeTensors, SpectrumGrid
from smartscan.seeding import SeedTree

__all__ = ["Scenario", "build_episode", "generate_scenario", "place_channels"]


@dataclass(frozen=True)
class Scenario:
    """A fully specified, reproducible episode definition.

    Attributes:
        emitters: The constructed emitter objects.
        grid: Frequency partition.
        seed: Scenario seed.
        config: The resolved configuration used.
    """

    emitters: tuple[em.BaseEmitter, ...]
    grid: SpectrumGrid
    seed: int
    config: Config


def place_channels(
    n: int,
    n_channels: int,
    rng: np.random.Generator,
    *,
    min_separation: int = 2,
    method: str = "poisson_disk",
) -> np.ndarray:
    """Assign home channels to ``n`` emitters.

    Poisson-disk (blue-noise) placement keeps emitters apart, so a scenario is
    not trivially solved by one lucky ``K``-channel window landing on half the
    order of battle. If the separation constraint cannot be satisfied it is
    relaxed progressively rather than failing -- a HARD scenario with 30 emitters
    in 64 channels genuinely cannot hold a separation of 2 everywhere.

    Args:
        n: Number of emitters to place.
        n_channels: Band size ``B``.
        rng: Generator for the draw.
        min_separation: Desired minimum channel gap between emitters.
        method: ``"poisson_disk"`` or ``"uniform"``.

    Returns:
        Int32 array of shape ``(n,)`` of channel indices.
    """
    if method == "uniform" or min_separation <= 0:
        return rng.integers(0, n_channels, size=n).astype(np.int32)

    for sep in range(min_separation, -1, -1):
        chosen: list[int] = []
        for cand in rng.permutation(n_channels):
            if all(abs(int(cand) - c) > sep for c in chosen):
                chosen.append(int(cand))
            if len(chosen) == n:
                break
        if len(chosen) == n:
            return np.asarray(chosen, dtype=np.int32)

    # Fewer channels than emitters: allow collisions, spread as evenly as possible.
    base = np.tile(np.arange(n_channels), int(np.ceil(n / n_channels)))[:n]
    return rng.permutation(base).astype(np.int32)


def _sample_link_budget(
    block: Any,
    defaults: Any,
    rng: np.random.Generator,
    *,
    cfg: Config | None = None,
    f_hz: float | None = None,
    channel_bw_hz: float | None = None,
    detection_mode: str = "energy",
    pulse_width_s: float = 1e-6,
) -> dict[str, float]:
    """Draw EIRP, range, threat priority and misc loss for one emitter.

    When the class carries an ``snr_db`` prior the range is **back-solved** from
    the link budget so the main-lobe SNR lands where the scenario author intends
    (:mod:`smartscan.env.calibration`). Otherwise ``range_km`` is sampled
    directly.
    """
    inh = block.inherited(defaults)
    eirp = float(inh["eirp_dbm"].sample(rng))
    misc = float(inh["misc_loss_db"])
    snr_prior = inh.get("snr_db")

    if snr_prior is not None and cfg is not None and f_hz is not None and channel_bw_hz is not None:
        bw = detection_bandwidth_hz(
            detection_mode, channel_bw_hz, pulse_width_s, cfg.receiver.detector.fft_size
        )
        range_km = range_for_snr_km(
            float(snr_prior.sample(rng)), eirp, f_hz, bw,
            cfg.receiver.noise_figure_db, cfg.receiver.antenna_gain_dbi, misc,
        )
    else:
        range_km = float(inh["range_km"].sample(rng))

    return {
        "eirp_dbm": eirp,
        "range_km": range_km,
        "threat_priority": float(np.clip(inh["threat_priority"].sample(rng), 0.0, 1.0)),
        "misc_loss_db": misc,
    }


def _burst_intervals(
    n_slots: int, mean_on: float, mean_off: float, sigma: float, dist: str, rng: np.random.Generator
) -> np.ndarray:
    """Sample alternating OFF/ON dwell times with long tails.

    Lognormal dwells (matched in mean to the configured Markov rates) give the
    heavy OFF tail the problem statement asks for: a channel silent for 3 s may
    still burst at 3.1 s, which punishes any belief that collapses to
    "confirmed idle, never revisit".

    Args:
        n_slots: Episode length.
        mean_on: Mean ON dwell in slots.
        mean_off: Mean OFF dwell in slots.
        sigma: Lognormal shape (or Pareto tail index driver).
        dist: ``"lognormal"`` or ``"pareto"``.
        rng: Generator.

    Returns:
        Int array of shape ``(n_bursts, 2)`` of ``[start, stop)`` ON intervals.
    """

    def draw(mean: float) -> float:
        if dist == "pareto":
            alpha = 1.0 + 1.0 / max(sigma, 1e-6)
            scale = mean * (alpha - 1.0) / alpha if alpha > 1.0 else mean
            return float(scale * (1.0 + rng.pareto(alpha)))
        mu = np.log(max(mean, 1e-9)) - 0.5 * sigma**2
        return float(rng.lognormal(mu, sigma))

    intervals: list[tuple[int, int]] = []
    t = draw(mean_off)
    while t < n_slots:
        on = max(draw(mean_on), 1.0)
        lo, hi = int(t), min(int(t + on), n_slots)
        if hi > lo:
            intervals.append((lo, hi))
        t = hi + max(draw(mean_off), 1.0)
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def _build_emitter(
    class_key: str, eid: int, channel: int, cfg: Config, grid: SpectrumGrid,
    rng: np.random.Generator, t_first: int,
) -> em.BaseEmitter:
    """Construct one emitter of ``class_key`` from its configured priors."""
    block = cfg.emitters.block(class_key)
    # Pulse width is drawn first: it sets the detection bandwidth against which
    # the SNR-space range solve is calibrated.
    pw = float(block.pulse_width_s.sample_log(rng)) if hasattr(block, "pulse_width_s") else 1e-6
    common = dict(
        emitter_id=eid,
        home_channel=int(channel),
        f_center_hz=float(grid.centers_hz[channel]),
        t_first_active=int(t_first),
        is_novel=True,
        n_slots=cfg.n_slots,
        dt_s=cfg.time.dt_s,
        **_sample_link_budget(
            block, cfg.emitters.defaults, rng,
            cfg=cfg, f_hz=float(grid.centers_hz[channel]),
            channel_bw_hz=float(grid.widths_hz[channel]),
            detection_mode=block.detection_mode, pulse_width_s=pw,
        ),
    )
    dt, n_slots = cfg.time.dt_s, cfg.n_slots

    if class_key == "fixed_cw":
        return em.FixedCW(**common)

    if class_key == "pulsed_radar":
        return em.PulsedRadar(
            **common,
            pri_s=float(block.pri_s.sample_log(rng)),
            pulse_width_s=pw,
            pri_mode=block.pri_mode,
            pri_jitter_frac=block.pri_jitter_frac,
            stagger_ratios=tuple(block.stagger_ratios),
            rng=rng,
        )

    if class_key == "circular_scan":
        return em.CircularScanRadar(
            **common,
            scan_period_s_=float(block.scan_period_s.sample(rng)),
            beamwidth_deg=float(block.beamwidth_deg.sample(rng)),
            sidelobe_db=float(block.sidelobe_db.sample(rng)),
            backlobe_db=block.backlobe_db,
            scan_phase_frac=float(block.scan_phase_frac.sample(rng)),
            pri_s=float(block.pri_s.sample_log(rng)),
            pulse_width_s=pw,
        )

    if class_key == "sector_scan":
        sector = float(block.sector_deg.sample(rng))
        return em.SectorScanRadar(
            **common,
            sector_deg=sector,
            scan_rate_deg_s=float(block.scan_rate_deg_s.sample(rng)),
            beamwidth_deg=float(block.beamwidth_deg.sample(rng)),
            turnaround_dwell_s=float(block.turnaround_dwell_s.sample(rng)),
            bidirectional=block.bidirectional,
            sidelobe_db=float(block.sidelobe_db.sample(rng)),
            # Place ourselves inside the swept sector, but off-centre, so the two
            # arrivals per frame are unequally spaced -- the estimator stress case.
            bearing_offset_deg=float(rng.uniform(-0.4, 0.4) * sector),
            pri_s=float(block.pri_s.sample_log(rng)),
            pulse_width_s=pw,
        )

    if class_key == "frequency_agile":
        size = int(np.clip(block.hop_set_size.sample_int(rng), 2, cfg.n_channels))
        if rng.random() < block.hop_set_contiguous_frac:
            start = int(rng.integers(0, max(cfg.n_channels - size, 1)))
            hop_set = tuple(range(start, start + size))
        else:
            hop_set = tuple(int(c) for c in rng.choice(cfg.n_channels, size=size, replace=False))
        hop_rate = float(block.hop_rate_hz.sample_log(rng))
        n_hops = int(np.ceil(n_slots * dt * hop_rate)) + 1
        if block.hop_pattern == "cyclic":
            seq = np.arange(n_hops) % size
        else:
            seq = rng.integers(0, size, size=n_hops)
        return em.FrequencyAgile(
            **common,
            hop_set=hop_set,
            hop_rate_hz=hop_rate,
            hop_sequence=seq,
            pri_s=float(block.pri_s.sample_log(rng)) if hasattr(block, "pri_s") else 5.0e-4,
            pulse_width_s=pw,
        )

    if class_key == "agile_beam":
        mean_gap = float(block.revisit_mean_s.sample(rng))
        k = block.revisit_shape_k
        horizon = n_slots * dt
        # Gamma(k, mean/k) inter-arrivals: k < 2 gives the heavy tail that makes
        # the revisit genuinely unpredictable rather than merely noisy.
        n_draw = int(np.ceil(horizon / mean_gap * 3.0)) + 8
        gaps = rng.gamma(k, mean_gap / k, size=n_draw)
        looks = np.cumsum(gaps)
        looks = looks[looks < horizon]
        return em.AgileBeamRadar(
            **common,
            look_times_s=looks,
            look_duration_s=float(block.look_duration_s.sample_log(rng)),
            pri_s=float(block.pri_s.sample_log(rng)),
            pulse_width_s=pw,
            revisit_mean_s=mean_gap,
        )

    if class_key == "comms_burst":
        mean_on = 1.0 / block.p_on_to_off
        mean_off = 1.0 / block.p_off_to_on
        return em.CommsBurst(
            **common,
            on_intervals=_burst_intervals(n_slots, mean_on, mean_off, block.dwell_sigma, block.dwell_dist, rng),
            mean_on_slots=mean_on,
            mean_off_slots=mean_off,
        )

    if class_key == "interferer":
        return em.Interferer(**common, duty_frac=float(block.duty_frac.sample(rng)))

    raise ValueError(f"unknown emitter class {class_key!r}")


def generate_scenario(
    seed: int,
    n_emitters: int | None = None,
    difficulty: str | None = None,
    config: Config | None = None,
) -> Scenario:
    """Build a reproducible episode definition.

    Args:
        seed: Scenario seed. The same seed and config always yield byte-identical
            tensors (acceptance test 1).
        n_emitters: Override the configured emitter count. The configured ``mix``
            is rescaled proportionally to match.
        difficulty: Override the configured tier label (record-keeping only; the
            mix comes from the config).
        config: Resolved configuration. Defaults to the built-in defaults.

    Returns:
        A :class:`Scenario`.
    """
    cfg = config or Config()
    if n_emitters is not None and n_emitters != cfg.scenario.n_emitters:
        cfg = _rescale_mix(cfg, n_emitters)
    if difficulty is not None and difficulty != cfg.scenario.difficulty:
        cfg = cfg.with_overrides(scenario={"difficulty": difficulty})

    tree = SeedTree(seed)
    rng = tree.rng("scenario")
    grid = cfg.grid()
    sc = cfg.scenario

    class_keys = [k for k, count in sc.mix.items() for _ in range(count)]
    rng.shuffle(class_keys)
    channels = place_channels(
        len(class_keys), cfg.n_channels, rng,
        min_separation=sc.min_channel_separation, method=sc.placement,
    )

    # Pop-ups: silent until t > popup_start_frac * T. Chosen from the tail of the
    # shuffled list, which is already class-random, so pop-ups are not biased
    # toward any one emitter type.
    t_first = np.zeros(len(class_keys), dtype=np.int64)
    if sc.n_popup > 0:
        popup_idx = rng.choice(len(class_keys), size=min(sc.n_popup, len(class_keys)), replace=False)
        floor = int(sc.popup_start_frac * cfg.n_slots)
        span = max(cfg.n_slots - floor - 1, 1)
        t_first[popup_idx] = floor + rng.integers(0, span, size=popup_idx.size)

    built = tuple(
        _build_emitter(key, eid + 1, int(channels[eid]), cfg, grid, tree.rng("emitter", eid), int(t_first[eid]))
        for eid, key in enumerate(class_keys)
    )
    return Scenario(emitters=built, grid=grid, seed=int(seed), config=cfg)


def _rescale_mix(cfg: Config, n_emitters: int) -> Config:
    """Rescale the configured mix proportionally to a new emitter count."""
    mix = cfg.scenario.mix
    total = sum(mix.values()) or 1
    scaled = {k: int(round(v * n_emitters / total)) for k, v in mix.items()}
    drift = n_emitters - sum(scaled.values())
    if drift:  # Repair rounding drift on the largest bucket.
        key = max(scaled, key=lambda k: scaled[k]) if scaled else next(iter(mix))
        scaled[key] = max(scaled[key] + drift, 0)
    return cfg.with_overrides(scenario={"n_emitters": n_emitters, "mix": scaled})


def _emitter_snr_db(
    e: em.BaseEmitter, act: em.Activity, cfg: Config, grid: SpectrumGrid
) -> np.ndarray:
    """Post-processing SNR for every entry of one emitter's activity.

    Detection bandwidth depends on the emitter's regime, not the receiver's
    settings (``propagation`` module docstring):

    * ``pulse``: an ES receiver has no matched filter, so it detects on a video
      bandwidth of roughly ``1 / PW``, capped at the channel width.
    * ``energy``: a narrowband signal lands in one FFT bin, giving
      ``10*log10(fft_size)`` dB of processing gain against a bin-width noise
      floor, scaled by the sub-slot duty actually present.

    Args:
        e: The emitter.
        act: Its activity triplet.
        cfg: Resolved configuration.
        grid: Frequency partition.

    Returns:
        Float64 SNR in dB, shape ``(len(act),)``.
    """
    det = cfg.receiver.detector
    ch_bw = grid.widths_hz[act.channels]
    p_rx = link_budget_dbm(
        e.eirp_dbm, act.gain_db, cfg.receiver.antenna_gain_dbi,
        grid.centers_hz[act.channels], e.range_km, e.misc_loss_db,
    )

    if cfg.receiver.straddle_enabled:
        # Scalloping: a tone at a channel edge loses up to straddle_loss_db.
        frac = grid.straddle_fraction(e.f_center_hz)
        p_rx = p_rx - cfg.receiver.straddle_loss_db * (1.0 - float(frac[0]))

    if e.detection_mode == "pulse":
        pw = float(getattr(e, "pulse_width_s", 1e-6))
        bw_det = np.minimum(ch_bw, 1.0 / max(pw, 1e-12))
        snr = p_rx - noise_power_dbm(bw_det, cfg.receiver.noise_figure_db)
    else:
        bw_bin = ch_bw / det.fft_size
        duty = np.maximum(act.duty.astype(np.float64), 1e-6)
        snr = p_rx - noise_power_dbm(bw_bin, cfg.receiver.noise_figure_db) + 10.0 * np.log10(duty)

    # Front-end floor: below tangential sensitivity the receiver cannot register
    # the signal at all, whatever the processing gain downstream.
    return np.where(p_rx >= cfg.receiver.sensitivity_dbm, snr, SNR_FLOOR_DB)


def build_episode(scenario: Scenario) -> EpisodeTensors:
    """Assemble the ground-truth tensors for a scenario.

    Where several emitters share a cell the strongest wins the ``SNR``,
    ``emitter_id`` and ``n_pulses`` labels, while ``duty`` accumulates (clipped
    to 1) because two emitters genuinely do fill more of the slot than one.

    Args:
        scenario: The episode definition.

    Returns:
        The populated :class:`EpisodeTensors`.
    """
    cfg, grid = scenario.config, scenario.grid
    b, t = cfg.n_channels, cfg.n_slots

    duty = np.zeros((b, t), dtype=np.float32)
    snr = np.full((b, t), SNR_FLOOR_DB, dtype=np.float32)
    eid = np.zeros((b, t), dtype=np.int16)
    npulse = np.zeros((b, t), dtype=np.int32)

    for e in scenario.emitters:
        act = e.activity()
        if len(act) == 0:
            continue
        snr_e = _emitter_snr_db(e, act, cfg, grid).astype(np.float32)
        ch, sl = act.channels, act.slots

        np.add.at(duty, (ch, sl), act.duty)
        stronger = snr_e > snr[ch, sl]
        if np.any(stronger):
            cs, ss = ch[stronger], sl[stronger]
            snr[cs, ss] = snr_e[stronger]
            eid[cs, ss] = e.emitter_id
            npulse[cs, ss] = act.n_pulses[stronger]

    np.clip(duty, 0.0, 1.0, out=duty)
    occupancy = (duty > 0.0).astype(np.uint8)
    # A cell with no emitter must carry the floor sentinel, not a stale value.
    snr[occupancy == 0] = SNR_FLOOR_DB

    return EpisodeTensors(
        occupancy=occupancy,
        duty=duty,
        snr_db=snr,
        emitter_id=eid,
        n_pulses=npulse,
        truth=tuple(e.truth() for e in scenario.emitters),
        grid=grid,
        dt_s=cfg.time.dt_s,
        n_slots=t,
        seed=scenario.seed,
        config_hash=cfg.hash(),
    )


def make_episode(seed: int, config: Config) -> EpisodeTensors:
    """Convenience: generate a scenario and build its tensors in one call.

    Args:
        seed: Scenario seed.
        config: Resolved configuration.

    Returns:
        The populated :class:`EpisodeTensors`.
    """
    return build_episode(generate_scenario(seed, config=config))
