"""Typed configuration: YAML -> Pydantic v2 models.

The models here are the **single source of truth** for types, units, ranges and
cross-field validity. ``docs/config_schema.md`` describes them; it does not
duplicate them at runtime.

Two properties are load-bearing:

* ``extra="forbid"`` -- an unknown key is a hard error. A typo'd
  hyper-parameter that silently does nothing is how you lose a day.
* Derived values (``n_slots``, channel widths, noise floor) are *computed*, never
  configured, so they cannot disagree with their inputs.

Layering is ``base.yaml`` <- ``<tier>.yaml`` (via ``extends:``) <- ``--set
a.b.c=v`` CLI overrides, resolved deterministically. The fully resolved mapping
is hashed into every artefact so no number is ever orphaned from the settings
that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from smartscan.env.types import SpectrumGrid

__all__ = [
    "EMITTER_CLASSES",
    "Config",
    "Range",
    "load_config",
    "resolve_config_mapping",
]

#: Canonical emitter class keys. The scenario ``mix`` must use exactly these.
EMITTER_CLASSES: tuple[str, ...] = (
    "fixed_cw",
    "pulsed_radar",
    "circular_scan",
    "sector_scan",
    "frequency_agile",
    "agile_beam",
    "comms_burst",
    "interferer",
)

CONFIG_DIR_CANDIDATES: tuple[Path, ...] = (
    Path("configs"),
    Path(__file__).resolve().parent / "configs",
    Path(__file__).resolve().parent.parent / "configs",
)


# --------------------------------------------------------------------------- #
# Range: the `[lo, hi]` sampling-prior syntax used throughout the YAML
# --------------------------------------------------------------------------- #
def _coerce_range(v: Any) -> Any:
    """Normalise a scalar or ``[lo, hi]`` pair into ``Range`` kwargs."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return {"lo": float(v), "hi": float(v)}
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"range must be a scalar or [lo, hi], got {v!r}")
        return {"lo": float(v[0]), "hi": float(v[1])}
    return v


class Range(BaseModel):
    """A closed sampling interval ``[lo, hi]``; a scalar collapses to ``lo == hi``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.lo > self.hi:
            raise ValueError(f"range lo ({self.lo}) must not exceed hi ({self.hi})")
        return self

    def sample(self, rng: np.random.Generator, size: int | None = None) -> Any:
        """Draw uniformly from the interval.

        Args:
            rng: Generator to draw from.
            size: Number of draws, or ``None`` for a scalar.

        Returns:
            A float, or a float64 array of shape ``(size,)``.
        """
        if self.lo == self.hi:
            return float(self.lo) if size is None else np.full(size, self.lo)
        return rng.uniform(self.lo, self.hi, size=size)

    def sample_int(self, rng: np.random.Generator, size: int | None = None) -> Any:
        """Draw uniform integers from ``[lo, hi]`` inclusive."""
        return rng.integers(int(self.lo), int(self.hi) + 1, size=size)

    def sample_log(self, rng: np.random.Generator, size: int | None = None) -> Any:
        """Draw log-uniformly from the interval.

        Used where a parameter spans orders of magnitude (PRI, hop rate); a
        uniform draw over ``[100, 10000]`` Hz would put 90 % of the mass above
        1 kHz and never exercise the slow-hopper regime.
        """
        if self.lo == self.hi:
            return float(self.lo) if size is None else np.full(size, self.lo)
        return np.exp(rng.uniform(np.log(self.lo), np.log(self.hi), size=size))


FloatRange = Annotated[Range, BeforeValidator(_coerce_range)]


class _Base(BaseModel):
    """Common model config: forbid unknown keys, freeze after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class RunConfig(_Base):
    """Seeding, output location and determinism switches."""

    name: str = "base"
    seed: int = 20260902
    n_seeds: int = Field(default=30, gt=0)
    out_dir: str = "runs"
    device: Literal["cpu", "cuda"] = "cpu"
    deterministic: bool = True
    torch_threads: int = Field(default=4, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class SpectrumConfig(_Base):
    """The frequency partition of the surveilled band."""

    f_start_hz: float = Field(default=0.5e9, gt=0)
    f_stop_hz: float = Field(default=18.0e9, gt=0)
    n_channels: int = Field(default=128, gt=0)
    partition: Literal["uniform", "log", "explicit"] = "uniform"
    edges_hz: list[float] | None = None

    @model_validator(mode="after")
    def _check(self) -> SpectrumConfig:
        # Validator 1: band ordering.
        if self.f_stop_hz <= self.f_start_hz:
            raise ValueError("spectrum.f_stop_hz must exceed f_start_hz")
        # Validator 2: explicit partition <-> edges.
        if self.partition == "explicit":
            if self.edges_hz is None:
                raise ValueError("spectrum.edges_hz is required when partition == 'explicit'")
            e = np.asarray(self.edges_hz, dtype=np.float64)
            if e.size != self.n_channels + 1:
                raise ValueError(f"edges_hz must have n_channels+1={self.n_channels + 1} entries")
            if not np.all(np.diff(e) > 0):
                raise ValueError("edges_hz must be strictly increasing")
            if not (np.isclose(e[0], self.f_start_hz) and np.isclose(e[-1], self.f_stop_hz)):
                raise ValueError("edges_hz endpoints must match f_start_hz / f_stop_hz")
        elif self.edges_hz is not None:
            raise ValueError("spectrum.edges_hz must be null unless partition == 'explicit'")
        return self

    def grid(self) -> SpectrumGrid:
        """Build the :class:`SpectrumGrid` implied by this configuration."""
        if self.partition == "explicit":
            edges = np.asarray(self.edges_hz, dtype=np.float64)
        elif self.partition == "log":
            # Constant fractional bandwidth -- what a real wideband channeliser
            # actually produces, and it puts more resolution at the low end
            # where channel spacing matters most.
            edges = np.geomspace(self.f_start_hz, self.f_stop_hz, self.n_channels + 1)
        else:
            edges = np.linspace(self.f_start_hz, self.f_stop_hz, self.n_channels + 1)
        return SpectrumGrid(
            edges_hz=edges,
            centers_hz=0.5 * (edges[:-1] + edges[1:]),
            widths_hz=np.diff(edges),
        )


class TimeConfig(_Base):
    """Slot duration and episode length."""

    dt_s: float = Field(default=1.0e-3, gt=0)
    episode_s: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _integral(self) -> TimeConfig:
        # Validator 4: dt must divide the episode, else slots are silently lost.
        ratio = self.episode_s / self.dt_s
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio):
            raise ValueError(f"time.episode_s / dt_s = {ratio} is not integral")
        return self

    @property
    def n_slots(self) -> int:
        """Derived episode length in slots, ``T``."""
        return int(round(self.episode_s / self.dt_s))


class DetectorConfig(_Base):
    """Square-law envelope detector parameters."""

    type: Literal["square_law"] = "square_law"
    pfa: float = Field(default=1.0e-4, gt=0.0, lt=1.0)
    swerling: Literal[0, 1] = 1
    n_integrate: int | Literal["auto"] = "auto"
    fft_size: int = Field(default=256, gt=0)
    n_integrate_max: int = Field(default=256, gt=0)
    snr_est_sigma_db: float = Field(default=2.0, ge=0)
    report_snr_on_miss: bool = False


class ReceiverConfig(_Base):
    """Receiver front end: IBW, retune cost, detector."""

    ibw_channels: int = Field(default=4, gt=0)
    action_space: Literal["center_index", "window_start"] = "center_index"
    mask_illegal_actions: bool = True
    t_settle_slots: int = Field(default=2, ge=0)
    straddle_enabled: bool = True
    straddle_loss_db: float = Field(default=1.5, ge=0)
    noise_figure_db: float = Field(default=4.0, ge=0)
    sensitivity_dbm: float = -110.0
    antenna_gain_dbi: float = 3.0
    backend: Literal["simulated", "soapy"] = "simulated"
    detector: DetectorConfig = DetectorConfig()


class EmitterDefaults(_Base):
    """Priors applied to every emitter class unless the class overrides them.

    When ``snr_db`` is set the generator authors the scenario in SNR space and
    back-solves ``range_km`` from the link budget; ``range_km`` is then a
    fallback used only if ``snr_db`` is null. See
    :mod:`smartscan.env.calibration` for why.
    """

    eirp_dbm: FloatRange = Range(lo=40.0, hi=70.0)
    range_km: FloatRange = Range(lo=20.0, hi=300.0)
    snr_db: FloatRange | None = Range(lo=6.0, hi=30.0)
    threat_priority: FloatRange = Range(lo=0.3, hi=0.9)
    misc_loss_db: float = 2.0


class _EmitterClass(_Base):
    """Base for a per-class prior block; unset link-budget fields inherit."""

    eirp_dbm: FloatRange | None = None
    range_km: FloatRange | None = None
    snr_db: FloatRange | None = None
    threat_priority: FloatRange | None = None
    misc_loss_db: float | None = None

    def inherited(self, defaults: EmitterDefaults) -> dict[str, Any]:
        """Return the link-budget priors, filling unset fields from ``defaults``."""
        return {
            "eirp_dbm": self.eirp_dbm or defaults.eirp_dbm,
            "range_km": self.range_km or defaults.range_km,
            "snr_db": self.snr_db or defaults.snr_db,
            "threat_priority": self.threat_priority or defaults.threat_priority,
            "misc_loss_db": defaults.misc_loss_db if self.misc_loss_db is None else self.misc_loss_db,
        }


class FixedCWConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "energy"


class PulsedRadarConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "pulse"
    pri_s: FloatRange = Range(lo=1.0e-4, hi=5.0e-3)
    pulse_width_s: FloatRange = Range(lo=1.0e-7, hi=5.0e-6)
    pri_mode: Literal["constant", "jittered", "stagger"] = "constant"
    pri_jitter_frac: float = Field(default=0.05, ge=0, lt=1)
    stagger_ratios: list[float] = Field(default_factory=lambda: [1.0, 1.12, 0.93, 1.07])


class CircularScanConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "pulse"
    scan_period_s: FloatRange = Range(lo=1.0, hi=12.0)
    beamwidth_deg: FloatRange = Range(lo=1.0, hi=6.0)
    sidelobe_db: FloatRange = Range(lo=-35.0, hi=-20.0)
    backlobe_db: float = -45.0
    scan_phase_frac: FloatRange = Range(lo=0.0, hi=1.0)
    pri_s: FloatRange = Range(lo=5.0e-4, hi=2.0e-3)
    pulse_width_s: FloatRange = Range(lo=5.0e-7, hi=3.0e-6)

    @model_validator(mode="after")
    def _ps_bounds(self) -> CircularScanConfig:
        # Validator 13: PS-fixed ranges must not drift.
        if not (self.scan_period_s.lo >= 1.0 and self.scan_period_s.hi <= 12.0):
            raise ValueError("circular_scan.scan_period_s must lie within the PS range [1, 12] s")
        if not (self.beamwidth_deg.lo >= 1.0 and self.beamwidth_deg.hi <= 6.0):
            raise ValueError("circular_scan.beamwidth_deg must lie within the PS range [1, 6] deg")
        return self


class SectorScanConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "pulse"
    sector_deg: FloatRange = Range(lo=60.0, hi=150.0)
    scan_rate_deg_s: FloatRange = Range(lo=30.0, hi=120.0)
    beamwidth_deg: FloatRange = Range(lo=1.5, hi=5.0)
    turnaround_dwell_s: FloatRange = Range(lo=0.05, hi=0.30)
    bidirectional: bool = True
    sidelobe_db: FloatRange = Range(lo=-32.0, hi=-22.0)
    pri_s: FloatRange = Range(lo=5.0e-4, hi=2.0e-3)
    pulse_width_s: FloatRange = Range(lo=5.0e-7, hi=3.0e-6)


class FrequencyAgileConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "pulse"
    hop_rate_hz: FloatRange = Range(lo=100.0, hi=10000.0)
    hop_set_size: FloatRange = Range(lo=4, hi=32)
    hop_set_contiguous_frac: float = Field(default=0.5, ge=0, le=1)
    hop_pattern: Literal["prng", "cyclic"] = "prng"

    @model_validator(mode="after")
    def _ps_bounds(self) -> FrequencyAgileConfig:
        # Validator 14: PS-fixed hop-rate range.
        if not (self.hop_rate_hz.lo >= 100.0 and self.hop_rate_hz.hi <= 10000.0):
            raise ValueError("frequency_agile.hop_rate_hz must lie within the PS range [100, 10000] Hz")
        return self


class AgileBeamConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "pulse"
    revisit_mean_s: FloatRange = Range(lo=0.5, hi=6.0)
    revisit_shape_k: float = Field(default=1.4, gt=0)
    look_duration_s: FloatRange = Range(lo=2.0e-3, hi=3.0e-2)
    beamwidth_deg: FloatRange = Range(lo=1.0, hi=3.0)
    pri_s: FloatRange = Range(lo=2.0e-4, hi=1.0e-3)
    pulse_width_s: FloatRange = Range(lo=3.0e-7, hi=2.0e-6)


class CommsBurstConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "energy"
    p_on_to_off: float = Field(default=0.004, gt=0, lt=1)
    p_off_to_on: float = Field(default=0.0008, gt=0, lt=1)
    dwell_dist: Literal["lognormal", "pareto"] = "lognormal"
    dwell_sigma: float = Field(default=1.1, gt=0)


class InterfererConfig(_EmitterClass):
    detection_mode: Literal["energy", "pulse"] = "energy"
    duty_frac: FloatRange = Range(lo=0.7, hi=1.0)


class EmittersConfig(_Base):
    """Per-class parameter priors used by the scenario generator."""

    defaults: EmitterDefaults = EmitterDefaults()
    fixed_cw: FixedCWConfig = FixedCWConfig()
    pulsed_radar: PulsedRadarConfig = PulsedRadarConfig()
    circular_scan: CircularScanConfig = CircularScanConfig()
    sector_scan: SectorScanConfig = SectorScanConfig()
    frequency_agile: FrequencyAgileConfig = FrequencyAgileConfig()
    agile_beam: AgileBeamConfig = AgileBeamConfig()
    comms_burst: CommsBurstConfig = CommsBurstConfig()
    interferer: InterfererConfig = InterfererConfig()

    def block(self, class_key: str) -> _EmitterClass:
        """Return the prior block for ``class_key``."""
        return getattr(self, class_key)


class ScenarioConfig(_Base):
    """Tier, emitter mix and channel placement."""

    difficulty: Literal["easy", "medium", "hard"] = "easy"
    n_emitters: int = Field(default=5, gt=0)
    mix: dict[str, int] = Field(default_factory=lambda: {"fixed_cw": 3, "pulsed_radar": 2})
    n_popup: int = Field(default=0, ge=0)
    popup_start_frac: float = Field(default=0.6, gt=0, lt=1)
    placement: Literal["poisson_disk", "uniform"] = "poisson_disk"
    min_channel_separation: int = Field(default=2, ge=0)
    allow_channel_collision: bool = True

    @model_validator(mode="after")
    def _check(self) -> ScenarioConfig:
        unknown = set(self.mix) - set(EMITTER_CLASSES)
        if unknown:
            raise ValueError(f"scenario.mix has unknown emitter classes: {sorted(unknown)}")
        # Validator 5: the mix must account for exactly n_emitters.
        total = sum(self.mix.values())
        if total != self.n_emitters:
            raise ValueError(f"sum(scenario.mix) = {total} != n_emitters = {self.n_emitters}")
        # Validator 6.
        if self.n_popup > self.n_emitters:
            raise ValueError("scenario.n_popup must not exceed n_emitters")
        if any(v < 0 for v in self.mix.values()):
            raise ValueError("scenario.mix counts must be non-negative")
        return self


class BeliefConfig(_Base):
    """Beta posterior, decay and derived feature layout."""

    alpha_prior: float = Field(default=1.0, gt=0)
    beta_prior: float = Field(default=1.0, gt=0)
    decay_half_life_slots: int = Field(default=2000, gt=0)
    ewma_activity_alpha: float = Field(default=0.02, gt=0, le=1)
    ewma_snr_alpha: float = Field(default=0.10, gt=0, le=1)
    n_features: int = 12
    n_global_features: int = 5
    period_estimator: Literal["lomb_scargle", "sdif", "none"] = "lomb_scargle"
    period_min_confidence: float = Field(default=0.6, ge=0, le=1)


class RewardConfig(_Base):
    """Reward weights ``w1..w6`` (architecture.md §11.4)."""

    w1_threat_intercept: float = 10.0
    w2_novelty: float = 5.0
    w3_reconfirm: float = 0.5
    w4_retune: float = 0.2
    w5_interferer_dwell: float = 1.0
    w6_staleness: float = 0.3
    reconfirm_cap_per_emitter: int = Field(default=20, ge=0)
    normalise_by_episode: bool = True


class SequentialSweepConfig(_Base):
    step_channels: int = Field(default=4, gt=0)
    direction: Literal["up", "updown"] = "up"
    #: Slots dwelt on each window before stepping. NOT cosmetic: with
    #: ``t_settle = 2`` a one-slot dwell spends two thirds of the episode
    #: settling, so total observation time per channel is ``T*d/(N_win*(d+2))``
    #: -- 208 slots at d=1 but 446 at d=5. Leaving this at 1 would make the
    #: incumbent baseline a strawman. TUNED by ``eval/ablation.py``: sweeping
    #: d over {1,2,3,5,8,12,20} on MEDIUM puts the best TTFI at d=3.
    dwell_slots: int = Field(default=3, gt=0)


class RandomScanConfig(_Base):
    seed_offset: int = 0
    dwell_slots: int = Field(default=1, gt=0)


class PriorityRRConfig(_Base):
    prior_wrong_frac: float = Field(default=0.4, ge=0, le=1)
    weight_floor: float = Field(default=0.05, ge=0, le=1)


class EpsilonGreedyConfig(_Base):
    epsilon: float = Field(default=0.10, ge=0, le=1)
    epsilon_decay: float = Field(default=0.9995, gt=0, le=1)
    epsilon_min: float = Field(default=0.01, ge=0, le=1)


class UCB1Config(_Base):
    c: float = 1.4142135623730951
    discounted: bool = True
    gamma: float = Field(default=0.9995, gt=0, le=1)


class ThompsonConfig(_Base):
    posterior: Literal["beta"] = "beta"


class WhittleConfig(_Base):
    discount: float = Field(default=0.99, gt=0, lt=1)
    subsidy_lo: float = 0.0
    subsidy_hi: float = 1.0
    bisect_tol: float = Field(default=1.0e-6, gt=0)
    bisect_max_iter: int = Field(default=60, gt=0)
    use_closed_form: bool = True
    check_indexability: bool = True
    min_transitions_for_estimate: int = Field(default=8, ge=0)


class CoprimeSweepConfig(_Base):
    base_period_s: float = Field(default=0.048, gt=0)
    dwell_slots: int = Field(default=1, gt=0)
    irrationality: Literal["golden", "silver", "explicit"] = "golden"
    avoid_detected_periods: bool = True


class PhaseLockedConfig(_Base):
    guard_sigma: float = Field(default=3.0, ge=0)
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    fallback: Literal["whittle", "sequential", "thompson"] = "whittle"


class AgentsConfig(_Base):
    """Per-scheduler hyper-parameters."""

    #: Weight on the staleness/coverage term shared by every value-based policy
    #: (bandits, Whittle). The mission objective is not "detect the most signal"
    #: but "detect signal AND keep the band covered", so the policies optimise
    #: the stated objective rather than a detection-only proxy. Without this term
    #: every value policy parks on its best window and coverage collapses --
    #: measured, not assumed. Swept in eval/ablation.py.
    coverage_weight: float = 1.0

    sequential_sweep: SequentialSweepConfig = SequentialSweepConfig()
    random_scan: RandomScanConfig = RandomScanConfig()
    priority_round_robin: PriorityRRConfig = PriorityRRConfig()
    epsilon_greedy: EpsilonGreedyConfig = EpsilonGreedyConfig()
    ucb1: UCB1Config = UCB1Config()
    thompson: ThompsonConfig = ThompsonConfig()
    whittle: WhittleConfig = WhittleConfig()
    coprime_sweep: CoprimeSweepConfig = CoprimeSweepConfig()
    phase_locked: PhaseLockedConfig = PhaseLockedConfig()


class DistillationConfig(_Base):
    """Privileged-teacher distillation, applied at **training time only**."""

    enabled: bool = True
    lambda_kd: float = Field(default=0.5, ge=0)
    temperature: float = Field(default=2.0, gt=0)
    teacher_epochs: int = Field(default=10, ge=0)


class PredictorConfig(_Base):
    """Supervised next-slot occupancy predictor."""

    arch: Literal["gru", "tcn", "transformer"] = "gru"
    window_slots: int = Field(default=128, gt=0)
    input_planes: int = Field(default=4, gt=0)
    hidden_dim: int = Field(default=128, gt=0)
    n_layers: int = Field(default=2, gt=0)
    tcn_dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    transformer_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    loss: Literal["masked_focal", "masked_bce"] = "masked_focal"
    focal_gamma: float = Field(default=2.0, ge=0)
    focal_alpha: float = Field(default=0.25, gt=0, lt=1)
    lr: float = Field(default=3.0e-4, gt=0)
    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=20, ge=0)
    distillation: DistillationConfig = DistillationConfig()


class PPOConfig(_Base):
    lr: float = Field(default=3.0e-4, gt=0)
    clip_range: float = Field(default=0.2, gt=0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    n_epochs: int = Field(default=4, gt=0)
    rollout_steps: int = Field(default=256, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)


class DQNConfig(_Base):
    lr: float = Field(default=1.0e-4, gt=0)
    buffer_size: int = Field(default=100000, gt=0)
    learning_starts: int = Field(default=5000, ge=0)
    target_update_interval: int = Field(default=1000, gt=0)
    train_freq: int = Field(default=4, gt=0)
    double: bool = True
    duelling: bool = True
    prioritised: bool = True
    exploration_final_eps: float = Field(default=0.02, ge=0, le=1)


class HybridConfig(_Base):
    predictor_checkpoint: str | None = None
    freeze_predictor: bool = True


class RLConfig(_Base):
    """Reinforcement-learning agent configuration."""

    algo: Literal["ppo", "dqn"] = "ppo"
    implementation: Literal["from_scratch", "sb3"] = "from_scratch"
    encoder: Literal["conv1d", "mlp"] = "conv1d"
    hidden_dim: int = Field(default=256, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    total_steps: int = Field(default=200000, gt=0)
    n_envs: int = Field(default=8, gt=0)
    ppo: PPOConfig = PPOConfig()
    dqn: DQNConfig = DQNConfig()
    hybrid: HybridConfig = HybridConfig()


class ScanOnScanConfig(_Base):
    poi_model: Literal["deterministic", "exponential"] = "deterministic"
    max_horizon_s: float = Field(default=120.0, gt=0)


class EstimatorsConfig(_Base):
    method: Literal["lomb_scargle", "sdif", "both"] = "both"
    period_grid_s: FloatRange = Range(lo=0.5, hi=15.0)
    n_period_bins: int = Field(default=4000, gt=0)
    deconvolve_window: bool = True
    ls_peak_snr_threshold: float = Field(default=6.0, ge=0)
    sdif_threshold_k: float = Field(default=3.0, gt=0)
    sdif_subharmonic_check: bool = True


class AnalysisConfig(_Base):
    scan_on_scan: ScanOnScanConfig = ScanOnScanConfig()
    estimators: EstimatorsConfig = EstimatorsConfig()


class EvalConfig(_Base):
    """Benchmark protocol."""

    agents: list[str] = Field(
        default_factory=lambda: ["sequential", "random", "priority_rr", "ucb1", "thompson", "whittle", "ppo"]
    )
    n_bootstrap: int = Field(default=10000, gt=0)
    ci: float = Field(default=0.95, gt=0, lt=1)
    paired: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: [
            "ttfi_hard_median_s",
            "ttfi_median_s",
            "twir_rate",
            "twir_coverage",
            "coverage",
            "staleness_max_s",
            "waste_fraction",
            "popup_detect_rate",
            "discovery_auc",
            "fa_burden",
        ]
    )
    baseline_agent: str = "sequential"
    save_plots: bool = True
    save_trajectories: bool = False

    @model_validator(mode="after")
    def _baseline_present(self) -> EvalConfig:
        # Validator 12.
        if self.baseline_agent not in self.agents:
            raise ValueError(
                f"eval.baseline_agent {self.baseline_agent!r} is not in eval.agents {self.agents}"
            )
        # Validator 16: a metric name that does not match a key produced by
        # evaluate_episode is silently skipped by the benchmark, so the
        # comparison just vanishes. Caught here instead.
        from smartscan.analysis.metrics import METRIC_KEYS

        unknown = [m for m in self.metrics if m not in METRIC_KEYS]
        if unknown:
            raise ValueError(
                f"eval.metrics contains names that evaluate_episode does not produce: "
                f"{unknown}. Valid keys: {sorted(METRIC_KEYS)}"
            )
        return self


class Config(_Base):
    """The fully resolved SmartScan configuration."""

    schema_version: int = 1
    run: RunConfig = RunConfig()
    spectrum: SpectrumConfig = SpectrumConfig()
    time: TimeConfig = TimeConfig()
    receiver: ReceiverConfig = ReceiverConfig()
    emitters: EmittersConfig = EmittersConfig()
    scenario: ScenarioConfig = ScenarioConfig()
    belief: BeliefConfig = BeliefConfig()
    reward: RewardConfig = RewardConfig()
    agents: AgentsConfig = AgentsConfig()
    predictor: PredictorConfig = PredictorConfig()
    rl: RLConfig = RLConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _cross_section(self) -> Config:
        # Validator 15.
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version {self.schema_version}; this build supports 1")
        # Validator 3.
        if self.receiver.ibw_channels > self.spectrum.n_channels:
            raise ValueError("receiver.ibw_channels must not exceed spectrum.n_channels")
        # Validator 9: belief feature counts are code constants, surfaced in config
        # only so a mismatch fails at load rather than as a shape error 40 minutes
        # into training.
        from smartscan.agents.belief import N_CHANNEL_FEATURES, N_GLOBAL_FEATURES

        if self.belief.n_features != N_CHANNEL_FEATURES:
            raise ValueError(f"belief.n_features must be {N_CHANNEL_FEATURES} (code constant)")
        if self.belief.n_global_features != N_GLOBAL_FEATURES:
            raise ValueError(f"belief.n_global_features must be {N_GLOBAL_FEATURES} (code constant)")
        return self

    # -- derived ---------------------------------------------------------- #
    @property
    def n_slots(self) -> int:
        """Episode length in slots."""
        return self.time.n_slots

    @property
    def n_channels(self) -> int:
        """Number of channels ``B``."""
        return self.spectrum.n_channels

    @property
    def n_actions(self) -> int:
        """Size of the discrete action space (``B``; illegal tunes are masked)."""
        return self.spectrum.n_channels

    def grid(self) -> SpectrumGrid:
        """Build the spectrum grid."""
        return self.spectrum.grid()

    def hash(self) -> str:
        """Return a stable blake2b digest of the resolved configuration.

        Embedded in every artefact so no number is ever orphaned from the
        settings that produced it.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def with_overrides(self, **sections: Any) -> Config:
        """Return a copy with whole sections replaced (models are frozen)."""
        data = self.model_dump()
        for key, value in sections.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = _deep_merge(data[key], value)
            else:
                data[key] = value
        return Config.model_validate(data)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``.

    Mappings merge key-wise; every other type (including lists) replaces
    wholesale. Replacing lists rather than concatenating is deliberate: a tier
    that sets ``eval.agents`` means *these agents*, not *these as well*.
    """
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _find_config(path: str | Path) -> Path:
    """Resolve a config path, falling back to the packaged ``configs/`` copy."""
    p = Path(path)
    if p.is_file():
        return p
    for base in CONFIG_DIR_CANDIDATES:
        cand = base / p.name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"config {path!r} not found; looked in {[str(c) for c in CONFIG_DIR_CANDIDATES]}"
    )


def _parse_scalar(text: str) -> Any:
    """Parse a ``--set`` value with YAML semantics (so ``true``/``3``/``[1,2]`` work)."""
    return yaml.safe_load(text)


def resolve_config_mapping(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    _seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Resolve a config file to a plain mapping, following ``extends:`` chains.

    Args:
        path: Config file path or bare filename (resolved against ``configs/``).
        overrides: Dotted-key overrides applied last, e.g.
            ``{"run.seed": 7, "scenario.n_emitters": 9}``.
        _seen: Internal cycle guard.

    Returns:
        The merged mapping, with ``extends`` stripped.

    Raises:
        ValueError: On a circular ``extends`` chain.
    """
    resolved = _find_config(path).resolve()
    key = str(resolved)
    if key in _seen:
        raise ValueError(f"circular 'extends' chain at {key}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    merged = _deep_merge(resolve_config_mapping(parent, None, _seen | {key}), raw) if parent else raw

    for dotted, value in (overrides or {}).items():
        node = merged
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {dotted!r} traverses non-mapping key {part!r}")
        node[parts[-1]] = _parse_scalar(value) if isinstance(value, str) else value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load, layer and validate a configuration.

    Args:
        path: Config file path or bare filename (e.g. ``"medium.yaml"``).
        overrides: Dotted-key overrides applied after the ``extends`` chain.

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        pydantic.ValidationError: If any field or cross-field rule fails. Unknown
            keys are errors, not warnings.
    """
    return Config.model_validate(resolve_config_mapping(path, overrides))
