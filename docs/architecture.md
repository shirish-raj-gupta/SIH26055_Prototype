# SmartScan — Architecture

**SIH 26055 · "Smart Scan Strategy for Electronic Warfare" (DRDO / iDEX)**

Status: **IMPLEMENTED.** This document was written and approved as a design
before any code existed; §21 records what changed once it met reality, and why.
Companion documents: [`docs/config_schema.md`](config_schema.md) (normative config),
[`docs/theory.md`](theory.md) (Whittle + scan-on-scan derivations),
[`docs/related_work.md`](related_work.md) (verified citations and the gap we fill),
[`docs/hardware_roadmap.md`](hardware_roadmap.md) (SDR bring-up).

---

## 1. Purpose and scope

SmartScan is a closed-loop **Electronic Support (ES) receiver scheduler**. At every dwell
slot it decides which slice of a 0.5–18 GHz surveilled band the receiver should tune to,
in order to intercept unknown emitters *fast* and *often*, with **no prior intelligence**
about emitter frequencies, scan periods, or activity patterns.

The receiver's instantaneous bandwidth (IBW) is 1/32 of the surveilled band, so at any
instant **31/32 of the spectrum is unobserved — not empty, unknown**. That single fact
drives every design decision below. (The design assumed 1/16; §21-C explains why it
had to be 1/32 for the problem to be non-trivial at all.)

In scope: a simulator that produces ground truth, a receiver model with realistic
detection statistics and retune cost, a shared belief tracker, nine scheduler families
(open-loop → bandit → restless-bandit → supervised → RL → hybrid), an analytical
scan-on-scan module, and a reproducible benchmark harness.

Out of scope for the prototype: emitter geometry/AOA, deinterleaving of overlapping pulse
trains into individual PRIs, multi-receiver fusion, and real RF capture (the hardware path
is *designed for* but stubbed — see §9).

---

## 2. Problem formalisation

SmartScan is a **POMDP** `(S, A, O, T, Z, R, γ)`:

| Symbol | Meaning | Prototype default |
|---|---|---|
| `S` | Hidden joint emitter state: per-emitter phase, hop index, beam pointing, Markov mode | continuous + discrete, factored per emitter |
| `A` | Centre-channel index `a ∈ {0..B-1}`, illegal edge tunes masked | `B = 128`, 125 legal |
| `O` | `(hits, snr_est, pfa_flags)` over the `K` in-IBW channels only | `K = 4` |
| `T` | Emitter dynamics — **exogenous**: the world evolves whether we look or not | restless |
| `Z` | Probabilistic detection `p_detect(SNR, N)`, false alarms at `Pfa` | §7.4 |
| `R` | Threat-weighted intercept − retune − staleness | §11.4 |
| `γ` | Discount | 0.99 |

Two structural properties define the difficulty and justify the algorithm choices:

1. **Restlessness.** Emitter state advances during the 15/16 of the band we are not
   watching. This rules out classical (rested) MAB optimality guarantees and points
   directly at the **restless multi-armed bandit / Whittle index** formulation (§10.3).
2. **Deterministic coincidence.** Both our sweep and a mechanically-scanning emitter are
   *periodic*. Two periodic processes do not intercept "with probability p per look" —
   they either drift into coincidence or lock out permanently. A uniform sweep whose
   period is commensurate with the emitter's can be **provably blind forever** (§12.1).
   This is the failure mode the problem statement is really about, and it is why a
   number-theoretic sweep and a phase-locked park beat any amount of naive averaging.

---

## 3. System overview

```
                    ┌───────────────────── configs/*.yaml (§14) ──────────────────────┐
                    │  seeded, Pydantic-validated, hashed into every artefact         │
                    └──────────────────────────┬─────────────────────────────────────┘
                                               ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ smartscan.env — GROUND TRUTH  (precomputed once per episode, fully vectorised)    │
  │                                                                                  │
  │   scenario_gen ──► [BaseEmitter]* ──► activity(t) ──► propagation ──► tensors    │
  │                                                        X[b,t]   duty[b,t]        │
  │                                                        SNR[b,t] E[b,t]           │
  └──────────────────────────────┬───────────────────────────────────────────────────┘
                                 │  (schedulers never touch these; eval does)
                                 ▼
  ┌────────────────────────────────────────────┐      ┌─────────────────────────────────┐
  │ Receiver (§8)                              │ HAL  │ ReceiverBackend (ABC) (§9)      │
  │  window [a−K/2, …) · t_settle · straddle   │◄────►│   SimulatedBackend              │
  │  detector: Pd / Pfa (§7.4)                 │      │   SoapySDRBackend  (stub)       │
  └────────────────┬───────────────────────────┘      └─────────────────────────────────┘
                   │  Observation(hits, snr_est, pfa_flags, slots_elapsed)
                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ BeliefState (§10.1) — the ONLY thing every scheduler shares                       │
  │  Beta(α,β) per channel · staleness · EWMA activity · online period estimate       │
  │  decay-to-prior for unvisited channels ──► features(): float32 [B,F] + [G]        │
  └────────────────┬─────────────────────────────────────────────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ Scheduler.act(belief, t) -> action                                                │
  │  baselines │ bandits │ whittle │ predictors │ rl_agents │ hybrid │ scan-on-scan   │
  └────────────────┬─────────────────────────────────────────────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ eval — metrics · benchmark (30 seeds, paired bootstrap CIs) · ablation · plots     │
  │        ──► runs/<name>/metrics.json  +  figures  +  dashboard feed                 │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

**Dependency rule (enforced by an import-linter test):** `agents/` may import `env.types`
and `agents.belief` but **never** `env.rf_environment` internals. No scheduler can read
`X`, `SNR`, or `E`. The single exception is the *privileged teacher* (§11.2), which is
train-time-only and lives behind an explicit `PrivilegedAccess` context manager that
raises if entered while `eval_mode` is set.

---

## 4. Module map

| Path | Responsibility | Key exports |
|---|---|---|
| `env/types.py` | Frozen dataclasses shared everywhere; zero internal deps | `Observation`, `SpectrumGrid`, `EmitterTruth`, `EpisodeTensors` |
| `env/emitters.py` | 8 emitter classes, all `BaseEmitter.activity(t) -> Activity` | `FixedCW`, `PulsedRadar`, `CircularScanRadar`, `SectorScanRadar`, `FrequencyAgile`, `AgileBeamRadar`, `CommsBurst`, `Interferer` |
| `env/propagation.py` | Link budget, noise floor, `p_detect`, `threshold_from_pfa` | `link_budget_dbm`, `p_detect`, `snr_at_receiver` |
| `env/rf_environment.py` | Scenario generator, tensor assembly, Gymnasium `Env` wrapper | `generate_scenario`, `RFEnvironment`, `SmartScanGymEnv` |
| `env/receiver.py` | IBW window, settling, straddle loss, observation synthesis | `Receiver.step()` |
| `hal/backend.py` | `ReceiverBackend` ABC: `tune / capture / get_detections` | ABC only |
| `hal/simulated.py` | Backend over `EpisodeTensors` | `SimulatedBackend` |
| `hal/soapy_stub.py` | Non-functional SoapySDR skeleton, same signature, `TODO(hardware)` | `SoapySDRBackend` |
| `agents/belief.py` | Beta posterior, decay, staleness, feature vector | `BeliefState` |
| `agents/baselines.py` | `SequentialSweep`, `RandomScan`, `PriorityRoundRobin` | — |
| `agents/bandits.py` | `EpsilonGreedy`, `UCB1`, `ThompsonSampling` (discounted) | — |
| `agents/whittle.py` | Restless index: closed form + numeric indexability check | `WhittleIndexScheduler` |
| `agents/predictors.py` | GRU / dilated TCN / Transformer; masked focal loss; distillation | `SequencePredictorScheduler` |
| `agents/rl_agents.py` | From-scratch PPO + Double-DQN with action masking | `PPOScheduler`, `DQNScheduler` |
| `agents/hybrid.py` | Predictor probabilities as an extra RL observation plane | `HybridScheduler` |
| `analysis/scan_on_scan.py` | POI / TTI closed forms, `CoprimeSweepScheduler`, `PhaseLockedScheduler` | — |
| `analysis/estimators.py` | Lomb–Scargle (window-deconvolved), CDIF/SDIF | `estimate_period_ls`, `estimate_period_sdif` |
| `analysis/metrics.py` | TTFI, TWIR, staleness, waste, pop-up latency, bootstrap CIs | `evaluate_episode` |
| `data/` | Optional offline datasets + predictor dataset builder | `kaggle_io`, `dataset_builder`, `loaders` |
| `eval/` | `benchmark.py` (30 seeds × agents), `ablation.py`, `plots.py` | — |
| `cli.py` | `run`, `benchmark`, `train`, `ablate`, `estimate`, `reproduce` | Typer app |

---

## 5. Core data contracts

```python
@dataclass(frozen=True)
class SpectrumGrid:
    edges_hz: np.ndarray       # (B+1,) float64, strictly increasing; supports NON-UNIFORM
    centers_hz: np.ndarray     # (B,)   float64
    widths_hz: np.ndarray      # (B,)   float64
    n_channels: int            # B

@dataclass(frozen=True)
class EpisodeTensors:          # GROUND TRUTH — never visible to a scheduler
    occupancy:  np.ndarray     # X    (B, T) uint8    1 if any emitter energy in the slot
    duty:       np.ndarray     # d    (B, T) float32  in [0,1], sub-slot occupied fraction
    snr_db:     np.ndarray     # SNR  (B, T) float32  peak SNR of the strongest emitter
    emitter_id: np.ndarray     # E    (B, T) int16    0 = noise; strongest emitter wins
    truth:      tuple[EmitterTruth, ...]
    grid: SpectrumGrid; dt_s: float; n_slots: int; config_hash: str; seed: int

@dataclass(frozen=True)
class Observation:
    window:        tuple[int, int]  # [lo, hi) channel indices actually observed
    hits:          np.ndarray       # (K,) bool     detection declared (may be a false alarm)
    snr_est_db:    np.ndarray       # (K,) float32  NaN where no hit
    pfa_flags:     np.ndarray       # (K,) bool     hit arose from noise alone (EVAL ONLY)
    slots_elapsed: int              # 1 + t_settle if retuned, else 1
    t: int
```

**Shapes are contracts.** `tests/test_contracts.py` asserts dtype and shape of every tensor
at every boundary; silent `float64` promotion is a real source of both slowness and
non-determinism.

---

## 6. Time–frequency grid, and why `duty` exists

Defaults: `B = 64` channels over 0.5–18 GHz (273.44 MHz each, uniform; non-uniform and log
partitions supported via explicit edges), `dt = 1 ms`, `T = 10 s` → **10 000 slots**.
IBW `K = 4` ≈ 1.09 GHz.

A binary `X[b,t]` alone is physically wrong here, because emitter timescales straddle the
slot boundary in *both* directions:

* A radar pulse is `PW ≈ 1 µs` inside a `1 ms` slot → occupies 0.1 % of the slot.
* A frequency-agile emitter hopping at 10 kHz changes channel **10 times per slot**.

So the environment carries `duty[b,t] ∈ [0,1]` = the fraction of slot `t` in which channel
`b` held energy, alongside the binary `X`. `duty` feeds detection; `X = (duty > 0)` is kept
because the PS asks for it and because the metrics are defined on it.

**Detection mode is a property of the emitter, not the receiver.** This matters:

| Mode | Used by | Model |
|---|---|---|
| `energy` | FixedCW, CommsBurst, Interferer | Noncoherent integration over the dwell; effective SNR gains `10·log10(N·duty)` |
| `pulse` | PulsedRadar, CircularScan, SectorScan, AgileBeam, FrequencyAgile | Per-pulse detection at *peak* SNR; `n = floor(dwell / PRI)` opportunities; `Pd_dwell = 1 − (1 − Pd_pulse)ⁿ` |

Real ES receivers use fast log-video detection precisely because integrating a 1 µs pulse
over a 1 ms dwell buries it. Modelling this correctly is what makes "dwell longer" a real
trade-off against "hop more often" rather than a free win.

---

## 7. Environment

### 7.1 Emitter interface

```python
class BaseEmitter(ABC):
    emitter_id: int; threat_priority: float; is_novel: bool
    detection_mode: Literal["energy", "pulse"]
    t_first_active: int                       # > 0.6T for pop-ups

    @abstractmethod
    def activity(self, t: np.ndarray) -> Activity: ...
    #   Activity(channels: (M,) int32, duty: (M,) float32, gain_db: (M,) float32)
    #   VECTORISED over the whole slot axis — see §15.
```

`gain_db` is the emitter's *instantaneous antenna gain toward us* (main lobe, sidelobe or
backlobe), kept separate from the link budget so one propagation path serves all classes.

### 7.2 The eight classes

| Class | Model | Why it is in the benchmark |
|---|---|---|
| **FixedCW** | one channel, duty 1, static SNR | trivial floor case; anything must find it |
| **PulsedRadar** | PRI, PW, duty; variants `constant / jittered(σ%) / stagger(list)` | tests pulse-mode detection and PRI-aliased revisit |
| **CircularScanRadar** | `Ts ∈ [1,12] s`, `θ_bw ∈ [1,6]°`; illuminated `θ_bw/360 · Ts` per revolution (**≈ 11 ms at 4 s / 1°**); sidelobes at `−35…−20 dB` give rare weak intercepts | **the core scan-on-scan target** |
| **SectorScanRadar** | raster over `[φ0, φ1]`, bidirectional, turnaround dwell | arrivals are *unevenly* spaced within a frame → breaks naive periodograms |
| **FrequencyAgile** | PRNG hop over a hop-set (4–32 channels), 100 Hz–10 kHz; sub-slot hops spread fractional duty across several channels in one slot | punishes per-channel independence; rewards learning the hop-set |
| **AgileBeamRadar** | AESA random look scheduling: revisit ~ heavy-tailed Gamma, no fixed period | **hardest**; period estimators must *fail gracefully* and the agent fall back to coverage |
| **CommsBurst** | 2-state Markov ON/OFF, log-normal / Pareto dwell times | long OFF tails punish "confirmed idle → never revisit" |
| **Interferer** | high duty, `threat_priority ≈ 0.02` | **the trap.** A pure detection-rate objective parks on it; threat weighting and `w5` must overcome that |

### 7.3 Propagation

`P_rx(dBm) = EIRP + G_emit(θ,t) + G_rx − FSPL(f,R) − L_atm(f,R) − L_misc`, with
`FSPL = 32.44 + 20·log10(f_MHz) + 20·log10(R_km)`,
noise floor `N = −174 + NF + 10·log10(BW_ch)`, and `SNR = P_rx − N`.
Atmospheric loss is a small tabulated dB/km vs frequency (the O₂ line at 60 GHz is out of
band; a linear-in-`f` fit over 0.5–18 GHz is adequate and is flagged as such in the
docstring).

### 7.4 Detection statistics — no hard-coded `Pd = 1`

Square-law envelope detector, `N` noncoherently integrated samples.

* Noise-only detector output is exponential; the sum of `N` samples is Erlang(`N`).
  Threshold from the required `Pfa`: `V_T = chi2.isf(Pfa, df=2N) / 2` (unit-variance
  quadrature normalisation).
* **Swerling 0 / non-fluctuating:** the statistic is non-central chi-square with `2N` dof
  and non-centrality `λ = 2·N·SNR_lin`, so
  `Pd = scipy.stats.ncx2.sf(2·V_T, df=2N, nc=λ)` — *exact*, not an approximation.
* **Swerling I** (scan-to-scan Rayleigh fluctuation — the right default for a scanning
  radar seen through a single beam pass): closed form for `N = 1`,
  `Pd = Pfa^(1/(1+SNR_lin))`; for `N > 1` the standard incomplete-gamma series.
* Albersheim's equation appears only as a fast sanity check in tests (validity
  `1e-7 ≤ Pfa ≤ 1e-3`, `0.1 ≤ Pd ≤ 0.9`, `1 ≤ N ≤ 8096`), never in the hot path.

Reference: M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., §6.
Every approximation used is named in the function docstring, as required.

**False alarms.** Each observed channel independently declares a hit with probability
`Pfa` when no emitter is present. At `Pfa = 1e-4`, `K = 4`, 10 000 slots that is ≈ 4 false
alarms per episode — enough to corrupt a naive period estimator, which is the point.
`pfa_flags` records them for *evaluation only*; the agent cannot tell.

### 7.5 Scenario generator

`generate_scenario(seed, n_emitters, difficulty) -> Scenario` — pure and reproducible.

| Tier | n | Mix | Notes |
|---|---|---|---|
| **EASY** | 5 | 3 FixedCW, 2 PulsedRadar (constant PRI) | static; must run end-to-end < 10 min CPU |
| **MEDIUM** | 15 | 3 CW, 4 pulsed, 3 circular, 1 sector, 2 freq-agile, 1 agile-beam, 1 comms | **the headline benchmark (acceptance §3)** |
| **HARD** | 30 | 8 circular, 3 sector, 4 freq-agile, 3 agile-beam, 4 pulsed, 3 comms, **5 interferers**, **2 pop-ups** (`t_first > 0.6T`) | stress + acceptance §5 |

Every sampled parameter comes from a named, config-declared prior (§14). Emitters are
placed on channels by a **blue-noise / Poisson-disk** sampler rather than a uniform draw,
so scenarios are not trivially solved by one lucky contiguous window and channel collisions
occur at a controlled rate.

---

## 8. Receiver

* Action `a` = **centre channel**. Observed window = `[a − ⌊(K−1)/2⌋, a + ⌈(K−1)/2⌉]`
  (for `K = 4`: `[a−1, a+2]`). Windows falling off a band edge are **illegal and masked**
  → 61 of 64 actions legal at `K = 4`. (`action_space: window_start` is the alternative
  convention, also 61 actions; both supported, `center_index` is the default.)
* **Retune cost.** Changing `a` costs `t_settle` slots (default 2) in which *nothing is
  observed*. Staying costs 0. This is the entire economics of the problem: a greedy hopper
  pays 3 slots per look, a parked receiver pays 1.
* **Straddle loss.** Energy near a channel edge loses `straddle_loss_db` (default 1.5 dB) —
  models channeliser scalloping. Switchable off for ablation.
* `Receiver.step(action) -> Observation`. Detection draws come from the RNG substream
  `("receiver",)` so the *same* agent trajectory replays bit-identically.

---

## 9. Hardware Abstraction Layer — the swap-in path

```python
class ReceiverBackend(ABC):
    @abstractmethod
    def tune(self, center_hz: float) -> None: ...
    @abstractmethod
    def capture(self, duration_s: float) -> CaptureHandle: ...
    @abstractmethod
    def get_detections(self, capture: CaptureHandle) -> list[Detection]: ...
    # + properties: ibw_hz, tune_range_hz, settle_time_s, noise_figure_db
```

`SimulatedBackend` reads `EpisodeTensors`. `SoapySDRBackend` is a **non-functional stub**
carrying the identical signature and a `TODO(hardware)` block that names the exact SoapySDR
calls (`Device.setFrequency`, `setupStream`, `readStream`), the Welch-PSD → CFAR chain that
replaces the simulator's detector, and the calibration steps (noise-figure measurement,
per-device `t_settle` measurement) for RTL-SDR / HackRF / ADALM-Pluto / USRP B2xx.

**Scheduler code never changes.** Everything above the HAL consumes `Observation`. This
hardware-readiness is a scored criterion, so it gets its own README section plus
`docs/hardware_roadmap.md` with a device comparison table (IBW, tune time, cost).

---

## 10. Belief and the model-based schedulers

### 10.1 `BeliefState`

Per channel `b`, updated only from `Observation` — never from ground truth:

* **Occupancy posterior** `Beta(α_b, β_b)`; on a look, `α += hit`, `β += (1 − hit)`.
* **Decay to prior** over `Δ` unobserved slots: `α ← 1 + (α−1)·ρ^Δ`, `β ← 1 + (β−1)·ρ^Δ`
  with `ρ = 2^(−1/half_life_slots)`. This is discounted-Bayesian tracking (Garivier &
  Moulines, discounted UCB; Raj & Kalyani, discounted Thompson sampling) — the principled
  way to encode *"what I learned 4 s ago about a scanning radar is nearly worthless."*
* `time_since_visit[b]`, `time_since_hit[b]` — staleness / age-of-information.
* EWMA activity estimate; EWMA SNR estimate.
* **Online period estimate** per channel from `analysis/estimators.py` (§12.2):
  `(T̂e, confidence, phase_to_next_arrival)`.

`BeliefState.features() -> (np.ndarray[B, F], np.ndarray[G])`:

| # | Per-channel feature (F = 12) | Scaling |
|---|---|---|
| 0 | `E[p] = α/(α+β)` | [0,1] |
| 1 | posterior std | [0,1] |
| 2, 3 | `log1p(α)`, `log1p(β)` | ÷5 |
| 4 | `log1p(time_since_visit)` | ÷log T |
| 5 | `log1p(time_since_hit)` | ÷log T |
| 6 | EWMA activity | [0,1] |
| 7 | EWMA SNR estimate (dB) | ÷40 |
| 8 | period-estimate confidence | [0,1] |
| 9, 10 | `sin`, `cos` of predicted phase-to-next-arrival | [−1,1] |
| 11 | learned interferer score (running threat estimate) | [0,1] |

Global block `G = 5`: `t/T`, current centre channel ÷B, slots since last retune, fraction
of band visited, running detection rate. Observation size `64×12 + 5 = 773` floats.

The **sin/cos phase pair** is deliberate: it lets a network represent "arrives soon"
without a discontinuity at the wrap, which is what makes phase-locking learnable rather
than only coverage.

### 10.2 Bandits, and why plain bandits are not enough

`EpsilonGreedy` / `UCB1` / `ThompsonSampling` operate on the decayed Beta posteriors. They
are *rested*-bandit algorithms applied to a restless problem, so they are expected to lose
to Whittle — and demonstrating exactly that, with CIs, is a result, not a failure.

### 10.3 `WhittleIndexScheduler` — the technical differentiator

Model each channel as a **Gilbert–Elliott** two-state Markov chain (busy/idle) with
transition probabilities `(p01, p11)` estimated online from observed transitions. Because
a channel is observed only when visited, the sufficient statistic is the scalar belief
`ω_b = P(busy)`, which evolves under the passive action as `ω ← ω·p11 + (1−ω)·p01`.

For the **positively correlated** case (`p11 ≥ p01`), Liu & Zhao (2010) give the closed-form
Whittle index and prove indexability; the general case is solved numerically by bisection
on the passive subsidy `λ`, with an explicit **indexability check** (the passive set must be
monotone non-decreasing in `λ`) asserted in tests and reported per channel. The derivation,
the value-function fixed point, and the K-out-of-N extension (we activate `K = 4` adjacent
arms per slot, not one) go in `docs/theory.md`.

Why this is the strongest baseline to beat: it is the provably near-optimal policy for
exactly this structure (restless arms, partial observation, per-slot budget), it needs no
training, and it runs in microseconds. If a learned agent beats Whittle we have a real
result; if it merely beats `SequentialSweep` we have a weak one. **Both numbers get
reported.**

---

## 11. Learned schedulers

### 11.1 `SequencePredictorScheduler`

Predict `P(X[b, t+1] = 1 | history)` for **all** `B` channels including unobserved ones,
then tune to the legal `K`-window maximising `Σ_b p̂_b · ŵ_b` (threat-weighted expected
detections) minus retune cost.

* **Input:** sliding window `W = 128` slots × `B` channels × 4 planes (visit mask, hit
  mask, SNR estimate, staleness) → `(4, B, W)`.
* **Output:** `B`-dim probability vector.
* **Architectures compared** (identical input, loss and parameter budget ≈ 200 k):
  * **GRU** over time with channel as feature dim — cheapest, strong on periodic signals.
  * **Dilated 1-D TCN** (dilations 1,2,4,…,64 → receptive field 128).
  * **Small Transformer encoder** with *separate* channel and time positional encodings —
    the only one that can attend across channels to learn FrequencyAgile hop-set structure.
* **Loss:** masked focal BCE. Masked because at train time only visited channels carry a
  label. Focal (`γ = 2`) because occupancy is ~2–5 % positive and plain BCE collapses to
  "always idle".

`W` is a swept ablation: a 4 s scan period at `dt = 1 ms` is 4000 slots, far beyond any
practical receptive field, so the predictor learns *short-horizon* structure (burst
persistence, hop-set correlation, beam dwell continuation) while §12 handles the long
periods analytically. Stating this division of labour up front is more honest than
pretending a 128-slot window can learn a 4 s period.

### 11.2 Privileged teacher → student distillation *(training-time only)*

In simulation we hold the full `X[b,t]`, so:

1. Train a **teacher** on the complete tensor (privileged information; it sees everything).
2. Train the **student** on observation history alone with
   `L = L_masked_focal(student, observed labels) + λ_KD · KL(student ‖ teacher)` over **all**
   `B` channels — the teacher supplies soft labels exactly where the student has none.

This is Hinton-style distillation used as learning-from-privileged-information (Vapnik &
Izmailov). It is **training-time only**; the deployed student consumes nothing but
observations. Enforced structurally: the teacher lives behind `PrivilegedAccess`, which
raises `RuntimeError` if entered while `eval_mode` is set. Stated in the README, in the
docstring and on the results plot — no leakage claim goes unlabelled.

### 11.3 RL: `DQNScheduler`, `PPOScheduler`

**Decision — from-scratch, not Stable-Baselines3 by default.** Rationale: (a) this machine
is Python 3.13, and the SB3/Gymnasium pin matrix on 3.13 is the single most likely thing to
break a live judge reproduction; (b) we need **action masking**, which SB3 offers only via
contrib `MaskablePPO`; (c) we need bit-level determinism. So: a compact PPO (GAE-λ, clipped
objective, masked categorical) and Double-DQN (duelling head, prioritised replay) at ~350
lines each, plus a `SmartScanGymEnv` that *is* a valid `gymnasium.Env` so SB3 or RLlib can
be dropped in by anyone who wants them. A Gymnasium conformance test runs in CI.

* **Observation:** `(B, F)` map + `G` globals. Default encoder is a small 1-D CNN over the
  channel axis — weight sharing across channels is the correct inductive bias, since
  channel 7 and channel 44 obey the same physics — followed by an MLP. Flattened-MLP is an
  ablation.
* **Action:** `Discrete(B)` with a boolean mask; masked logits set to `−inf` **before** the
  softmax (after would leak probability mass).

### 11.4 Reward

```
r_t =  + w1 · Σ_e threat_priority(e)    for each emitter newly intercepted this slot
       + w2 · novelty_bonus             first-ever detection of an emitter ID
       + w3 · n_reconfirmed             re-confirming an already-tracked emitter
       − w4 · retune_cost               1 if a_t != a_{t−1}
       − w5 · dwell_on_known_interferer
       − w6 · max_b(time_since_visit[b]) / T     coverage staleness
```

Defaults `w = (10.0, 5.0, 0.5, 0.2, 1.0, 0.3)`, all in config, all swept in
`eval/ablation.py` (one-at-a-time × 5 levels × 10 seeds) with the sensitivity table
reported. `w6` uses the **max** over channels, not the mean — that is what makes it a
genuine coverage guarantee rather than an average that can be gamed by starving a few
channels.

Note `w5` must key off the *agent's own* estimate of "interferer": ground-truth threat is
not observable. The proxy is "channel with high occupancy that keeps yielding detections
but never yields novelty" — an emergent quantity carried in belief feature 11.

### 11.5 `HybridScheduler`

Predictor output `p̂ ∈ [0,1]^B` is appended as a 13th feature plane to the RL observation
(predictor frozen; no gradient flows back). Reported honestly: the hybrid must beat **both**
parents on the same seeds, or we say that it did not.

**It did not.** On MEDIUM it reaches +119 % TWIR against the tuned sweep, above
`ppo` (−41 %) but below `predictor` (+159 %), so it does not beat both parents even
on the metric that flatters it — and the log-rank puts its hard-target hazard at
0.777 (p = 8.6e-03), significantly *worse* than the sweep. On EASY and HARD it does
not schedule at all: under the greedy argmax it tunes to one channel for the whole
episode. See §21-L for the diagnosis and for why its training return is not
evidence to the contrary.

---

## 12. Scan-on-scan analysis

### 12.1 Deterministic coincidence

For receiver sweep period `Tr` with window `wr` (time our IBW covers the emitter's channel
per sweep) and emitter scan period `Te` with illumination window `we = (θ_bw/360)·Te`, an
intercept requires the two windows to overlap. Writing the relative phase
`φ_n = (n·Tr) mod Te`:

* **Incommensurate `Tr/Te`** (irrational): `φ_n` is equidistributed on `[0, Te)` (Weyl), so
  intercept is certain, with `E[TTI] ≈ Tr·Te / (wr + we)`. The *distribution* of gaps is
  governed by the **three-distance theorem** — gaps take at most three distinct values, set
  by the continued-fraction convergents of `Tr/Te`.
* **Commensurate `Tr/Te = p/q`** (small `q`): `φ_n` takes only `q` distinct values. If none
  lands within `wr + we`, **`POI = 0` for all time** — the synchronism / blindness
  pathology. A uniform sweep is exactly the policy most likely to fall into it.
* `POI(t)` is therefore a staircase, not `1 − exp(−t/τ)`. Both are implemented, and the
  exponential is plotted as the commonly-assumed and often-wrong approximation.

References: Self & Smith, *Intercept time and its prediction*, IEE Proc. F 132(4), 1985;
Clarkson & Pollington, *Performance limits of sensor-scheduling strategies in electronic
support*, IEEE Trans. AES 43(2), 2007; Wiley, *ELINT*, Artech House, 2006.

**`CoprimeSweepScheduler`** picks `Tr` so that `Tr/T̂e` is *badly approximable* — bounded
away from every small rational. Concretely it scales `Tr` toward the golden ratio times the
detected periods (the golden ratio maximises the minimum phase gap; the same argument as
golden-ratio / Weyl sampling) and actively avoids the detected `T̂e` set.

**`PhaseLockedScheduler`** once `T̂e` clears a confidence threshold: predict the next beam
arrival `t̂ = t_last + T̂e` and park the receiver on that channel from `t̂ − guard`, where
`guard = 3σ_est + t_settle`. Between predicted arrivals it falls back to the Whittle policy
so the receiver is never idle.

### 12.2 Online period estimation from sparse, irregular hits

Two estimators, compared head-to-head:

1. **Lomb–Scargle** (`scipy.signal.lombscargle`) — the correct tool for unevenly sampled
   data. **Critical correction:** our hit times are the product of emitter activity *and*
   our own visit schedule, so a raw periodogram peaks at **our sweep period**. We therefore
   also compute the periodogram of the visit indicator (the spectral window) and score
   candidate periods by hit-power *not explained by* window-power. Without this the
   estimator confidently reports its own tail. This is the easiest thing in the module to
   get wrong, and it gets a dedicated test.
2. **Histogram-of-differences / CDIF–SDIF** — cumulative and sequential difference
   histograms with a decreasing detection threshold; the classical PRI-deinterleaving
   technique (Mardia 1989; Milojević & Popović 1992) applied at scan-period timescale.
   Robust to missing arrivals via subharmonic checking, weak against heavy jitter.

Validation metric, as specified: **average intercept time error**
`mean |t_predicted_arrival − t_true_arrival|` over all predicted arrivals, plus relative
period error `|T̂e − Te| / Te`.

> ⚠️ **Open issue — needs your call (§17-A).** Acceptance test 4 asks for a 4.0 s scan
> period recovered to within 2 %, but the default episode is `T = 10 s` = **2.5
> revolutions**. Two or three arrivals cannot support a 2 % estimate under jitter and
> missed looks. Proposal: run estimator validation on a dedicated
> `configs/scan_on_scan.yaml` with `episode_s = 120` (30 revolutions; at `dt = 1 ms` that
> is 120 k slots and still ≈ 2 s of CPU, because the environment is precomputed and
> vectorised). The 10 s tiers stay exactly as they are for scheduler benchmarking.

---

## 13. Determinism and reproducibility

* **One root seed → a tree of named substreams.** `numpy.random.SeedSequence(root)` is
  spawned through a *fixed registry* of stream names — `("scenario",)`, `("emitter", i)`,
  `("receiver",)`, `("agent",)`, `("eval_bootstrap",)`. Consequence: adding a 16th emitter
  does **not** perturb the first 15, and changing the agent does not change the world.
  Worth stating explicitly, because the naive `np.random.seed()` approach silently breaks
  every ablation you will want to run.
* **Byte-identical tensors** (acceptance §1): `tests/test_reproducibility.py` hashes
  `blake2b(X ‖ duty ‖ SNR ‖ E)` for a fixed seed and compares against a checked-in golden
  digest, on every tier.
* **Torch:** `use_deterministic_algorithms(True)`, `manual_seed`, `cudnn.benchmark = False`,
  and `torch.set_num_threads(1)` for reproduce runs — CPU reduction order varies with
  thread count and will silently move the fifth decimal.
* **Config hash.** Every artefact (`metrics.json`, checkpoint, figure) embeds the blake2b of
  the *resolved* config plus `git rev-parse HEAD`, so no number is ever orphaned from the
  settings that produced it.
* `make reproduce` regenerates every headline number in the README from scratch.

---

## 14. Configuration system

Normative field list: **[`docs/config_schema.md`](config_schema.md)**.

* YAML → **Pydantic v2** models (`smartscan/config.py`) are the single source of truth for
  types, units, ranges and cross-field validation. Unknown keys are a hard error
  (`extra="forbid"`) — a typo'd hyper-parameter that silently does nothing is the classic
  way to lose a day.
* Layering: `base.yaml` (every default, annotated) ← tier overlay (`easy/medium/hard`) ←
  `--set a.b.c=v` CLI overrides. Resolution order is deterministic, and the *resolved*
  config is written beside the results.
* Derived values (`n_slots`, channel widths, noise floor) are **computed, never
  configured**, so they cannot disagree with their inputs.
* Units live in the field name (`_hz`, `_s`, `_db`, `_dbm`, `_slots`). No bare numbers.
* Canonical location is repo-root `configs/` so `--config configs/medium.yaml` works from a
  checkout (acceptance §2); `pyproject.toml` force-includes it into the wheel at
  `smartscan/configs/`, and the CLI falls back to the packaged copy for `pip install` users.

---

## 15. Performance budget (EASY tier < 10 min CPU — the live-demo constraint)

The load-bearing decision: **ground truth is precomputed once per episode, vectorised over
the whole time axis**, after which stepping is an O(K) array slice. A per-slot Python loop
over emitters would be 10⁴ slots × 30 emitters = 3 × 10⁵ calls per episode and would make
CPU RL training impossible.

| Stage | Budget | Note |
|---|---|---|
| Scenario gen + tensors (EASY, `B=64, T=10⁴`) | < 150 ms | fully vectorised; ≈ 4.5 MB |
| One episode, analytic agent (sweep / bandit / Whittle) | < 200 ms | ~50 k episodes/hr |
| 30-seed benchmark, all analytic agents | < 60 s | |
| PPO train, EASY, 200 k steps, CPU | < 5 min | small CNN+MLP, 8 parallel envs |
| Predictor train, EASY, CPU | < 3 min | GRU, 20 epochs |
| **`make reproduce-easy` total** | **< 10 min** | the number judges will actually run |

HARD-tier RL and the full three-architecture predictor sweep are GPU/overnight work and
ship as checkpoints; `make reproduce` consumes them without retraining unless `--retrain`
is passed.

---

## 16. Evaluation protocol

**Never a single seed.** Every headline number: 30 seeds, **paired** across agents (same
scenario, different policy), 10 000-resample **cluster bootstrap over seeds**, percentile
95 % CIs, reported on the *paired difference* against `SequentialSweep` — paired
differencing removes scenario variance and is what makes a 25 % claim defensible.

| Metric | Definition |
|---|---|
| **TTFI** | median over emitters and seeds of time to first *true* intercept (false alarms excluded) |
| **TWIR (rate)** | `Σ_e π_e · (detected slots_e / active slots_e) / Σ_e π_e` — headline |
| **TWIR (coverage)** | `Σ_e π_e · 1[ever intercepted] / Σ_e π_e` — secondary |
| **Revisit staleness** | mean and **max** over `b,t` of `time_since_visit` |
| **Waste fraction** | dwell slots spent on interferers ÷ total dwell |
| **Pop-up latency** | `t_detect − t_first_active` for `t_first > 0.6T` emitters (acceptance §5) |
| **Discovery curve** | unique emitters found vs `t`, and its area |
| **FA burden** | declared hits carrying `pfa_flag` ÷ total declared hits |
| **Period error** | `|T̂e − Te| / Te`; plus **average intercept time error** (§12.2) |

**TTFI is compared by log-rank, not by the paired bootstrap.** A never-intercepted
emitter has a time to first intercept of `+inf`, and the bootstrap drops non-finite
pairs — which removes each policy's *failures* and scores it only on the runs where
it succeeded, flattering the worst performers most. On MEDIUM that once left 3 of
30 seeds for some agents while every other metric kept all 30. Two guards now apply:

* a comparison backed by fewer than `min_paired_seeds = 10` finite pairs is
  **withheld and recorded as withheld**, so an absent row reads as "not tested"
  rather than "tested and did not win";
* hard-class TTFI is judged by a Mantel-Haenszel **log-rank test**, which keeps the
  never-intercepted as right-censored observations contributing to the risk set.

The test is calibrated before use: 400 null replicates give P(p<0.01)=0.013,
P(p<0.05)=0.068, P(p<0.10)=0.113 with a near-uniform p-distribution, and the
degenerate case — neither policy ever intercepts — returns p=1.0 rather than
raising. Censoring is reported **per group**, because a policy censoring more than
the baseline is itself the signal.

Emitted as `metrics.json` (schema-versioned), a markdown table, and Plotly figures.

---

## 17. Risks and open issues

**A. Acceptance §4 (4.0 s period to 2 %) is not reachable inside a 10 s episode** — 2.5
revolutions. Proposal in §12.2: a dedicated 120 s estimator-validation config. *Needs your
approval.*

**B. Acceptance §3 (≥ 25 % TTFI, ≥ 15 % TWIR over sweep) is a target, not a guarantee.**
Expected ordering: `PhaseLocked / Whittle > Hybrid ≈ PPO > Predictor > TS / UCB > Sweep >
Random`. The near-certain winners are Whittle and PhaseLocked — no training, and they
exploit the actual structure. PPO needs a training budget to get there. If RL
underperforms we report that, and the *claim* rests on Whittle/PhaseLocked, both of which
are adaptive closed-loop policies in the PS's sense and neither of which is the open-loop
strawman.

**C. `SequentialSweep` is a stronger baseline than it looks — CONFIRMED, twice over.**
First as predicted: on EASY it is near-optimal and nothing beats it. Second, and less
comfortably: the sweep's dwell length turned out to be a free parameter worth 2x, and
leaving it at the textbook one slot would have made the incumbent a strawman. It is now
**tuned** (`dwell_slots = 3`, swept over {1,2,3,5,8,12,20}) and every headline comparison
is against the tuned version. See §21-D.

**D. Python 3.13 + SB3 / Gymnasium pin risk** → mitigated by the from-scratch RL decision
(§11.3), with SB3 kept as an optional extra.

**E. Reward hacking on `w3` (re-confirmation).** An agent can farm re-confirmations by
parking on one loud emitter. Mitigated by the `w6` max-staleness term and a per-emitter
diminishing-returns cap on `w3`; *verified* by the waste-fraction metric rather than
assumed.

**F. Sim-to-real gap.** The prototype's claim is scheduling-policy quality, not RF realism.
`docs/hardware_roadmap.md` states plainly what changes on real hardware (non-Gaussian
interference, AGC transients, real `t_settle` of 0.1–10 ms depending on device, LO drift)
and which numbers would move.

---

## 18. Acceptance-test traceability

| # | Requirement | Where satisfied | Test |
|---|---|---|---|
| 1 | pytest green; reproducibility | §13 | `tests/test_reproducibility.py` (golden blake2b) |
| 2 | CLI runs `sequential` + `ppo` → metrics JSON | `cli.py`, §14 | `tests/test_cli_smoke.py` |
| 3 | ≥ 25 % TTFI, ≥ 15 % TWIR, 30 seeds, 95 % CI | §16, risk B | `eval/benchmark.py`, `tests/test_acceptance.py` |
| 4 | 4.0 s period within 2 % | §12.2 + **open issue A** | `tests/test_scan_on_scan.py` |
| 5 | Pop-up detected faster than sweep | §7.5 HARD, §16 | `tests/test_popup.py` |

Coverage gate: **≥ 80 % on `env/` and `analysis/`**, enforced in CI via
`--cov=smartscan/env --cov=smartscan/analysis --cov-fail-under=80`.

---

## 19. Build order

1. `env/types.py`, `config.py`, the configs, `propagation.py` (+ detection tests against
   published Pd/Pfa curves) — **the foundation everything else is measured against**.
2. `emitters.py`, `rf_environment.py`, `receiver.py`, the HAL, the reproducibility golden test.
3. `belief.py`, baselines, bandits → first end-to-end benchmark numbers.
4. `analysis/` (scan-on-scan, estimators, metrics) + `whittle.py` → the theory results.
5. `predictors.py` (+ distillation), `rl_agents.py`, `hybrid.py`.
6. `eval/` sweeps, notebooks, docs, dashboard.

Stages 1–4 alone satisfy acceptance 1, 2, 4 and very likely 3 and 5. Stages 5–6 are the
upside. This ordering means there is a demonstrable system at every checkpoint.

---

## 20. References

Verified citations with an annotated "gap we fill" table:
[`docs/related_work.md`](related_work.md). Derivations: [`docs/theory.md`](theory.md).

The eight that matter most:

1. A. G. Self, B. G. Smith, "Intercept time and its prediction," *IEE Proc. F* 132(4), 215-222, 1985 -- the foundational analytic treatment of interception as time coincidence.
2. I. V. L. Clarkson, A. D. Pollington, "Performance limits of sensor-scheduling strategies in electronic support," *IEEE Trans. AES* 43(2), 645-650, 2007 -- bounds on what any *periodic* scheduler can achieve.
3. R. Winsor, E. Hughes, "Optimisation and evaluation of receiver search strategies for electronic support," *IET Radar, Sonar & Navigation*, 2011 -- closest prior work; assumes a **known threat list**, which the problem statement removes.
4. P. Whittle, "Restless bandits: activity allocation in a changing world," *J. Applied Probability* 25A, 1988 -- the index policy our classical tier implements.
5. K. Liu, Q. Zhao, "Indexability of restless bandit problems and optimality of Whittle index for dynamic multichannel access," *IEEE Trans. Inf. Theory* 56(11), 2010.
6. S. Wang et al., "Deep reinforcement learning for dynamic multichannel access in wireless networks," *IEEE Trans. CCN* 4(2), 2018 -- closest RL prior work; throughput reward, not threat value.
7. US Patent 6,020,842, "ESM duty dithering scheme for improved probability of intercept at low ESM utilization" -- independent validation of the blind-zone problem, and of dithering as the fix. We replace random dither with a golden-ratio Weyl sequence, which is a deterministic guarantee rather than an expectation.
8. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., 2014, ch. 6 -- detection statistics.

Also: Papadimitriou & Tsitsiklis (1999) on restless-bandit intractability;
Kaelbling, Littman & Cassandra (1998) for the POMDP formalisation; Wiley (2006)
and Haigh & Andrusenko (2021) as the domain texts; Lomb (1976) / Scargle (1982)
and Milojevic & Popovic (1992) for the two period estimators; Gunn et al. (2026)
for the Turing Synthetic Radar Dataset named in the brief.

---

## 21. What changed during implementation

The design above was approved before any code existed. Five things changed once
it met measurement. Each is recorded here with the evidence that forced it,
because a design document that is quietly edited to match the outcome is worth
nothing.

### A. Scenarios are authored in SNR space, not (EIRP, range)

**Symptom.** Circular-scan radars were detectable through their **sidelobes for
50 % of the episode** — 5024 of 10 000 slots for a 2.34 s scanner whose main beam
illuminates us for 31 ms per revolution (1.3 %).

**Cause.** Sampling EIRP over 30 dB and range over 23 dB spans 53 dB of received
power. A loud, close scanner then clears the detection threshold even 30 dB down
its sidelobes, at which point it is not a scan-on-scan target at all — it is a
continuous emitter, and the entire problem the brief poses has been calibrated
away.

**Fix.** `emitters.*.snr_db` gives the **main-lobe SNR** the receiver should see,
and `smartscan/env/calibration.py` back-solves the range that produces it. The
link budget is untouched and still fully physical; we are choosing where in its
domain the scenario sits, exactly as an exercise designer chooses engagement
geometry. Sidelobes were tightened to −45…−30 dB, which places sidelobe
intercepts at a few per episode — "possible but weak", as the brief asks.

**After.** Scanners are detectable in 1.5–6.9 % of slots, matching beam dwell.

### B. Detection is calibrated per emitter regime, and `Pd` had a NaN at the floor

The Swerling I closed form is an `∞ · 0` indeterminate as `χ → 0`:
`(1 + 1/(Nχ))^(N−1)` overflows float64 while the incomplete gamma underflows to
exactly zero. It produced NaN **precisely at the SNR floor**, which is where most
cells live. Now evaluated in log space; both the exact and large-`N` branches
tend correctly to `Pfa` (`test_pd_at_snr_floor_tends_to_pfa`).

### C. `B = 64` made the benchmark unable to tell policies apart

**Symptom.** Every scheduler scored an identical 43 ms median TTFI on the hard
emitter classes.

**Cause.** The detectable illumination window is ~2.4× the 3 dB beamwidth once
the SNR margin is accounted for (`docs/theory.md` §4), giving 100–600 ms. A
`B = 64` sweep takes 48 ms. A sweep shorter than every illumination window
**cannot miss**, so there is nothing to schedule.

**Fix.** `B = 128` (`K/B = 1/32`, sweep 96 ms). TTFI on scanning emitters moved
from 0.043 s to 3.06 s for the tuned sweep, and the policies separated.

### D. The baseline needed tuning before it could be beaten honestly

`dwell_slots` controls how long the sweep holds a window before stepping. Total
observation per channel is `T·d/(N_win·(d + t_settle))` — 208 slots at `d = 1`
but 375 at `d = 3`. Swept over {1,2,3,5,8,12,20}; `d = 3` is best on MEDIUM and
is now the default. Every headline number is against the tuned sweep.

### E. Two metrics were measuring the wrong thing

* **TTFI over all emitters is saturated.** CW, pulsed and interferer emitters are
  found on the first pass by any policy, so the median cannot discriminate. The
  headline is now `ttfi_hard_median_s`, restricted to `HARD_CLASSES`
  (circular, sector, agile-beam).
* **Pop-up latency conflated two failures.** Folding a never-detected pop-up in
  as "the whole episode" made every scheduler score ~5.0 s — the arithmetic of
  one found and one missed. Detection **rate** and latency are now separate, and
  pop-ups that are never physically interceptable are excluded from the
  denominator. Measured on HARD: UCB1 finds 91 % of reachable pop-ups against the
  sweep's 73 %, while the sweep is marginally faster on the ones it does find.
  Both halves are reported.

### F. The period estimators were reading harmonics, and SDIF never fired

**Symptom.** Lomb-Scargle returned 1.000 s and 2.000 s for a 4.0 s scanner with
high confidence (`Te/4` and `Te/2`); SDIF returned "no period" on every episode.

**Cause.** The raw hit series from a scanning radar is a **pulse train**, not a
point process: one beam pass is a contiguous run of ~100 hit slots. Its spectrum
has comparable power at every harmonic up to `1/we`, so the argmax lands
wherever. For SDIF the first-order differences were dominated by the one-slot
gaps *inside* a pass, and the classical Milojević–Popović threshold — calibrated
for PRI deinterleaving with thousands of pulses — evaluated to ~17 against ~30
scan arrivals and never fired.

**Fix.** `cluster_arrivals` collapses each pass into one arrival before
estimation (which is also what a real ES system does when forming scan marks);
a harmonic check prefers the fundamental; the SDIF threshold is now a Poisson
tail against a uniform-arrival null.

**After.** 4.0 s recovered to **0.16 %** (target 2 %), from 75 % error. Both
estimators now work and can be compared: LS 0.26 % median, SDIF 0.41 %, LS
resolving more cases. The residual 50 % errors are all `SectorScanRadar`, where
a bidirectional sweep genuinely produces two arrivals per frame — a real
ambiguity, documented rather than papered over.

### G. The measured ordering versus the predicted one

Risk B predicted `PhaseLocked/Whittle > Hybrid ~ PPO > Predictor > TS/UCB >
Sweep > Random`. Measured at 30 paired seeds on threat-weighted interception
ratio:

    thompson (+96.6%) > phase_locked (+76.5%) > whittle (+67.3%) > SWEEP
      > ucb1 (-8.6%) > coprime_sweep (-38.7%) ~ ppo (-41.3%) ~ random (-43.0%)

Two corrections to the prediction, both worth stating:

* **Thompson sampling beat the restless-bandit index.** Predicted to lose,
  because it is a rested algorithm on a restless problem. It wins on TWIR by
  exploiting harder -- and pays for it in coverage, dropping to 0.733 against the
  sweep's 0.857. On the mission objective as a whole that is a worse trade than
  the ranking alone suggests, which is why coverage is reported beside it.
* **`coprime_sweep` loses on TWIR.** Expected: it is a coverage policy, not an
  exploitation one, and its whole purpose is the blindness result in §12.1
  rather than raw interception ratio. It holds the highest coverage entropy
  (0.998) of any non-sweep policy.

### H. External validation turned out to be possible, and changed the bridge

The design assumed the Turing dataset would be reachable through the Hugging
Face `datasets` loader with columns named something like `toa`/`rf`. It is not.
The live repository is **9021 raw HDF5 files** across three subsets, with a
``(N, 5)`` float array and a separate label vector, and the units are
**microseconds and MHz** — an adapter that forgets either produces a silently
empty tensor with no error.

Rewritten against the real format. Two consequences worth recording:

* The `archive` subset spans **0.36–12 GHz over 9.5 s with up to 88 emitters**,
  which lands on a SmartScan episode almost exactly. External validation is
  therefore a genuine replay, not a rescaled analogy.
* The amplitude mapping was originally a fudge (`amplitude_ref_db`,
  `snr_scale_db`). It is now **physical**: `PA` is read as received power in dBm
  and compared against this receiver's thermal noise in the pulse-detection
  bandwidth, exactly as the simulator does. That puts the median pulse at
  ~13.7 dB SNR without a tuned constant.

Measured: UCB1 leads on interception ratio (+44 % over the sweep) and holds
worst-case staleness at 0.18 s against 0.62 s. Reported separately from the
synthetic tiers, because the binning is an assumption of the bridge and TSRD
carries no threat model.

### I. RL underperforms at the training budget available — as predicted

Risk B above said the near-certain winners were Whittle and PhaseLocked, and
that PPO would need a training budget to compete. At 400 k CPU steps PPO reaches
a return of 192 against the tuned sweep's 214, and its TWIR is 41 % worse. Its
max staleness is the full 10 s episode: it learned to park on one window and
abandon the band. The headline claim therefore rests on
the restless-bandit and scan-on-scan policies, both of which are adaptive
closed-loop schedulers in the PS's sense and neither of which is the open-loop
strawman. The RL result is reported as measured.

### J. The censored analysis inverted the leaderboard

Ranked by threat-weighted interception ratio, `epsilon_greedy` leads at +305 %
over the tuned sweep and `predictor` follows at +159 %. Both figures are real and
survive Holm correction at 30 seeds.

Both are also misleading. TWIR counts interceptions without asking *which* emitter
was intercepted, so a policy can inflate it by concentrating on always-on emitters
and never finding a scanning radar — and that failure is invisible to a mean,
because the emitters it fails on contribute `+inf` and get dropped. The log-rank
test keeps them:

| agent | TWIR | hazard | p | never intercepted |
|---|---|---|---|---|
| `priority_rr` | −7 % | **1.207** | 1.4e-02 | **53** / 146 |
| `whittle` | +67 % | 1.042 | 0.60 | 64 / 146 |
| `phase_locked` | +77 % | 1.028 | 0.73 | 66 / 146 |
| `sequential` | baseline | 1.000 | — | 68 / 146 |
| `hybrid` | +119 % | 0.777 | 8.6e-03 | 94 / 146 |
| `thompson` | +97 % | 0.682 | 1.8e-04 | 102 / 146 |
| `predictor` | +159 % | 0.536 | 1.7e-08 | 112 / 146 |
| `epsilon_greedy` | **+305 %** | **0.513** | **1.2e-08** | **115** / 146 |

The ordering is close to inverted. `epsilon_greedy` misses 115 of 146 scanning and
agile emitters against the sweep's 68. The result therefore rests on `whittle` and
`phase_locked` — the only policies that raise interception ratio while leaving
hard-target hazard within noise of 1.0 — which is where §17 Risk B predicted it
would land, though not for the reason given there.

### K. `coverage_weight` was tuned against the wrong objective, twice

The first sweep reported +29 % TWIR for Whittle at a weight of 2.0 on five seeds.
It did not replicate at twelve (+14.7 %, CI [−26 %, +58 %]) and was retracted.

The retraction fixed the sample size but not the objective: TWIR is the quantity
§21-J shows to be misleading. Re-swept against hard-target hazard on 20 paired
seeds, the shipped default is the peak for both policies —

    whittle       0.0 → 0.828   0.5 → 0.879   **1.0 → 0.981**   2.0 → 0.919   8.0 → 0.891
    phase_locked  0.0 → 0.828   0.5 → 0.905   **1.0 → 0.972**   2.0 → 0.887   8.0 → 0.891

— and falls away on either side, with no cell significant. The knob stays at 1.0.
Recorded because a sweep can be repeated with a better sample size and still be
asking the wrong question.

### L. The hybrid does not learn, and its training return conceals it

Under the greedy argmax used at evaluation, `hybrid_easy` and `hybrid_hard` each
tune to a **single channel** for all 9 998 slots — coverage 0.000, TWIR 0.0000 on
EASY — while reporting training returns of 62.4 and 236.0. Only `hybrid_medium`
behaves. It is not a budget problem: `hybrid_hard` had 3 000 000 steps.

Ruled out before reaching for hyperparameters: the inference path (`observe`
forwards to the predictor's window, `reset` resets it); a layout mismatch
(`flat_features` and the adapter are both channel-major and `_split_obs` matches);
an ordering skew (both paths hold dwells `0..t` when acting at `t+1`); the
predictor itself (varies across channels, std 0.029, observation max |x| 1.59);
and any shared cause (`predictor`, `dqn`, `ppo`, `ucb1` all spread over 38–125
channels on every tier).

What remains is that the policy never forms a stable preference ordering: entropy
falls to 0.362 mid-run and recovers to near-uniform, so the argmax is decided by a
vanishing margin that lands on the same channel every time. `rl.hybrid.entropy_coef`
— separate from `rl.ppo.entropy_coef`, because plain PPO trains fine at 0.01 —
removes the collapse **on EASY only**: 1 → 49/60/23 distinct channels, coverage
0.000 → 1.000/1.000/0.600. On HARD the same change failed — 3 000 000 steps and
2.4 h later the policy still tunes to 1–4 channels at coverage 0.000–0.179,
while its training return *rose* from 236.0 to 253.8. And even where it works,
EASY's TWIR is 0.0028 against the sweep's 0.0176: no longer parked, close to
random. **A partial pathology fix, not a performance result.** The consequence
worth carrying: a hybrid training return is not evidence that the hybrid learned
anything — 236.0 and 253.8 both describe a policy that tunes to one channel.

