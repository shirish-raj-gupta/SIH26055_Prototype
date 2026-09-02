# SmartScan — Configuration Schema (normative)

Status: **DESIGN — awaiting approval.**
Concrete files: [`configs/base.yaml`](../configs/base.yaml) (annotated defaults),
`easy.yaml`, `medium.yaml`, `hard.yaml`, `scan_on_scan.yaml`.
Design rationale: [`docs/architecture.md`](architecture.md) §14.

---

## 0. Conventions

| Rule | Why |
|---|---|
| **Units in the field name** — `_hz _s _db _dbm _km _deg _slots _frac` | no bare numbers; a unit mistake becomes a review-visible typo |
| **`[lo, hi]` = uniform sampling range**; a scalar = fixed value | one syntax for "prior" vs "constant"; `dist:` overrides where needed |
| **Derived values are never configured** — `n_slots`, channel widths, noise floor | they cannot then disagree with their inputs |
| **`extra="forbid"`** — an unknown key is a hard error | a typo'd hyper-parameter that silently does nothing is how you lose a day |
| **YAML floats need a dot and a signed exponent**: `1.0e-3`, `0.5e+9` | PyYAML 1.1 parses bare `1e-4` and `0.5e9` as **strings**; a `StrictFloat` annotation catches it at load, but writing them correctly avoids the trip entirely |
| **Layering** `base.yaml` ← tier overlay ← `--set a.b.c=v` | deterministic resolution order; the *resolved* config is written beside results and hashed into every artefact |

Loader: `smartscan/config.py`, Pydantic v2. The models are the single source of truth —
this document describes them, it does not duplicate them at runtime.

---

## 1. Top-level sections

| Section | Purpose | Consumed by |
|---|---|---|
| `schema_version` | int, currently `1`; mismatch is a hard error | loader |
| `run` | seeding, output, determinism | everything |
| `spectrum` | frequency partition | `env`, `agents` |
| `time` | slot size, episode length | `env` |
| `receiver` | IBW, retune cost, detector | `env/receiver`, `hal` |
| `emitters` | per-class parameter **priors** | `env/emitters`, scenario gen |
| `scenario` | tier, emitter mix, placement | `env/rf_environment` |
| `belief` | posterior, decay, features | `agents/belief` |
| `reward` | RL weights `w1..w6` | `env`, `agents/rl_agents` |
| `agents` | per-scheduler hyper-parameters | `agents/*` |
| `predictor` | supervised model + distillation | `agents/predictors` |
| `rl` | PPO / DQN / hybrid | `agents/rl_agents`, `hybrid` |
| `analysis` | scan-on-scan + estimators | `analysis/*` |
| `eval` | benchmark protocol | `eval/*` |

---

## 2. Field reference

Only non-obvious fields are annotated; `configs/base.yaml` carries a comment on every line.

### 2.1 `run`

| Field | Type | Default | Constraint / note |
|---|---|---|---|
| `name` | str | `base` | output at `{out_dir}/{name}/` |
| `seed` | int | `20260902` | **root** seed; substreams via a fixed `SeedSequence` name registry (architecture §13) |
| `n_seeds` | int | `30` | benchmark uses `seed … seed+n_seeds−1` |
| `device` | enum | `cpu` | `cpu \| cuda` |
| `deterministic` | bool | `true` | sets `torch.use_deterministic_algorithms(True)` |
| `torch_threads` | int ≥ 1 | `1` | **must be 1** for bit-identical CPU reductions |

### 2.2 `spectrum`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `f_start_hz` | float > 0 | `0.5e+9` | |
| `f_stop_hz` | float | `18.0e+9` | **must exceed** `f_start_hz` |
| `n_channels` | int > 0 | `64` | `B` |
| `partition` | enum | `uniform` | `uniform \| log \| explicit` |
| `edges_hz` | list[float] \| null | `null` | **required iff** `partition == explicit`; length `B+1`, strictly increasing, `edges[0] == f_start_hz`, `edges[-1] == f_stop_hz` |

Non-uniform partitioning is a PS requirement and is supported through `explicit` (arbitrary
edges) and `log` (constant fractional bandwidth — the partition a real wideband
channeliser actually produces).

### 2.3 `time`

`dt_s` (default `1.0e-3`) and `episode_s` (default `10.0`). **Derived:**
`n_slots = round(episode_s / dt_s)` — validated to be a positive integer within 1e-9 of an
integer ratio, so a `dt_s` that does not divide `episode_s` is rejected rather than
silently truncated.

### 2.4 `receiver`

| Field | Type | Default | Note |
|---|---|---|---|
| `ibw_channels` | int > 0 | `4` | `K`; **must satisfy `K ≤ n_channels`** |
| `action_space` | enum | `center_index` | `center_index \| window_start`; both yield `B−K+1 = 61` legal actions |
| `mask_illegal_actions` | bool | `true` | edge windows are masked, **not clipped** — clipping would silently over-sample the band edges |
| `t_settle_slots` | int ≥ 0 | `2` | slots lost on **every** retune; the central cost term |
| `straddle_loss_db` | float ≥ 0 | `1.5` | channeliser scalloping; `straddle_enabled` toggles for ablation |
| `sensitivity_dbm` | float | `-75.0` | hard floor: below it, no detection regardless of `Pd` |
| `backend` | enum | `simulated` | `simulated \| soapy`; `soapy` raises `NotImplementedError` **by design** (§9) |

`receiver.detector`

| Field | Type | Default | Note |
|---|---|---|---|
| `type` | enum | `square_law` | only implemented model |
| `pfa` | float in (0,1) | `1.0e-4` | per observed channel per dwell |
| `swerling` | int | `1` | `0` = non-fluctuating (exact `ncx2.sf`); `1` = Rayleigh, closed form |
| `n_integrate` | `auto` \| int ≥ 1 | `auto` | `auto` derives from dwell ÷ `dt_s` |
| `snr_est_sigma_db` | float ≥ 0 | `2.0` | agents receive a **noisy** SNR read, not truth |

### 2.5 `emitters` — parameter priors

`emitters.defaults` applies to every class; a per-class block overrides it. Ranges marked
"PS" are fixed by the problem statement and must not drift:

| Class | PS-fixed field | Range |
|---|---|---|
| `circular_scan` | `scan_period_s` | `[1.0, 12.0]` |
| `circular_scan` | `beamwidth_deg` | `[1.0, 6.0]` |
| `frequency_agile` | `hop_rate_hz` | `[100.0, 10000.0]` |

Consequences worth stating, because they drive the environment design:

* `circular_scan` beam dwell = `(beamwidth_deg / 360) · scan_period_s` → **11 ms at 4 s /
  1°**, i.e. ~11 slots of opportunity every 4000 slots. This is the needle.
* `hop_rate_hz > 1/dt_s = 1000` means **sub-slot hopping** — up to 10 hops inside one slot.
  This is why the environment carries `duty[b,t] ∈ [0,1]` and not only a binary `X`
  (architecture §6).
* `interferer` is deliberately given the loudest `eirp_dbm` and the shortest `range_km`, so
  it is the single most tempting target for any detection-rate-maximising policy. That is
  its entire purpose.

`detection_mode` (`energy` \| `pulse`) is a per-class field, not a receiver setting —
see architecture §6.

### 2.6 `scenario`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `difficulty` | enum | `easy` | `easy \| medium \| hard` |
| `n_emitters` | int > 0 | `5` | **`sum(mix.values()) == n_emitters`** (cross-field validator) |
| `mix` | dict[class → int ≥ 0] | see tiers | keys must be exactly the 8 class names |
| `n_popup` | int ≥ 0 | `0` | **must be ≤ `n_emitters`** |
| `popup_start_frac` | float in (0,1) | `0.6` | pop-ups first active at `t > frac · T` |
| `placement` | enum | `poisson_disk` | `poisson_disk \| uniform` |
| `min_channel_separation` | int ≥ 0 | `2` | prevents scenarios solvable by one lucky window |

### 2.7 `belief`

`alpha_prior`/`beta_prior` default to `Beta(1,1)` — an honest "I don't know" over an
unsurveyed band. `decay_half_life_slots = 2000` (2 s) sets how fast unvisited channels
relax toward that prior; it is the single most influential belief hyper-parameter and is
swept in the ablation. `n_features = 12` and `n_global_features = 5` are **fixed by code**
and appear in config only so that a mismatch fails loudly at load rather than as a shape
error 40 minutes into training.

### 2.8 `reward`

`w1 … w6` map one-to-one onto the PS's reward specification (architecture §11.4). Two
non-obvious choices:

* `w6` penalises **`max_b` staleness, not mean** — a mean can be gamed by starving a few
  channels; a max cannot.
* `reconfirm_cap_per_emitter: 20` caps `w3` per emitter. Without it, parking on one loud
  emitter farms re-confirmation reward indefinitely (risk E).

All six are swept one-at-a-time × 5 levels × 10 seeds in `eval/ablation.py`, and the
sensitivity table is reported as required.

### 2.9 `agents`

One block per scheduler. The two that carry design weight:

* `priority_round_robin.prior_wrong_frac: 0.4` — the PS asks for pre-mission data that is
  wrong 40 % of the time, so this baseline demonstrates *graceful degradation*, not
  competence. `weight_floor` keeps a wrongly-deprioritised channel from never being
  revisited, which is what makes the degradation graceful rather than total.
* `whittle.check_indexability: true` — the Whittle index is only valid if the passive set
  is monotone in the subsidy `λ`. We **verify** this numerically per channel and report
  violations rather than assuming it. `use_closed_form` takes the Liu & Zhao (2010) closed
  form when `p11 ≥ p01` and falls back to bisection otherwise.

### 2.10 `predictor`

`arch: gru | tcn | transformer` selects the model; all three share input, loss and a
~200 k parameter budget so the comparison is fair. `loss: masked_focal` with
`focal_gamma: 2.0` — occupancy is 2–5 % positive and plain BCE collapses to "always idle".

`predictor.distillation` is **training-time only**. `lambda_kd` weights the KL to a teacher
that sees the ground-truth tensor. The deployed student consumes observations alone; this
is enforced by the `PrivilegedAccess` guard (architecture §11.2) and labelled in the README
and on the results plot.

### 2.11 `rl`

`implementation: from_scratch` is the default — on Python 3.13 the SB3/Gymnasium pin matrix
is the likeliest thing to break a live judge reproduction, and we need action masking plus
bit-level determinism (architecture §11.3). `implementation: sb3` is a supported optional
extra. `encoder: conv1d` shares weights across the channel axis, because channel 7 and
channel 44 obey the same physics.

### 2.12 `analysis`

`estimators.deconvolve_window: true` is the load-bearing flag in the whole module: hit
times are the product of emitter activity **and our own visit schedule**, so a raw
Lomb–Scargle periodogram peaks at *our* sweep period. Setting this false is supported only
so the failure can be demonstrated in the notebook.

### 2.13 `eval`

`paired: true` and `baseline_agent: sequential`: every headline number is a **paired
difference** vs `SequentialSweep` over identical scenarios, with a 10 000-resample cluster
bootstrap over seeds and percentile 95 % CIs. Pairing removes scenario variance and is what
makes the ≥ 25 % claim defensible on 30 seeds.

---

## 3. Cross-field validators (fail at load, not at hour two)

1. `spectrum.f_stop_hz > spectrum.f_start_hz`
2. `partition == "explicit"` ⟺ `edges_hz is not None`; length `B+1`; strictly increasing; endpoints match
3. `receiver.ibw_channels ≤ spectrum.n_channels`
4. `episode_s / dt_s` is an integer within 1e-9 → `n_slots`
5. `sum(scenario.mix.values()) == scenario.n_emitters`
6. `scenario.n_popup ≤ scenario.n_emitters`
7. `0 < detector.pfa < 1`; `detector.swerling ∈ {0, 1}`
8. every `[lo, hi]` range satisfies `lo ≤ hi`
9. `belief.n_features` / `n_global_features` match the code constants
10. `rl.implementation == "sb3"` → the `sb3` extra is importable, else a clear install message
11. `receiver.backend == "soapy"` → `NotImplementedError` with a pointer to `docs/hardware_roadmap.md`
12. `eval.baseline_agent ∈ eval.agents`
13. `emitters.circular_scan.scan_period_s` ⊂ `[1, 12]` and `beamwidth_deg` ⊂ `[1, 6]` (PS-fixed)
14. `emitters.frequency_agile.hop_rate_hz` ⊂ `[100, 10000]` (PS-fixed)
15. `schema_version == 1`

---

## 4. The five shipped configs

| File | Purpose | `episode_s` | `n_emitters` |
|---|---|---|---|
| `base.yaml` | annotated defaults; never run directly | 10.0 | 5 |
| `easy.yaml` | 5 static emitters; the < 10 min live-demo path | 10.0 | 5 |
| `medium.yaml` | **headline benchmark**, acceptance test 3 | 10.0 | 15 |
| `hard.yaml` | 30 emitters, 5 interferers, 2 pop-ups; acceptance test 5 | 10.0 | 30 |
| `scan_on_scan.yaml` | estimator validation, acceptance test 4 — **⚠ needs approval** | **120.0** | 6 |

`scan_on_scan.yaml` is the one place the design departs from the brief. Acceptance test 4
asks for a 4.0 s scan period recovered to within 2 %, but a 10 s episode contains 2.5
revolutions — two or three beam arrivals, which no estimator can turn into a 2 % figure
under jitter and missed looks. The proposal is a dedicated 120 s (30-revolution) config for
*estimator validation only*; the 10 s tiers are untouched for scheduler benchmarking, and
the extra cost is ~2 s of CPU and ~55 MB of tensors. See architecture §17-A.
