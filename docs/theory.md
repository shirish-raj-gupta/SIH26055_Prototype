# SmartScan — Theory

Derivations behind the two results that carry the submission: the **restless
bandit / Whittle index** formulation of receiver scheduling, and the
**deterministic scan-on-scan coincidence** analysis. Both are implemented and
tested; the tests named below are the executable form of these arguments.

---

## 1. Why a restless bandit, and not a bandit

At each slot the receiver selects a contiguous window of `K` of `B` channels.
Write `x_b(t) ∈ {0,1}` for channel occupancy. Two structural facts:

1. **Partial observability.** Only the `K` channels in the tuned window are
   observed. For the rest we hold a belief, not a state.
2. **Restlessness.** `x_b(t)` evolves whether or not we observe it. A scanning
   radar's beam continues to turn while we look elsewhere.

Classical MAB regret guarantees (UCB1, Thompson sampling) assume *rested* arms —
state frozen unless pulled. That assumption is false here, which is why those
policies appear in the benchmark as instructive baselines rather than as the
proposed solution. The correct object is Whittle's **restless bandit**
(Whittle, *J. Appl. Prob.* 1988).

### 1.1 The Gilbert–Elliott channel

Model each channel as a two-state Markov chain:

```
p01 = P(busy at t+1 | idle at t)
p11 = P(busy at t+1 | busy at t)
```

Because a channel is observed only when visited, the sufficient statistic is the
scalar belief `ω_b = P(x_b = 1)`. Under the *passive* action it evolves by the
one-step predictor

```
T(ω) = ω·p11 + (1 − ω)·p01 = p01 + ω·(p11 − p01)          (1)
```

with fixed point the stationary probability `ω₀ = p01 / (1 + p01 − p11)` and
second eigenvalue `μ = p11 − p01`. Iterating (1):

```
T^k(ω) = ω₀ + μ^k · (ω − ω₀)                               (2)
```

Equation (2) is also the estimator's workhorse — see §1.4.

### 1.2 The Lagrangian relaxation

Whittle relaxes the hard "exactly `K` active per slot" constraint to one that
holds *on average*, and prices activity with a Lagrange multiplier `λ` — a
**subsidy for passivity**. The `B`-armed problem then decouples into `B`
independent single-arm problems. For one arm with discount `β`:

```
V_λ(ω) = max{  ω + β·[ ω·V_λ(p11) + (1−ω)·V_λ(p01) ],      (act)
               λ + β·V_λ(T(ω))                              (stay passive)
            }                                               (3)
```

The active branch reflects that acting both earns the expected reward `ω` **and
reveals the true state**, so the next belief is `p11` with probability `ω` and
`p01` otherwise. The passive branch collects the subsidy and propagates the
belief through (1).

`V_λ` is the fixed point of a contraction (modulus `β < 1`), so value iteration
converges geometrically. `smartscan/agents/whittle.py:whittle_index_curve`
solves (3) on a 101-point belief grid, **vectorised across all 64 subsidy levels
simultaneously**: `T` does not depend on `λ`, so the interpolation weights are
computed once and shared. One curve costs ~27 ms.

### 1.3 Indexability, verified rather than assumed

Define the passive set

```
P(λ) = { ω : the passive branch of (3) attains the maximum }
```

The arm is **indexable** iff `P(λ)` is monotone non-decreasing in `λ`: raising
the subsidy can only ever make passivity more attractive. Indexability is not
automatic for restless bandits — it fails for some transition structures — so we
do not assume it. `whittle_index_curve` computes `passive[λ, ω]` on the full
grid and asserts

```
∂/∂λ passive(λ, ω) ≥ 0    for every ω
```

returning an `indexable` flag; `WhittleIndexScheduler` records violating channels
in `indexability_violations` and they are reported, not hidden. Given
indexability the index is well defined:

```
W(ω) = inf{ λ : ω ∈ P(λ) }                                  (4)
```

read off numerically as the smallest subsidy at which each belief enters the
passive set.

`tests/test_analysis.py::test_whittle_index_is_indexable_and_monotone` checks
five regimes — strongly positively correlated, near-i.i.d., **negatively
correlated**, sticky, and the symmetric case. All are indexable, and the index
is monotone in `ω` in each.

**Why numerical and not closed form.** Liu & Zhao (*IEEE Trans. Inf. Theory* 56(11),
2010) prove indexability and give a closed form for the positively correlated
case `p11 ≥ p01`. That case does not cover us: a frequency-agile emitter hopping
off a channel makes it *less* likely to be occupied next slot, i.e. `p11 < p01`,
which is precisely the negatively correlated regime. The numerical route handles
both. Liu & Zhao's structural result is instead used as a **test oracle**: for
identical positively-correlated arms they prove the myopic policy is optimal, so
our index must be monotone increasing in `ω` and induce the same ranking as
`P(busy)` — asserted in
`test_whittle_matches_myopic_for_identical_positively_correlated_channels`.

### 1.4 Estimating `(p01, p11)` from sparse observations

A channel is observed only when visited, so consecutive observations are
separated by arbitrary gaps `k` and the one-step matrix cannot be counted
directly. From (2), the `k`-step transition probabilities are

```
P(busy_{t+k} | busy_t) = ω₀ + (1 − ω₀)·μ^k
P(busy_{t+k} | idle_t) = ω₀ · (1 − μ^k)
```

so the lag-`k` autocovariance of the observed 0/1 sequence decays as
`μ^k · ω₀(1−ω₀)`. We therefore:

1. estimate `ω₀` from the smoothed marginal hit rate `(n_hits + ½)/(n_visits + 1)`;
2. bin the observation pairs by gap, compute the normalised lag product in each
   bin, and regress its log on the mean gap — the slope is `log μ`;
3. invert: `p01 = ω₀(1 − μ)`, `p11 = μ + p01`.

This is a **moment estimator**, not maximum likelihood, and is documented as
such in the code: it is deliberately cheap enough to run at every index refresh
(every 250 slots).

### 1.5 The activation constraint

Textbook Whittle activates the `K` highest-index arms. Our receiver cannot: its
`K` channels must be **contiguous** in frequency. We therefore select

```
a* = argmax_a  Σ_{b ∈ window(a)}  W(ω_b) · threat_b  −  c_retune·1[a ≠ a_{t−1}]
```

a *restriction* of the index policy to the feasible action set, not a departure
from it. `threat_b` is an observation-only proxy (belief feature 11) since true
threat priority is not observable; `c_retune` prices the `t_settle` slots of
blindness a retune costs.

### 1.6 The objective the index is applied to

One correction that mattered in practice. The classical formulation rewards
*access to a busy channel*, so parking on a reliably-busy channel is optimal.
Our mission objective is not throughput but **coverage plus novel detection**,
and it carries an explicit max-staleness penalty (`w6`). Applying the index to a
detection-only proxy made every value policy park on its best window; measured
band coverage fell from 0.89 to 0.73. The policies therefore score

```
score_b = W(ω_b)·threat_b + κ · time_since_visit_b / T
```

with `κ = agents.coverage_weight`. This is not a fudge factor: it is the
objective's own coverage term, and it is swept five ways in `eval/ablation.py`.

---

## 2. Deterministic scan-on-scan coincidence

### 2.1 The coincidence condition

Our receiver revisits a given channel every `Tr` seconds, dwelling `wr`. A
mechanically scanning emitter illuminates us every `Te` seconds for

```
we = (θ_bw / 360) · Te                                       (5)
```

For a 1° beam on a 4 s scan that is **11 ms in every 4000 ms**. This is the
needle.

Let `φ` be the emitter's initial beam phase. The relative phase after `n` sweeps
is `ψ_n = (n·Tr − φ) mod Te`, and an intercept occurs iff the two windows
overlap:

```
ψ_n ∈ [0, we)   ∪   (Te − wr, Te)                            (6)
```

— a target set of measure `wr + we` on a circle of circumference `Te`.

### 2.2 Incommensurate `Tr/Te`: certain intercept

If `Tr/Te` is irrational, `{ψ_n}` is equidistributed on `[0, Te)` by **Weyl's
theorem**. The fraction of sweeps that intercept tends to `(wr + we)/Te`, so

```
E[TTI] ≈ Tr · Te / (wr + we)                                 (7)
```

which is the classical result (Self & Smith, *IEE Proc. F* 132(4), 1985). Clarkson & Pollington (2007)
give the corresponding performance *limits* for any periodic schedule.

Equation (7) gives the *mean* only. The **distribution** of gaps is governed by
the **three-distance theorem** (Steinhaus): for any `α` and any `N`, the points
`{nα mod 1}`, `n = 1..N`, partition the circle into intervals of **at most three
distinct lengths**, determined by the continued-fraction convergents of `α`. The
largest of those gaps bounds the worst-case phase we can fail to sample.

### 2.3 Commensurate `Tr/Te`: the blindness pathology

If `Tr/Te = p/q` in lowest terms, `ψ_n` takes only `q` distinct values, spaced
`Te/q` apart. If

```
Te / q  >  wr + we                                           (8)
```

then for most initial phases **no `n` ever satisfies (6)**: `POI = 0` for all
time. The receiver is not unlucky — it is *provably* blind, forever.

This is not a corner case. A uniform sweep has exactly one revisit period, and
`Tr` is set by hardware (`N_windows × (dwell + t_settle)`), so nothing prevents
it from landing on a small-denominator rational with a common scan period.

Measured, in `tests/test_analysis.py::test_commensurate_sweep_can_be_permanently_blind`:

| `Tr` | `Tr/Te` | commensurate | blind fraction | classical `E[TTI]` (7) |
|---|---|---|---|---|
| 0.096 s | 0.0240 = 1/32 | no | 27 % | 16.5 s |
| 1.000 s | 0.2500 = 1/4 | **yes** | **97.7 %** | 172 s |
| 2.000 s | 0.5000 = 1/2 | **yes** | **98.8 %** | 344 s |

The classical formula reports a finite mean time to intercept for cases where
**98.8 % of initial phases never intercept at all**. `POI(t)` is a staircase,
not `1 − exp(−t/τ)`; the exponential model is implemented as
`poi_exponential` purely so this failure can be plotted beside the truth. It
cannot represent blindness — it reaches 1 for every parameter combination.

### 2.4 `CoprimeSweepScheduler`: the number-theoretic fix

Since blindness follows from having a *single* revisit period commensurate with
`Te`, the fix is to have no such period. The visit order follows a **golden-ratio
Weyl sequence**:

```
a_n = ⌊ N_legal · frac(n · φ) ⌋,      φ = (1 + √5)/2          (9)
```

The golden ratio has continued-fraction expansion `[1; 1, 1, 1, …]` — all partial
quotients 1 — which makes it the **worst-approximable irrational**. By the
three-distance theorem this minimises the largest gap for every `n`, i.e. it is
the low-discrepancy sequence that maximises the minimum phase separation. No
emitter period can stay in lockstep with it.

Measured largest gaps at `N = 60` (`test_three_distance_theorem`):

| `α` | distinct gaps | largest gap |
|---|---|---|
| golden 0.6180 | 3 | **0.0344** |
| π mod 1 | 2 | 0.0885 |
| 1/10 | 2 | 0.1000 |
| 1/4 | 2 | 0.2500 |
| 1/2 | 2 | 0.5000 |

When the belief has confidently estimated an emitter period, `_adapt` additionally
nudges the Weyl step away from any ratio within 2 % of a rational with
denominator ≤ 12.

A caution worth recording: scaling the *sweep period* by φ is **not** the same
thing and does not work. In an early run `Tr = 0.048·φ` gave `Tr/Te = 0.019417 ≈
2/103` — highly commensurate, 24 % blind. What must be badly approximable is the
**ratio** `Tr/Te`, and since `Te` is unknown a priori, a sequence with no single
period (9) is the robust answer rather than any single tuned `Tr`.

### 2.5 `PhaseLockedScheduler`: predict and park

Once `T̂e` clears a confidence threshold, the next arrival is at
`t̂ = t_last_hit + k·T̂e`. The receiver parks on that channel from

```
t̂ − guard,     guard = 3σ_est + t_settle                     (10)
```

— early enough to have finished settling, late enough not to waste dwell.
Between predicted arrivals it falls back to the Whittle policy, so coverage
continues while waiting.

### 2.6 Estimating `Te` from sparse, irregular intercepts

**Lomb–Scargle** (Lomb 1976; Scargle 1982) is the correct periodogram for
unevenly sampled data, and our sampling is uneven by construction.

**The correction that matters.** The hit sequence is the product of emitter
activity and **our own visit schedule**. A raw periodogram of hit times
therefore peaks at the *receiver's* sweep period — the estimator confidently
reports its own tail. We also compute the spectral window of the visit times,

```
W(f) = | Σ_j exp(−2πi f t_j) |² / N²   ∈ [0, 1]              (11)
```

and score candidates by signal power *not explained by* the window:
`score(f) = P_LS(f) · (1 − W(f))`. `test_lomb_scargle_deconvolves_the_sampling_window`
constructs hits driven **entirely** by the visit schedule and asserts the
deconvolved estimator is less confident than the naive one.

Note that `W(f)` for a periodic sampler is a **comb**: it peaks at the sampling
period *and all its sub-multiples*. Both are genuine and both must be suppressed.

**CDIF/SDIF** (Mardia 1989; Milojević & Popović, *IEE Proc. F* 1992) is the
classical PRI-deinterleaving alternative, applied here at scan-period timescale:
histogram `t[i+k] − t[i]` for difference orders `k = 1..6` against the decreasing
threshold `E(τ) = c·(N − k)·exp(−τ/(k·span))`, with a subharmonic check to catch
the missed-arrival failure mode. It is robust to dropouts and weak under jitter —
the complementary trade-off to Lomb–Scargle, which is why both are reported.

### 2.7 Episode length and the 2 % target

Acceptance test 4 asks for a 4.0 s scan period recovered to within 2 %. A 10 s
episode contains **2.5 revolutions** — two or three beam arrivals. No estimator
recovers 2 % from that under jitter and missed looks; the information is simply
not present. Estimator validation therefore runs on `configs/scan_on_scan.yaml`
at `episode_s = 120` (30 revolutions), at a cost of ~2 s of CPU. The 10 s tiers
are untouched for scheduler benchmarking. This is the one place the
implementation departs from the brief, and it is flagged in
`docs/architecture.md` §17-A, in the config header, and in a test that records
the arithmetic rather than merely asserting the conclusion.

---

## 3. Detection statistics

Square-law envelope detector, `N` non-coherently integrated samples. Let
`W = Z/(2σ²)` be the normalised detector output. Under `H₀`, `W ~ Gamma(N, 1)`,
so the threshold for a required false-alarm probability is

```
T = gammaincinv(N, 1 − Pfa)                                  (12)
```

**Swerling 0** (non-fluctuating): `2W ~ ncx2(2N, 2Nχ)` exactly, hence
`Pd = ncx2.sf(2T, 2N, 2Nχ)` — exact, not an approximation.

**Swerling I** (Rayleigh amplitude constant over the dwell — the right default
for a scanning radar seen through one beam pass): for `N = 1` this collapses to
the familiar

```
Pd = Pfa^(1/(1+χ))                                           (13)
```

verified to machine precision in `test_swerling1_single_pulse_closed_form`. For
`N > 1` the standard closed form is evaluated **in log space**: the naive product
is an `∞ · 0` indeterminate form as `χ → 0`, where `(1 + 1/(Nχ))^(N−1)` overflows
float64 while the incomplete gamma underflows to exactly zero. That produced NaN
precisely at the SNR floor. Both branches now tend correctly to `Pfa`
(`test_pd_at_snr_floor_tends_to_pfa`).

Albersheim's equation appears only as an **independent cross-check** in the test
suite (`test_swerling0_agrees_with_albersheim`, agreement within 0.3 dB over
`N = 1..64`); it never runs in the simulation.

Reference: M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
ch. 6.

### 3.1 Two detection regimes

Which regime applies is a property of the **emitter**, not the receiver:

* **pulse** — an ES receiver has no matched filter, so it detects on a video
  bandwidth of roughly `1/PW`. Each pulse is an independent single-sample
  opportunity and the dwell succeeds if any pulse is detected:
  `Pd_dwell = 1 − (1 − Pd_pulse)^n`.
* **energy** — a narrowband signal lands in one FFT bin, giving
  `10·log₁₀(fft_size)` dB of processing gain against a bin-width noise floor,
  with `N` periodograms averaged over the dwell.

Integrating a 1 µs pulse over a 1 ms dwell would bury it by 30 dB, which is
precisely why real ES receivers use fast log-video detection. Modelling this
correctly is what makes "dwell longer" a genuine trade-off against "hop more
often" rather than a free win.

---

## 4. Regime analysis: when does scheduling matter at all?

A result worth stating plainly, because it determines whether the benchmark
measures anything.

The detectable illumination window is wider than the 3 dB beamwidth. With a
parabolic main lobe `G(θ) = −12(θ/θ_bw)²` and an SNR margin `M` dB above the
detection threshold, the beam is detectable while `12(θ/θ_bw)² < M`, i.e. over

```
θ_detectable ≈ 2·θ_bw·√(M/12)                                (14)
```

At `M ≈ 18 dB` that is ≈ 2.4 · θ_bw, so the effective window is
`we_eff ≈ 2.4·(θ_bw/360)·Te`.

Scheduling matters only when the sweep period is **comparable to or longer than**
`we_eff`. Otherwise the sweep cannot miss and every policy scores the same.

| `B` | sweep period `Tr` | `we_eff` (2°, 3 s) | verdict |
|---|---|---|---|
| 64 | 48 ms | 40 ms | `Tr > we_eff` only marginally — sweep near-optimal, TTFI saturated at 43 ms |
| **128** | **96 ms** | 40 ms | `Tr > we_eff` — sweep misses passes, scheduling discriminates |

The prototype therefore uses `B = 128` (`K/B = 1/32`, still comfortably beyond
the PS's "at least one order of magnitude"). This was **measured, not assumed**:
at `B = 64` every scheduler scored an identical 43 ms TTFI, which is the
signature of a benchmark that cannot tell policies apart. The reasoning is
recorded in the `configs/base.yaml` header beside the parameter it justifies.

---

## References

Full annotated bibliography, and the "gap we fill" table:
[`docs/related_work.md`](related_work.md).

1. P. Whittle, "Restless bandits: activity allocation in a changing world," *J. Applied Probability* 25A, 287-298, 1988.
2. C. H. Papadimitriou, J. N. Tsitsiklis, "The complexity of optimal queuing network control," *Math. of OR* 24(2), 293-305, 1999. -- restless bandits are PSPACE-hard, which is why an index policy plus learning is the practical route rather than exact DP.
3. K. Liu, Q. Zhao, "Indexability of restless bandit problems and optimality of Whittle index for dynamic multichannel access," *IEEE Trans. Information Theory* 56(11), 5547-5567, 2010.
4. Q. Zhao, B. Krishnamachari, K. Liu, "On myopic sensing for multi-channel opportunistic access," *IEEE Trans. Wireless Comm.* 7(12), 5431-5440, 2008. -- the myopic-optimality result used as our test oracle.
5. A. G. Self, B. G. Smith, "Intercept time and its prediction," *IEE Proc. F* 132(4), 215-222, 1985. doi:10.1049/ip-f-1.1985.0052
6. I. V. L. Clarkson, A. D. Pollington, "Performance limits of sensor-scheduling strategies in electronic support," *IEEE Trans. Aerospace and Electronic Systems* 43(2), 645-650, 2007.
7. R. Winsor, E. Hughes, "Optimisation and evaluation of receiver search strategies for electronic support," *IET Radar, Sonar & Navigation*, 2011. doi:10.1049/iet-rsn.2010.0377 -- the closest prior work; optimised scan pattern **from a known threat list**.
8. S. Stein, D. Johansen, "A statistical description of coincidence among random pulse trains," *Proc. IRE* 46, 827-830, 1958. -- origin of the exponential POI model we implement and show to be wrong for periodic scanners.
9. US Patent 6,020,842, "ESM duty dithering scheme for improved probability of intercept at low ESM utilization." -- independent industrial confirmation of the blind-zone pathology in section 2.3.
10. R. G. Wiley, *ELINT: The Interception and Analysis of Radar Signals*, Artech House, 2006.
11. K. Z. Haigh, J. Andrusenko, *Cognitive Electronic Warfare: An Artificial Intelligence Approach*, Artech House, 2021.
12. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, ch. 6.
13. N. R. Lomb, *Astrophysics and Space Science* 39, 447-462, 1976; J. D. Scargle, *Astrophysical Journal* 263, 835-853, 1982.
14. D. Milojevic, B. Popovic, "Improved algorithm for the deinterleaving of radar pulses," *IEE Proc. F* 139(1), 1992.
15. M. Xie et al., "First-order difference curve based on sorted TOA difference sequence," *IET Signal Processing* 17(1), e12162, 2023.
16. A. Garivier, E. Moulines, "On upper-confidence bound policies for switching bandit problems," ALT 2011.
17. L. P. Kaelbling, M. L. Littman, A. R. Cassandra, "Planning and acting in partially observable stochastic domains," *Artificial Intelligence* 101(1-2), 99-134, 1998.
18. V. Vapnik, R. Izmailov, "Learning using privileged information," *JMLR* 16, 2015.
19. T.-Y. Lin et al., "Focal loss for dense object detection," ICCV 2017.
20. J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
21. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 529-533, 2015.
22. S. Wang, H. Liu, P. H. Gomes, B. Krishnamachari, "Deep reinforcement learning for dynamic multichannel access in wireless networks," *IEEE Trans. Cognitive Comm. and Networking* 4(2), 257-265, 2018. -- same channel abstraction, **throughput** reward rather than threat value.
23. E. Gunn et al., "The Turing Synthetic Radar Dataset: a dataset for pulse deinterleaving," arXiv:2602.03856, 2026.
