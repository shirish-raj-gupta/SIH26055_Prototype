# SmartScan — Smart Scan Strategy for Electronic Warfare

**SIH 26055 · DRDO / iDEX**

A closed-loop **Electronic Support (ES) receiver scheduler**. At every 1 ms dwell
it decides which slice of a 0.5–18 GHz band to tune to, in order to intercept
unknown emitters fast and often — with **no prior intelligence** about their
frequencies, scan periods or activity patterns.

The receiver sees **1/32 of the band at a time**. The other 31/32 is not empty;
it is *unknown*. Everything here follows from that.

```bash
pip install -e ".[ml,viz,demo]"
make demo         # live dashboard in a browser — offline, one command
make benchmark    # results.parquet, leaderboard.md/.tex and figures F1–F7
pytest -q         # 175 tests
```

---

## The problem, and why the obvious answer fails

A sweeping receiver and a rotating radar are both **periodic**. Two periodic
processes do not intercept "with probability *p* per look" — they either drift
into coincidence or they **lock out permanently**.

Measured here, for a 4.0 s / 2° scanner:

| Receiver sweep `Tr` | `Tr/Te` | Initial phases that **never** intercept | Classical `E[TTI]` formula says |
|---|---|---|---|
| 0.096 s | 1/32 | 27 % | 16.5 s |
| 1.000 s | **1/4** | **97.7 %** | 172 s |
| 2.000 s | **1/2** | **98.8 %** | 344 s |

The textbook closed form reports a finite mean time to intercept for cases where
**98.8 % of encounters never happen at all**. This is not a corner case: a
uniform sweep has exactly one revisit period, so it is the policy *most* exposed
to it. US Patent 6,020,842 exists because fielded ESM sets hit this, and fixes it
with random duty dithering.

SmartScan's answer is a **golden-ratio Weyl sequence** — the worst-approximable
irrational, which by the three-distance theorem minimises the largest phase gap
for every prefix length. A deterministic guarantee where dithering gives an
expectation. Largest gap at N = 60: **0.034** (golden) vs **0.500** (1/2-periodic).

Derivations: [`docs/theory.md`](docs/theory.md).

---

## What is here

```
smartscan/
  env/         emitters (8 classes) · propagation · calibration · receiver · gym_env
  agents/      belief · baselines · bandits · whittle · predictors · rl_agents · hybrid
  analysis/    scan_on_scan · estimators · metrics
  eval/        benchmark · ablation · scan_validation · plots
  data/        schema · dataset_builder · kaggle_io · tsrd_bridge
  hal/         backend (ABC) · simulated · soapy_stub
  cli.py       run | benchmark | grid | train | estimate | ablate | external
               data | credentials | demo | reproduce | info
dashboard/     app.py — live Streamlit demo
configs/       base · easy · medium · hard · scan_on_scan
docs/          architecture · theory · related_work · config_schema · hardware_roadmap
notebooks/     4 local + 2 Kaggle training notebooks
scripts/       publish_kaggle.py
tests/         env · analysis · reproducibility · acceptance · agents · data
```

### Nine schedulers, one interface

All implement `act(belief, t) -> action` and see the **same** belief — so a
comparison between them is a comparison of policies, not of information.

| Family | Schedulers |
|---|---|
| Open-loop baselines | `sequential` (tuned saw-tooth), `random`, `priority_rr` (briefing wrong 40 % of the time) |
| Statistical bandits | `epsilon_greedy`, `ucb1` (discounted), `thompson` |
| **Restless bandit** | `whittle` — numerical index with **verified** indexability |
| **Scan-on-scan** | `coprime_sweep` (golden-ratio Weyl), `phase_locked` (predict-and-park) |
| Supervised | `predictor` — GRU / dilated TCN / Transformer, privileged distillation |
| RL | `dqn`, `ppo` — from scratch, action-masked |
| Hybrid | `hybrid` — predictor output as an extra RL observation plane |

---

## Headline results — MEDIUM tier, 30 paired seeds

Against the **tuned** sweep (`dwell_slots` swept over {1,2,3,5,8,12,20} and set to
its best value). Beating a deliberately weak incumbent would prove nothing.
Paired bootstrap CIs, Wilcoxon signed-rank, Holm-Bonferroni corrected.

### Threat-weighted interception ratio — the claim that holds up

| Scheduler | vs tuned sweep | 95 % CI | p (Holm) | effect size |
|---|---|---|---|---|
| `epsilon_greedy` | **+305.0 %** | [+209.7 %, +465.6 %] | 1.9e-07 | +1.00 |
| `predictor` | **+159.3 %** | [+75.3 %, +347.4 %] | 2.4e-05 | +0.94 |
| `hybrid` | **+118.6 %** | [+58.9 %, +199.4 %] | 2.9e-04 | +0.87 |
| `thompson` | **+96.6 %** | [+36.5 %, +237.5 %] | 1.6e-05 | +0.94 |
| `phase_locked` | **+76.5 %** | [+38.0 %, +145.2 %] | 1.5e-04 | +0.89 |
| `dqn` | **+73.2 %** | [+35.0 %, +145.3 %] | 2.2e-04 | +0.88 |
| `whittle` | **+67.3 %** | [+37.7 %, +129.1 %] | 7.7e-05 | +0.91 |

Seven schedulers clear the ≥ 15 % target with confidence intervals **entirely
above zero** and near-maximal effect sizes. This is the robust result.

Three of those rows are new, and only because every learned agent is now
genuinely trained. In earlier runs `predictor`, `dqn` and `hybrid` had no
checkpoint and silently substituted an analytic policy — the leaderboard
reported UCB1's numbers under their names. The benchmark now records each
agent's own name and flags any substitution, so a row means what it says.

**And the same table contains three significant regressions**, which belong
here rather than in a footnote:

| Scheduler | vs tuned sweep | 95 % CI | p (Holm) |
|---|---|---|---|
| `coprime_sweep` | **−38.7 %** | [−45.5 %, −33.2 %] | 2.2e-04 |
| `ppo` | **−41.3 %** | [−54.9 %, −29.7 %] | 4.9e-03 |
| `random` | **−43.0 %** | [−49.9 %, −32.7 %] | 1.9e-07 |

### What the interception gains cost — read this beside the table above

Interception ratio is not free, and the agents that win on it lose elsewhere.
Worst-case staleness, MEDIUM, median over 30 seeds:

| Scheduler | max staleness | band coverage |
|---|---|---|
| `ucb1` | **0.246 s** | 0.857 |
| `sequential` (baseline) | 0.623 s | 0.857 |
| `predictor` | 1.170 s | 0.714 |
| `epsilon_greedy` | 2.369 s | 0.724 |
| `dqn` | 2.546 s | 0.800 |
| `hybrid` | 9.861 s | 0.733 |
| `ppo` | **10.000 s** | 0.829 |

`ppo` and `hybrid` leave part of the band unvisited for **the entire 10-second
episode**. They are not scheduling; they have learned to park on the emitters
that pay and abandon the rest. That is a real failure of the reward function as
a proxy for the mission, and it is why `epsilon_greedy`'s +305 % should not be
read as "best scheduler" — it buys interception by dropping coverage from 0.857
to 0.724.

The honest summary: **no single policy dominates.** `ucb1` holds the band best,
`epsilon_greedy` and `predictor` intercept most, and the tuned sweep remains
hard to beat on both at once.

### Time to first intercept, scanning and agile emitters — point estimate only

| Scheduler | vs tuned sweep | 95 % CI | significant? |
|---|---|---|---|
| `ucb1` | +96.4 % | [−86.4 %, +99.1 %] | **no** |
| `coprime_sweep` | +43.7 % | [−75.1 %, +75.6 %] | **no** |
| `phase_locked` | +29.6 % | [−260.8 %, +98.5 %] | **no** |

The point estimates clear the ≥ 25 % target, but **the confidence intervals
straddle zero, so this half of the claim is not statistically supported at 30
seeds.** Reported that way deliberately.

Why: TTFI on hard emitters is a Kaplan-Meier median over roughly five scanning
emitters per seed, a third of which are never intercepted by anyone. That is a
high-variance statistic, and an eight-seed pilot run gave a confident-looking
+26.8 % for `ucb1` that did not survive at thirty. The 30-seed requirement in the
brief exists for exactly this reason, and it earned its place here.

**Known weakness, stated plainly.** The paired test drops non-finite pairs, and
TTFI is `+inf` exactly when an agent never intercepted a hard-class emitter — so
it discards each agent's failures and scores it only on the runs where it
succeeded, flattering the worst performers most. On MEDIUM this once left as few
as **3 of 30 seeds** for some agents while every other metric kept all 30. The
harness now withholds any comparison with fewer than `MIN_PAIRED_SEEDS = 10`
finite pairs and reports what it withheld, so an absent row reads as "not
tested" rather than "tested and did not win".

That stops unsound numbers being published; it does not recover the information
censoring throws away. The correct fix is a **log-rank test on the Kaplan-Meier
curves**, which use the never-intercepted observations rather than discarding
them. The survival curves in F2 already handle censoring properly — the
comparison does not. This is the weakest piece of methodology in the project and
is listed under Known limitations.

### Retracted: the `coverage_weight` tuning result

A five-seed ablation reported that raising `agents.coverage_weight` from 1.0 to
2.0 lifted Whittle's TWIR by 29 %. **It does not.** At twelve *paired* seeds
(`scripts/sweep_coverage_weight.py`, sharing both scenario and detection luck)
the effect is +14.7 % with a 95 % CI of [−26 %, +58 %], and TWIR then falls away
above 2.0 — 0.023 at the default, 0.027 at 2.0, 0.019 at 4.0, 0.014 at 8.0. No
weight beats the default on TWIR at usable confidence for any agent.

What survives is `staleness_max_s`, monotone in the weight and significant from
4.0 up (Whittle 1.82 s → 1.01 s → 0.61 s) at flat coverage — which is close to
tautological, since the term penalises staleness. So the knob is an
interception-versus-worst-case-staleness trade, documented as such, and the
default stays at 1.0.

The tell was on the F7 figure the whole time: weight **0.5 read +28 %** and
weight **2.0 read +29 %** — near-identical gains on *both sides* of the default,
which no monotone effect can produce. F7 now prints its seed count and carries
"point estimates, no confidence intervals — screening only". Treat a tornado bar
as a hypothesis to test, never as a result.

### Other measured results

| Metric | Result |
|---|---|
| Pop-up detection rate (HARD) | `ucb1` 91 % vs sweep 73 % of *reachable* pop-ups |
| 4.0 s scan period recovery | **0.16 %** error (target 2 %), from 75 % before the arrival-clustering fix |
| Estimator comparison | Lomb-Scargle 0.26 % median error, CDIF/SDIF 0.41 %; LS resolves more cases |
| Coverage | `whittle`/`phase_locked` hold the sweep's 0.857, `thompson` drops to 0.733 |

Reproduce: `make reproduce`. Every artefact embeds the resolved config hash.

**PPO is reported as measured, and it loses.** At 1.5 M CPU steps it reaches a
return of 192 against the tuned sweep's 214, and its TWIR is 41 % *worse*. Its
max staleness is 10 s — the entire episode — meaning it learned to park on one
window and abandon the band. Policy entropy did fall from 4.83 (uniform over 125
legal actions) to 2.3, so it is learning something decisive, just not something
good. The design document predicted this ordering (§17-B) before it was measured,
and the headline claim was placed on the analytic policies accordingly.

---

## Design decisions worth a reviewer's time

**Ground truth is precomputed and vectorised.** A per-slot Python loop over 30
emitters is 3×10⁵ calls per episode. Precomputing makes stepping an `O(K)` array
slice: **149 µs/step**, which is what makes CPU RL training possible at all.

**Common random numbers.** The whole detection realisation is drawn *once per
episode, before any scheduler runs*. Two policies on the same seed therefore face
the same world **and the same luck**, so paired comparisons measure policy
quality rather than detection noise.

**Seeding is a `SeedSequence` tree with a fixed name registry**, not
`np.random.seed()`. Adding a 16th emitter does not perturb the first 15; changing
the scheduler does not change the world. Without this, every ablation is silently
invalid.

**`duty[b,t] ∈ [0,1]` alongside binary occupancy.** A 1 µs pulse fills 0.1 % of a
1 ms slot; a 10 kHz frequency-agile emitter hops **ten times inside one slot**. A
binary tensor cannot represent either.

**Detection regime is a property of the emitter, not the receiver.** Pulse-mode
emitters get per-pulse detection at peak SNR (an ES receiver has no matched
filter); continuous emitters get integrated energy. Integrating a 1 µs pulse over
a 1 ms dwell would bury it by 30 dB — which is exactly why real ES sets use fast
log-video detection.

**Censoring is kept.** Emitters never intercepted are right-censored in a
Kaplan-Meier estimate, not dropped. Dropping them is the standard error in this
literature: on a worked example it turns a median of 3.0 into 2.0.

**The Lomb-Scargle periodogram is window-deconvolved.** Hit times are the product
of emitter activity *and our own visit schedule*, so a raw periodogram peaks at
**our** sweep period — the estimator confidently reporting its own tail.

---

## The dataset

A citable, reproducible corpus of 3000 episodes (1000 EASY / 1200 MEDIUM /
800 HARD), published to Kaggle under **CC BY-SA 4.0**.

```bash
make dataset          # build the full corpus  (~45 min, 4 workers)
make dataset-verify   # integrity, splits, byte accounting
make publish-dry      # preflight the upload, send nothing
make publish          # create or version the public Kaggle dataset
```

Per episode: `truth_occupancy.npz` (bit-packed occupancy plus SNR, duty,
emitter id and pulse counts at occupied cells), `emitter_manifest.parquet` (the
ground-truth order of battle) and `observations.parquet` (replayed dwell traces
from seven schedulers). A root `index.parquet` and a `dataset_card.md`
documenting schema, units, provenance and **known limitations**.

**Splits are assigned by scenario seed, never by time slice.** Cutting one
episode into a train half and a test half would let a model memorise the very
emitters it is about to be scored on, and the leakage would be invisible in
every aggregate metric. Assignment is a hash of the seed, so it is stable as the
corpus grows — adding episodes never moves an existing one between splits.

**The loader cannot fail because of wifi.** `load_dataset` tries an explicit
path, then the cache, then Kaggle, and finally **regenerates the identical
episodes from their seeds**. Every episode is a deterministic function of
`(tier, seed, config)`, so a missing download costs ~0.8 s per episode rather
than the demo. A test asserts the regenerated tensors are byte-identical.

**Two Kaggle limits get confused constantly**, and only one binds here:
100 GB is the per-user *dataset storage* quota (the builder's budget, asserted
before upload); 20 GB is a *notebook's writable disk*. Attached datasets mount
read-only at `/kaggle/input` and do **not** count against it, so the training
notebooks stream rather than copy.

The corpus uses **under 1 %** of that 100 GB. Size is no longer the binding
constraint — build time and scientific value are — so the headroom is spent on
*depth* rather than padding:

| Use of headroom | Cost | Worth it? |
|---|---|---|
| `duty` tensor now serialised | +12 % | **Yes.** Sub-slot hopping and 1 µs pulses are representable *only* here |
| 7 replayed schedulers, not 3 | +2.3× | **Yes.** Offline policy evaluation needs data from the policies being evaluated |
| A 120 s `scan_on_scan` tier | +12× per episode | **Available, not built by default.** The only configuration that supports period-estimation research; run `smartscan data build --root build/dataset` with `counts={'scan_on_scan': 200}` |
| More 10 s episodes | linear | **Marginal.** 3000 already saturates the variance of every headline metric |

### External validation — against real third-party data

The problem statement cites the **Turing Synthetic Radar Dataset** (Gunn et al.,
arXiv:2602.03856). Verified against the live repository: **Apache-2.0 but
access-gated**, distributed as HDF5 pulse-descriptor-word arrays, with ToA in
microseconds and RF in MHz.

`smartscan external` fetches it at runtime with **your own token**, bins the PDW
streams onto our `[b, t]` grid, and replays every scheduler over real pulse
trains. Measured on 3 `archive/test` records — 29,748 PDWs spanning 0.36–12 GHz
over 9.5 s, with **73 distinct emitters**, 97.6 % of pulses landing in band:

| scheduler | TWIR | coverage | max staleness (s) |
|---|---|---|---|
| `ucb1` | **0.0085** | 0.324 | **0.180** |
| `sequential` | 0.0059 | 0.333 | 0.623 |
| `whittle` | 0.0023 | 0.333 | 1.853 |
| `phase_locked` | 0.0023 | 0.333 | 1.853 |

UCB1 leads on interception ratio (+44 % over the sweep) and holds the band far
better (0.18 s vs 0.62 s worst-case staleness).

**These numbers are reported separately and tagged `external: true`, and they
are not comparable to the synthetic tiers.** The PDW-to-occupancy binning is an
assumption of the bridge, the emitter mix is whatever TSRD contains rather than
our tiers, and TSRD carries no threat model — so every emitter is scored at a
uniform priority, which is why the absolute TWIR values are an order of
magnitude below the synthetic ones.

The SNR mapping is physical rather than fitted: TSRD's `PA` is read as received
power in dBm and compared against this receiver's own thermal noise in the
pulse-detection bandwidth, exactly as the simulator does. That puts the median
pulse at ~13.7 dB SNR — a plausible intercept, arrived at without a tuned
constant.

**We do not mirror it.** Without a grant, `smartscan external` degrades to
actionable guidance — never a crash, and never synthetic data dressed up as real.

### Credentials

```bash
cp .env.example .env       # then edit; .env is gitignored
smartscan credentials      # reports what is configured, never a value
```

Nothing here needs credentials. The simulator, all nine schedulers, the
benchmark and the tests run without them. Secrets are read from the environment,
a gitignored `.env`, or the provider's own config file — never from the tree,
and only ever reported as an 8-character fingerprint so a rotation is verifiable
without the value appearing in a log.

---

## Live demo

`make demo` opens a browser. No network call, ever — episodes are generated from
seeds in ~50 ms, so a captive-portal venue cannot break it.

The waterfall is the whole argument in one picture: **grey** is what is on the
air, the **blue band** is where the receiver is looking right now, **red** marks
are confirmed intercepts, and **dark red** is signal that transmitted while the
receiver was looking elsewhere.

* **A/B mode** runs two schedulers on the *same seed*, side by side, with a
  running delta. Because the detection realisation is drawn from the scenario
  seed, both face identical luck — so every difference is the policy.
* **Scheduler reasoning** names the top five channels by belief and explains the
  current choice in one line (`ch 41–44: P(active)=0.72 + stale 820 ms`).
  Explainability is what a defence evaluator asks for second, right after
  "does it work".
* **Pop-up injection** spawns a threat mid-episode so you can watch which policy
  notices.
* **60-second auto-demo** plays unattended.

A test asserts the dashboard's live run is *identical* to `run_episode` on the
same seed — a demo that quietly disagrees with the reported numbers is worse
than no demo.

---

## Hardware readiness

Everything above `smartscan/hal/backend.py` consumes `Observation`; everything
below deals in Hz and seconds. Switching to a real SDR is **one config line** —
`receiver.backend: soapy` — with no change to any scheduler, belief, metric or
benchmark.

`SoapySDRBackend` is a **non-functional stub that raises**, carrying the exact
SoapySDR calls it would make. A stub returning plausible fake data would turn a
missing driver into a silently wrong result; `test_soapy_backend_refuses_to_pretend`
asserts the refusal.

Bring-up plan, device comparison, the two measurements that must come first
(`t_settle` by CW step response, NF by Y-factor), and an honest list of what gets
*worse* on real hardware: [`docs/hardware_roadmap.md`](docs/hardware_roadmap.md).

The strongest analytic policies run in microseconds with no learned weights, so
**the headline capability does not depend on shipping a neural network**.

---

## Where this sits in the literature

Every optimised-scan result in the accessible literature needs a prior emitter
list. The problem statement removes it.

| Prior work | Assumes |
|---|---|
| Self & Smith (1985) | emitter periodicities **known** |
| Clarkson & Pollington (2007) | schedule **periodic and fixed** |
| Winsor & Hughes (2011) | threats from a **known list** |
| US 6,020,842 | dither is **random**, not learned |
| US 11,747,438 | **multiple** receivers plus a wideband cue |
| Wang et al. (2018) | **throughput** reward, not threat value |

Online learned scheduling, on a **single** receiver, with **no prior emitter
list**, under a **threat-weighted** reward, is the intersection nobody occupies.
Full annotated bibliography: [`docs/related_work.md`](docs/related_work.md).

---

## Commands

| Command | Purpose |
|---|---|
| `smartscan demo` | 1-minute end-to-end smoke run |
| `smartscan info -c configs/medium.yaml` | resolved config summary and hash |
| `smartscan run -c … -a whittle --n-seeds 5` | one scheduler → metrics JSON |
| `smartscan benchmark -c … --n-seeds 30` | paired leaderboard + Wilcoxon/Holm |
| `smartscan train -c … --what ppo` | train a learned scheduler |
| `smartscan estimate -c configs/scan_on_scan.yaml` | scan-period validation |
| `smartscan ablate -c …` | reward / IBW / retune / density / belief sweeps |
| `smartscan reproduce` | regenerate every headline number |

`make help` lists the equivalent targets.

---

## Reproducibility

* One root seed → a tree of named RNG substreams.
* `tests/test_reproducibility.py` hashes the ground-truth tensors against
  checked-in golden digests, per tier.
* `torch_threads` is **pinned** (not core-count-derived), so CPU reduction order
  is identical across machines.
* Every `metrics.json`, checkpoint and figure embeds the resolved config hash.

---

## Known limitations

Stated because they are the first things a reviewer should ask about.

1. **Acceptance test 4 needs a 120 s episode.** A 4.0 s scanner in a 10 s episode
   is 2.5 revolutions — two or three arrivals. No estimator recovers 2 % from
   that. `configs/scan_on_scan.yaml` extends the horizon for *estimator
   validation only*; the 10 s tiers are untouched for scheduler benchmarking.
2. **The TTFI comparison discards censored observations.** It drops `+inf`
   pairs, which occur exactly when an agent never intercepted a hard-class
   emitter, so it scores agents only on runs where they succeeded. Comparisons
   below 10 finite pairs are now withheld and reported as withheld, but the
   correct fix is a log-rank test on the Kaplan-Meier curves. F2's survival
   curves already censor correctly; the paired test does not.
3. **Two learned policies fail by parking.** `ppo` and `hybrid` reach a median
   worst-case staleness of 10.0 s and 9.86 s on MEDIUM — the whole episode —
   leaving part of the band unvisited throughout. Their interception gains are
   bought by abandoning coverage, which is a failure of the reward function as
   a proxy for the mission rather than a training bug.
4. **The predictors saw ~5 % of the available data.** Each was trained on ~40
   episodes regenerated from seeds; the published corpus holds 854 train
   episodes for MEDIUM alone. `--dataset` now streams the full corpus, but the
   shipped checkpoints predate it. Note that `predictor_easy` reached AUC 0.911
   from the *smallest* corpus of the three, so more data is not obviously the
   limitation.
5. **RL has not been trained to convergence** (§17-B, §21-G).
6. **Sector-scan period estimates are ambiguous by a factor of 2.** A
   bidirectional sweep genuinely illuminates twice per frame.
7. **The TTFI half of acceptance test 3 is a point estimate, not a significant
   result.** See the results section: the CI straddles zero at 30 seeds.
8. **Sim-to-real gap.** The claim is scheduling-policy quality, not RF realism.

---

## Licence

MIT. See [`docs/architecture.md`](docs/architecture.md) for the full design.
