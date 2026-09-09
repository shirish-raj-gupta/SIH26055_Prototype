# SmartScan — Related Work and the Gap We Fill

**SIH 26055 · "Smart Scan Strategy for Electronic Warfare"**

Citations verified against source rather than recalled. Where a reference
justifies a specific implementation choice, the file and function that rests on
it are named — a bibliography that cannot be traced to code is decoration.

---

## 1. The gap, stated plainly

| Prior work | What it gives | What it assumes |
|---|---|---|
| **Self & Smith (1985)** | Closed-form intercept-time statistics for coinciding parametric windows | Emitter periodicities are **known** |
| **Clarkson & Pollington (2007)** | Performance *limits* for periodic sensor schedules in ES | The schedule is periodic and **fixed** |
| **Clarkson (2005)** | Min-max intercept time, jointly over sweep period *and* per-band dwells, via a Farey/(α,ε) triangulation | The schedule is **periodic and fixed**, and emitter scan periods are **known** |
| **Balaban (2012)** | Per-band *revisit periods* chosen from POI; short frequent dwells beat one long dwell | Emitter scan periods and PRIs **known**; no threat weighting |
| **Teissier et al. (2024)** | Multi-armed-bandit multichannel scanning; **no prior emitter list**; ESPE learns scan periods online | Unweighted binary intercept; multichannel (n = 1…20) |
| **Winsor & Hughes (2012)** | Optimised scan pattern via evolutionary search, high POI on one narrowband receiver | Threat emitters come **from a known list** |
| **US 6,020,842** | Duty *dithering* to escape blind zones | Dither is **random**, not learned |
| **US 11,747,438** | Priority-based band allocation | Needs a **wideband cueing receiver plus multiple** narrowband receivers |
| **Turing Synthetic Radar Dataset (Gunn et al., 2026)** | Realistic PDW data with ground truth | Built for **deinterleaving**, not scheduling |
| **DSA-RL literature (Wang et al., 2018)** | Learned channel selection under partial observability | **Throughput** reward, not threat value |

An earlier version of this section claimed the contribution was the intersection
of *learned*, *single-receiver*, *no prior emitter list* and *threat-weighted*.
Three of those four are occupied, and saying so is worth more than the claim was.
**Teissier et al. (2024)** do learned scheduling with no prior emitter list, and
sweep channel counts from n = 1, so the single-receiver case is theirs too. The
honest statement is narrower:

> **What is ours is the reward and the adaptation, not the absence of a list.**
> Prior optimised-scan work either fixes the schedule in advance from a known
> emitter list (Clarkson 2005, Balaban 2012, Winsor & Hughes 2012) or learns
> online against an *unweighted* intercept count (Teissier et al. 2024). We
> schedule from a belief updated *within the episode* under a **threat-weighted**
> reward, and report the mission metric (hard-target hazard) alongside the
> flattering one.

### 1.0 The impossibility result we have to answer

Clarkson & Pollington (2007) — reference 2 below, and already load-bearing here —
is stronger than the table admits. As Winsor & Hughes summarise it: *"it is not
possible to design a deterministic receiver search strategy with superior
performance to a search strategy using a random pattern of frequency selection
against emitters with unknown parameters."*

Taken at face value that guts the "no prior emitter list" framing: with unknown
parameters, no fixed schedule beats random. Our answer is that the bound is about
**fixed** schedules chosen in advance. `whittle`, `predictor` and `thompson` are
not fixed; they re-plan from intercepts accumulated *during* the episode, so the
parameters stop being unknown as the episode runs. That is why
`analysis/estimators.py` exists at all. The bound is the reason our two analytic
winners must beat `random` — and in the 30-seed grid they do, `whittle` at
+67 % TWIR and `phase_locked` at +77 %, both with CIs above zero, against
`random` at −43 %.

The corollary is a constraint we accept: any of our schedulers that stops
adapting is, by this theorem, no better than random. `coprime_sweep` is exactly
that case, and it measures −38.7 %.

### 1.1 Where SmartScan sits on each axis

| Axis | Prior work | SmartScan |
|---|---|---|
| Emitter periods | assumed known (Self & Smith) | **estimated online** from sparse intercepts — `analysis/estimators.py` |
| Schedule | fixed periodic (Clarkson & Pollington) | **belief-driven, aperiodic** — `agents/whittle.py`, `analysis/scan_on_scan.py` |
| Emitter list | known a priori (Clarkson, Balaban, Winsor & Hughes) | **none** — as in Teissier et al.; `priority_rr` additionally models a briefing that is wrong 40 % of the time and shows graceful degradation |
| Dwell allocation | one long dwell per band per sweep (Clarkson 2005) | short, **frequent** revisits — the effect Balaban (2012) measures and `phase_locked` exploits |
| Blind-zone escape | random dither (US 6,020,842) | **golden-ratio Weyl sequence** — provably minimises the largest phase gap (three-distance theorem) |
| Receivers | wideband cue + N narrowband (US 11,747,438) | **one** receiver, `K/B = 1/32` |
| Reward | throughput (Wang et al.) | **threat-weighted** intercept, novelty, coverage staleness — `runner.RewardAccountant` |

### 1.2 Independent corroboration of our central finding

The most useful thing in this list, from a defensibility standpoint, is
**US Patent 6,020,842**. It exists because ESM receivers with regular duty
cycles suffer *blind zones* against periodic emitters, and its remedy is duty
dithering. That is independent, industrial confirmation that the synchronism
pathology in `docs/theory.md` §2.3 is a real operational problem and not a
simulation artefact.

Our measurement of it: at `Tr/Te = 1/2`, **98.8 % of initial phases never
intercept**, while the classical `E[TTI] = Tr·Te/(wr+we)` formula reports a
finite 344 s. Reproduced in
`tests/test_analysis.py::test_commensurate_sweep_can_be_permanently_blind`.

Where we go beyond the patent: dithering is random, so it escapes lockout only
in expectation. A golden-ratio Weyl sequence is the **worst-approximable**
choice, which by the three-distance theorem minimises the largest gap for every
prefix length — a deterministic guarantee rather than an expectation.
Measured largest gap at `N = 60`: 0.034 (golden) versus 0.500 (a 1/2-periodic
schedule).

### 1.3 Dwell-efficiency: the literature names our worst result

The most uncomfortable number this project produces is that `predictor`, our
strongest agent on threat-weighted interception (+195 %), is simultaneously the
**worst** policy in the log-rank table: it never intercepts 126 of 146 scanning
and agile emitters, against the tuned sweep's 68. Two independent sources name
the mechanism, and neither is ours.

**Teissier et al. (2024)** call it *dwell-efficiency* — "the capacity of a
scanning method to monitor intercepted emitters with as few resources as possible
to maximise the probability of intercept for other emitters." Their isolated
Emitter 3 is the same story in miniature: *"no method is better than random…
Exploitation of other emitters can only increase the intercept time for this
emitter."* A policy that dwells to confirm what it already believes is spending
the budget that finding the next emitter needs.

**Balaban (2012)** measures the same trade from the opposite side. Replacing one
long dwell per sweep with short, frequent revisits shortened intercept time
against every emitter in his list *and* left the receiver idle 30 % of the time;
against a single emitter he matched the intercept time of a 100 %-duty schedule
using 0.077 % of receiver time. Long dwells buy confirmation, not coverage.

This is why the improvement is legible as a regression. Retraining lifted the
predictor's ranking quality (AUC 0.683 → 0.763) and its TWIR (+159 % → +195 %),
and pushed its hard-target hazard the wrong way (0.536 → 0.362, never-intercepted
112 → 126) over the same 30 seeds, with every policy that does not read predictor
weights unchanged to the digit. A better occupancy model made the argmax more
confident, and a more confident argmax parks harder. `phase_locked` — which
estimates the scan period and dwells only when the emitter is *predicted active*,
the same idea as Teissier's ESPE — is the dwell-efficient counterpart, and it is
one of the two policies that gains TWIR without losing hard-target coverage.

Where we go beyond ESPE: their period estimate is a Gaussian-smoothed histogram
of intercept-time differences. `analysis/estimators.py` opens by naming the
failure that construction has — "a raw periodogram of hits peaks at the
**receiver's sweep period**; the estimator confidently reports its own tail" —
and corrects for the observation window with Lomb-Scargle before falling back to
CDIF/SDIF. The bias is real and unaddressed in the 2024 paper.

### 1.4 What is *not* in scope, and why

Three of the surveys consulted for this section — Lesieur et al. (2025), Qu et
al. (2026), Mottier et al. (2023) — are about **deinterleaving**: sorting an
interleaved PDW stream by emitter. That is the problem *after* ours. Lesieur et
al. make the boundary explicit, listing "cognitive radar: optimise frequency band
switching strategy for surveillance" among related topics that "cannot be
considered as being within the scope." It is the same boundary that makes the
Turing Synthetic Radar Dataset a corpus we can borrow for realism but not a
benchmark we can be scored against: it is built for deinterleaving, and carries
no scheduling ground truth.

---

## 2. Annotated bibliography

### 2.1 Electronic warfare — probability of intercept and scan strategy

1. **Self, A. G. & Smith, B. G. (1985).** "Intercept time and its prediction."
   *IEE Proceedings F* 132(4), 215–222. doi:10.1049/ip-f-1.1985.0052
   *The* foundational analytic treatment of interception as time coincidence
   between parametric windows. → `analysis/scan_on_scan.py`: the coincidence
   condition (6) and `expected_time_to_intercept`.

2. **Clarkson, I. V. L. & Pollington, A. D. (2007).** "Performance limits of
   sensor-scheduling strategies in electronic support." *IEEE Trans. Aerospace
   and Electronic Systems* 43(2), 645–650.
   Theoretical bounds on what *any* periodic scheduler can achieve — the number-
   theoretic argument that motivates a badly-approximable sweep ratio.
   → `CoprimeSweepScheduler`, `three_distance_gaps`.

3. **Winsor, C. & Hughes, E. J. (2012).** "Optimisation and evaluation of receiver
   search strategies for electronic support." *IET Radar, Sonar & Navigation.*
   doi:10.1049/iet-rsn.2010.0377
   The closest prior work: a single narrowband receiver achieving high POI via
   an evolutionary-optimised scan pattern — **from a known threat list**. Our
   contrast case: we hold no list and learn online.

4. **Stein, S. & Johansen, D. (1958).** "A statistical description of
   coincidence among random pulse trains." *Proc. IRE* 46, 827–830.
   The *random* counterpart to Self & Smith's deterministic analysis; the origin
   of the exponential POI model we implement as `poi_exponential` **and
   demonstrate to be wrong for periodic scanners**.

5. **Reddy, R. & Sinha, S. (2025).** "State-of-the-art review: electronic
   warfare against radar systems." *IEEE Access.*

6. **Clarkson, I. V. L. (2005).** "Optimal periodic sensor scheduling in
   Electronic Support." *Proc. Defence Applications of Signal Processing*.
   Min-max intercept time optimised jointly over sweep period **and** per-band
   dwell times, using a Farey-series triangulation of the (α, ε) plane into
   regions of constant intercept time. The nearest prior art to our scheduling
   problem, and the strongest fixed-schedule baseline in the literature: on his
   own three-emitter example, joint optimisation reaches 25.16 s max intercept
   against 65 s for dwell-only. Assumes scan periods known and the schedule
   fixed in advance. → the baseline `sequential` is tuned in this spirit
   (`dwell_slots` swept over {1,2,3,5,8,12,20}); beating an untuned sweep would
   prove nothing.

7. **Balaban, H. S. (2012).** "Optimum search strategies for Electronic Support
   Measures receivers." M.Sc. thesis, Middle East Technical University.
   Implements Simple Search and Clarkson's algorithm, then proposes giving each
   band its **own revisit period** derived from a target probability of
   intercept. Finding we lean on in §1.3: short frequent revisits beat one long
   dwell on every emitter tested, and against a single emitter match a
   100 %-duty schedule's intercept time using 0.077 % of receiver time.
   → corroborates `phase_locked` and the dwell-efficiency reading of the
   `predictor` log-rank result.

8. **Teissier, G., Toumi, A., Comblet, F. & Khenchaf, A. (2024).** "Adaptive
   multichannel scanning strategies for electronic support using multi-armed
   bandits." *IEEE RADAR*.
   The closest work to ours and the reason §1's novelty claim is narrowed.
   Bandit-driven scanning with **no prior emitter list**, channel counts swept
   from n = 1, and ESPE — an online scan-period estimator built from a
   Gaussian-smoothed histogram of intercept-time differences. Source of the
   *dwell-efficiency* framing in §1.3. Differs from us in the reward: an
   unweighted binary intercept, not threat value. Its period estimator does not
   correct for observation-window bias; ours does (`analysis/estimators.py`).

9. **Deinterleaving literature — adjacent, not competing.** Lesieur, L., Le
   Caillec, J.-M., Khenchaf, A., Guardia, V. & Toumi, A. (2025), "An overview
   and classification of machine learning approaches for radar signal
   deinterleaving," *IEEE Access* 13, 28008; Qu, Z., Zhang, J., Zhou, Y. & Ni,
   L. (2026), "The intelligent evolution of radar signal deinterleaving,"
   *Sensors* 26(1), 248; Mottier, M., Chardon, G. & Pascal, F. (2023),
   "Deinterleaving RADAR emitters with optimal transport distances,"
   arXiv:2312.11178.
   Deinterleaving sorts a received PDW stream by emitter — the task *after*
   scheduling. Lesieur et al. put "cognitive radar: optimise frequency band
   switching strategy for surveillance" explicitly outside their scope, which is
   the same boundary that makes the Turing dataset (§1) a realism corpus rather
   than a benchmark we can be scored against. Cited to fix the boundary, not
   because they bear on scheduling.

### 2.2 Patents — prior art and freedom to operate

6. **US 6,020,842** — "ESM duty dithering scheme for improved probability of
   intercept at low ESM utilization." Independent validation of the blind-zone
   problem and of dithering as the remedy (§1.2).

7. **US 11,747,438** — Cognitive electronic warfare scheduler. Priority-based
   band allocation, but requires a wideband cueing receiver plus multiple
   narrowband receivers. SmartScan targets the single-receiver case.

### 2.3 Standard texts

8. **Wiley, R. G. (2006).** *ELINT: The Interception and Analysis of Radar
   Signals.* Artech House. — intercept receiver architectures, scan-on-scan.
9. **Schleher, D. C. (1999).** *Electronic Warfare in the Information Age.* Artech House.
10. **Adamy, D.** *EW 101: A First Course in Electronic Warfare.* Artech House.
11. **Haigh, K. Z. & Andrusenko, J. (2021).** *Cognitive Electronic Warfare: An
    Artificial Intelligence Approach.* Artech House. — the framing this project sits in.

### 2.4 Deinterleaving and emitter identification

12. **Gunn, E. et al. (2026).** "The Turing Synthetic Radar Dataset: A dataset
    for pulse deinterleaving." arXiv:2602.03856. — the dataset the problem
    statement names. **Verified against the live repository**: Apache-2.0 but
    **access-gated**; HDF5 pulse-descriptor-word arrays, not band-occupancy
    matrices; ToA in microseconds and RF in MHz. Subsets are `archive`
    (0.36–12 GHz over ~9.5 s, up to ~88 emitters), `stare` (oracle receiver) and
    `scan` (a sweeping receiver). Implemented in `data/tsrd_bridge.py`, which
    fetches at runtime with the user's own token and bins PDW streams onto our
    `[b, t]` grid. **Not mirrored.** External results are reported separately and
    tagged `external: true` — see `README.md` for the measured table.
13. **Gunn, E. et al. (2025).** "Radar pulse deinterleaving with transformer-based
    deep metric learning." IEEE RADAR 2025, arXiv:2503.13476.
14. **Qu, Z. et al. (2025).** "The intelligent evolution of radar signal
    deinterleaving." *Sensors* 26(1), 248.
15. **Nuhoglu, M. A. & Cirpan, H. A. (2023).** *IEEE Access* 11, 142043–142061.
16. **Xie, M. et al. (2023).** "First-order difference curve based on sorted TOA
    difference sequence." *IET Signal Processing* 17(1), e12162. → the
    difference-histogram family our `estimate_period_sdif` belongs to.
17. **Campello, Moulavi & Sander (2013).** HDBSCAN. PAKDD.

### 2.5 Scheduling under partial observability

18. **Whittle, P. (1988).** "Restless bandits: activity allocation in a changing
    world." *J. Applied Probability* 25A, 287–298. → `agents/whittle.py`, eq. (3).
19. **Papadimitriou, C. H. & Tsitsiklis, J. N. (1999).** "The complexity of
    optimal queuing network control." *Math. of OR* 24(2), 293–305.
    Establishes that restless bandits are **PSPACE-hard**, which is precisely
    why an index policy plus learning is the practical route rather than exact
    dynamic programming. This is the citation that justifies the whole approach.
20. **Liu, K. & Zhao, Q. (2010).** "Indexability of restless bandit problems and
    optimality of Whittle index for dynamic multichannel access." *IEEE Trans.
    Information Theory* 56(11), 5547–5567. → our indexability check and the
    closed-form regime used as a **test oracle**.
21. **Zhao, Q., Krishnamachari, B. & Liu, K. (2008).** "On myopic sensing for
    multi-channel opportunistic access." *IEEE Trans. Wireless Comm.* 7(12),
    5431–5440. → the myopic-optimality result asserted in
    `test_whittle_matches_myopic_for_identical_positively_correlated_channels`.
22. **Kaelbling, L. P., Littman, M. L. & Cassandra, A. R. (1998).** "Planning and
    acting in partially observable stochastic domains." *Artificial Intelligence*
    101(1–2), 99–134. → the POMDP formalisation in `docs/architecture.md` §2.

### 2.6 Reinforcement learning

23. **Sutton, R. S. & Barto, A. G. (2018).** *Reinforcement Learning: An
    Introduction*, 2nd ed. MIT Press.
24. **Mnih, V. et al. (2015).** *Nature* 518, 529–533. → `DQNScheduler`.
25. **Schulman, J. et al. (2017).** "Proximal policy optimization algorithms."
    arXiv:1707.06347. → `PPOScheduler`, `train_ppo`.
26. **Wang, S., Liu, H., Gomes, P. H. & Krishnamachari, B. (2018).** "Deep
    reinforcement learning for dynamic multichannel access in wireless networks."
    *IEEE Trans. Cognitive Comm. and Networking* 4(2), 257–265.
    The closest RL prior work — same Gilbert-Elliott channel abstraction, but a
    **throughput** objective. Our reward is threat-weighted, which changes the
    optimal policy: throughput rewards parking on a reliably-busy channel, threat
    coverage does not (`docs/theory.md` §1.6 records the measured consequence).
27. **Raffin, A. et al. (2021).** "Stable-Baselines3." *JMLR* 22(268). — supported
    as an optional path; the bundled PPO/DQN are the default (architecture §11.3).

### 2.7 Detection and estimation

28. **Richards, M. A. (2014).** *Fundamentals of Radar Signal Processing*, 2nd ed.
    McGraw-Hill, ch. 6. → `env/propagation.py`, Swerling 0/I.
29. **Lomb, N. R. (1976).** *Astrophysics and Space Science* 39, 447–462;
    **Scargle, J. D. (1982).** *Astrophysical Journal* 263, 835–853.
    → `estimate_period_ls`.
30. **Milojević, D. & Popović, B. (1992).** "Improved algorithm for the
    deinterleaving of radar pulses." *IEE Proc. F* 139(1). → CDIF/SDIF.
31. **Garivier, A. & Moulines, E. (2011).** "On upper-confidence bound policies
    for switching bandit problems." ALT. → the decayed posterior in `agents/belief.py`.
32. **Vapnik, V. & Izmailov, R. (2015).** "Learning using privileged
    information." *JMLR* 16. → the teacher/student scheme in `agents/predictors.py`.
33. **Lin, T.-Y. et al. (2017).** "Focal loss for dense object detection." ICCV.
    → `masked_focal_loss`.

### 2.8 Data and tooling

- `huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset`
  — **gated**, request access early. Do **not** mirror or re-upload; access at
  runtime with a user-supplied token and report external validation separately,
  stating the licence and access conditions.
- `github.com/alan-turing-institute/turing-deinterleaving-challenge` — loader
  library and benchmark metrics.
- SoapySDR; Analog Devices ADALM-Pluto; Great Scott Gadgets HackRF documentation
  → `docs/hardware_roadmap.md`.

---

## 3. A correction to an earlier draft

An earlier version of `docs/theory.md` cited Clarkson under the title
*"Optimisation of periodic search strategies for electronic support"*. The
verified reference is **Clarkson & Pollington (2007), "Performance limits of
sensor-scheduling strategies in electronic support," IEEE T-AES 43(2),
645–650**. Corrected throughout. Recorded here rather than silently swapped,
because a bibliography whose provenance is untraceable is worth less than one
with a visible erratum.
