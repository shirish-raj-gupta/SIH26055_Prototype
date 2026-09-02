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
| **Winsor & Hughes (2011)** | Optimised scan pattern via evolutionary search, high POI on one narrowband receiver | Threat emitters come **from a known list** |
| **US 6,020,842** | Duty *dithering* to escape blind zones | Dither is **random**, not learned |
| **US 11,747,438** | Priority-based band allocation | Needs a **wideband cueing receiver plus multiple** narrowband receivers |
| **Turing Synthetic Radar Dataset (Gunn et al., 2026)** | Realistic PDW data with ground truth | Built for **deinterleaving**, not scheduling |
| **DSA-RL literature (Wang et al., 2018)** | Learned channel selection under partial observability | **Throughput** reward, not threat value |

Every optimised-scan result in this list needs a prior emitter list. The problem
statement explicitly removes it.

> **Nobody in the accessible literature does online *learned* scheduling, on a
> *single* receiver, with *no prior emitter list*, under a *threat-weighted*
> reward.** That intersection is the contribution.

### 1.1 Where SmartScan sits on each axis

| Axis | Prior work | SmartScan |
|---|---|---|
| Emitter periods | assumed known (Self & Smith) | **estimated online** from sparse intercepts — `analysis/estimators.py` |
| Schedule | fixed periodic (Clarkson & Pollington) | **belief-driven, aperiodic** — `agents/whittle.py`, `analysis/scan_on_scan.py` |
| Emitter list | known a priori (Winsor & Hughes) | **none**; `priority_rr` models a briefing that is wrong 40 % of the time and shows graceful degradation |
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

3. **Winsor, R. & Hughes, E. (2011).** "Optimisation and evaluation of receiver
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
