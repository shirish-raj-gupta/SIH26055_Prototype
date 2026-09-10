# Best scheduler with the better predictor

_6 paired seeds per tier, full-corpus HARD predictor (AUC 0.7459), identical
inference budget (`predict_every=16`) for every predictor-based arm._

Scored on the two objectives SIH 26055 names, against the tuned sweep, with
never-intercepted alongside as the coverage diagnostic.

## `easy`

| scheduler | intercept time | interception rate | never intercepted |
|---|---|---|---|
| `sequential` (baseline) | — | — | 0 |
| `coprime_sweep` | +12.6% | -45.1% | 0 |
| `predictor (all slots)` | +37.1% | +602.3% | 0 **← both improved** |
| `predictor_sweep r=1` | +16.4% | -24.7% | 0 |
| `predictor_sweep r=2` | +14.5% | -4.4% | 0 |
| `predictor_sweep r=4` | -19.5% | +38.5% | 0 |

## `medium`

| scheduler | intercept time | interception rate | never intercepted |
|---|---|---|---|
| `sequential` (baseline) | — | — | 14 |
| `coprime_sweep` | +12.2% | -37.2% | 7 |
| `predictor (all slots)` | -616.6% | +324.9% | 34 |
| `predictor_sweep r=1` | +19.7% | -27.8% | 9 |
| `predictor_sweep r=2` | +14.4% | -15.1% | 12 |
| `predictor_sweep r=4` | -6.6% | +2.8% | 18 |

## `hard`

| scheduler | intercept time | interception rate | never intercepted |
|---|---|---|---|
| `sequential` (baseline) | — | — | 25 |
| `coprime_sweep` | +26.8% | -43.3% | 26 |
| `predictor (all slots)` | -15.8% | +61.6% | 39 |
| `predictor_sweep r=1` | +37.7% | -38.8% | 25 |
| `predictor_sweep r=2` | +5.3% | -28.9% | 24 |
| `predictor_sweep r=4` | +7.0% | -35.7% | 27 |

## What this settles

**`predictor_sweep` does not dominate.** It was built so the predictor could help
without costing coverage: the golden-ratio sweep keeps its phase and the predictor
may only substitute a window within `refine_radius` of the sweep's own choice.
That part works exactly as designed -- on HARD it holds never-intercepted at 24-27
against the sweep's 26 while `predictor`, given every slot, collapses to 39, and
`refine_radius` moves the trade smoothly rather than falling off a cliff.

But preserving coverage is not the same as winning. Across every tier and radius,
`predictor_sweep` reliably improves intercept time -- it is a sweep -- and
reliably loses interception rate. There is no radius that beats the baseline on
both. The dial interpolates between the sweep and the predictor without finding a
point above either.

**On `easy` the answer is unambiguous and it is the plain predictor:** +37.1%
intercept time and +602.3% interception rate, both improved, with zero emitters
never intercepted. Coverage is free on that tier -- every policy reaches zero --
so constraining the predictor to a sweep spends slots on a guarantee that costs
nothing to keep, and the unconstrained model wins outright. This is the better
predictor converting directly into both named objectives.

**On `medium` and `hard` nothing improves both**, which matches
`ps_objectives.md` on the full 30-seed grid. The frontier there is real: the
sweep family for intercept time and completeness, `predictor` for collection
rate, and the mission chooses.
