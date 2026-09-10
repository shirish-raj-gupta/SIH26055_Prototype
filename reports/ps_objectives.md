# The two objectives SIH 26055 names

> _"Development of a robust scheduler using machine learning to **minimize
> intercept time** and **ensure a high interception rate** is the primary
> objective of the strategy."_

Improvement over the **tuned** sequential sweep, paired seed-for-seed, with
95% bootstrap CIs and Holm correction across each tier's family of tests. A
scheduler **qualifies** only by improving *both* objectives significantly --
winning interception rate by parking on one loud emitter while intercept time
collapses is not a smart scan strategy.

## `easy` — 30 paired seeds vs `sequential`

| scheduler | intercept time | interception rate | verdict |
|---|---|---|---|
| `predictor` | **+36.9%** [+10%, +51%] p=0.0204 | **+87.6%** [+86%, +91%] p=4.47e-08 | **both improved** |
| `priority_rr` | **+53.0%** [+43%, +61%] p=0.0043 | **+29.1%** [+24%, +33%] p=6.33e-08 | **both improved** |
| `ucb1` | **+36.9%** [+19%, +58%] p=0.00461 | **+18.9%** [+13%, +23%] p=8.38e-08 | **both improved** |
| `whittle` | **+36.9%** [+19%, +58%] p=0.00461 | **+11.8%** [+5%, +17%] p=8.36e-05 | **both improved** |
| `phase_locked` | **+36.9%** [+19%, +58%] p=0.00461 | **+11.8%** [+5%, +17%] p=8.36e-05 | **both improved** |
| `ppo` | +36.9% [+7%, +55%] p=0.0524 | **+90.8%** [+88%, +92%] p=4.47e-08 | improved, no harm |
| `epsilon_greedy` | +16.8% [-5%, +34%] p=0.162 | **+73.2%** [+70%, +77%] p=4.47e-08 | improved, no harm |
| `thompson` | +18.8% [-6%, +38%] p=0.167 | **+64.4%** [+60%, +66%] p=4.47e-08 | improved, no harm |
| `dqn` | -11.4% [-57%, +5%] p=0.162 | **-73.9%** [-87%, -65%] p=0.0434 | no |
| `random` | -8.7% [-46%, +25%] p=0.781 | **-74.7%** [-86%, -70%] p=4.47e-08 | no |
| `coprime_sweep` | +22.8% [+4%, +37%] p=0.135 | **-75.9%** [-86%, -73%] p=4.47e-08 | no |
| `hybrid` | **-8172.2%** [-9663%, -6163%] p=6.33e-08 | **-636.6%** [-830%, -472%] p=4.47e-08 | no |

**Recommended for `easy`: `predictor`** — intercept time +36.9%, interception rate +87.6%, both significant after Holm correction.

## `medium` — 30 paired seeds vs `sequential`

| scheduler | intercept time | interception rate | verdict |
|---|---|---|---|
| `epsilon_greedy` | -10.0% [-69%, +4%] p=0.208 | **+75.3%** [+68%, +82%] p=4.47e-08 | improved, no harm |
| `predictor` | **-482.5%** [-754%, +19%] p=0.0284 | **+66.1%** [+59%, +80%] p=4.47e-08 | no |
| `hybrid` | -23.1% [-88%, +17%] p=0.208 | **+55.2%** [+32%, +69%] p=9.56e-05 | improved, no harm |
| `thompson` | +18.3% [-33%, +30%] p=0.894 | **+49.1%** [+27%, +70%] p=3.44e-06 | improved, no harm |
| `phase_locked` | +27.5% [+19%, +35%] p=0.208 | **+43.4%** [+27%, +59%] p=3.2e-05 | improved, no harm |
| `dqn` | **-68.6%** [-112%, -40%] p=0.000355 | **+42.3%** [+26%, +59%] p=4.97e-05 | no |
| `whittle` | +27.5% [+19%, +35%] p=0.208 | **+40.2%** [+27%, +56%] p=1.67e-05 | improved, no harm |
| `priority_rr` | +21.8% [+3%, +48%] p=0.33 | -7.0% [-25%, +5%] p=0.208 | no |
| `ucb1` | +32.8% [+19%, +40%] p=0.209 | -9.4% [-24%, +3%] p=0.209 | no |
| `coprime_sweep` | +24.0% [-3%, +35%] p=0.208 | **-63.2%** [-85%, -50%] p=4.97e-05 | no |
| `ppo` | **-98.3%** [-258%, -23%] p=0.00088 | **-70.2%** [-123%, -43%] p=0.000917 | no |
| `random` | -31.4% [-63%, +6%] p=0.154 | **-75.5%** [-100%, -48%] p=4.47e-08 | no |

**Recommended for `medium`: `epsilon_greedy`** — interception rate +75.3% (significant), intercept time -10.0% (p=0.208, unchanged). No scheduler improves both significantly on this tier, but this one improves the rate at no measured cost in time.

## `hard` — 30 paired seeds vs `sequential`

| scheduler | intercept time | interception rate | verdict |
|---|---|---|---|
| `predictor` | **-1397.9%** [-2905%, -1002%] p=3.58e-06 | **+71.1%** [+64%, +75%] p=3.91e-08 | no |
| `dqn` | **-44.0%** [-93%, -9%] p=0.00932 | **+54.6%** [+50%, +60%] p=3.91e-08 | no |
| `phase_locked` | +3.0% [-53%, +21%] p=0.336 | **+51.5%** [+43%, +60%] p=2.35e-07 | improved, no harm |
| `whittle` | +3.0% [-53%, +21%] p=0.336 | **+48.8%** [+41%, +59%] p=4.43e-07 | improved, no harm |
| `hybrid` | — | +46.8% [-285%, +59%] p=0.219 | no |
| `thompson` | **-273.9%** [-904%, -85%] p=0.000123 | +32.8% [-23%, +52%] p=0.249 | no |
| `ppo` | -4182.0% [-6667%, -67%] p=0.125 | +17.5% [-45%, +27%] p=0.428 | no |
| `ucb1` | +17.5% [+2%, +31%] p=0.336 | **-50.6%** [-62%, -40%] p=4.43e-07 | no |
| `priority_rr` | **-57.3%** [-104%, -17%] p=0.000534 | **-72.0%** [-88%, -52%] p=1.55e-05 | no |
| `coprime_sweep` | +23.5% [+3%, +31%] p=0.161 | **-72.3%** [-86%, -55%] p=3.91e-08 | no |
| `random` | **-21.8%** [-50%, -7%] p=0.00373 | **-75.9%** [-98%, -59%] p=1.4e-05 | no |

**Recommended for `hard`: `phase_locked`** — interception rate +51.5% (significant), intercept time +3.0% (p=0.336, unchanged). No scheduler improves both significantly on this tier, but this one improves the rate at no measured cost in time.
