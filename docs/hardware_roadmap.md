# SmartScan — Hardware Roadmap

How the simulated scheduler becomes a real one, and what changes when it does.

The claim this prototype makes is about **scheduling policy quality**, not RF
realism. This document states plainly what a real radio would change, which
numbers would move, and in which direction — so nothing here has to be
discovered on a bench at 2 a.m.

---

## 1. Why no scheduler code changes

Everything above the HAL consumes exactly one type,
`smartscan.env.types.Observation`. Everything below it deals in Hz and seconds.
The seam is `smartscan/hal/backend.py`:

```python
class ReceiverBackend(ABC):
    def tune(self, center_hz: float) -> None: ...
    def capture(self, duration_s: float) -> CaptureHandle: ...
    def get_detections(self, capture: CaptureHandle) -> list[Detection]: ...
    # + ibw_hz, tune_range_hz, settle_time_s, noise_figure_db
```

Two implementations share it:

| | `SimulatedBackend` | `SoapySDRBackend` |
|---|---|---|
| `tune` | sets an index into precomputed tensors | `Device.setFrequency`, then wait for LO lock |
| `capture` | advances the slot clock | `readStream` into a complex64 buffer |
| `get_detections` | reads a precomputed `Pd` realisation | Welch PSD → CA-CFAR → cluster → Hz |

Switching is one config line — `receiver.backend: soapy`. The belief, all nine
schedulers, the metrics and the benchmark harness are untouched.

`SoapySDRBackend` is **non-functional today and raises `NotImplementedError`
with the exact call it would make**. A stub that returns plausible-looking fake
data is worse than one that refuses, because it turns a missing driver into a
silently wrong result. `tests/test_env.py::test_soapy_backend_refuses_to_pretend`
asserts the refusal.

---

## 2. Candidate hardware

| Device | IBW | Tune range | LO settle (typ.) | Notes |
|---|---|---|---|---|
| RTL-SDR v4 | 2.4 MHz | 24–1766 MHz | ~0.3 ms | cheapest possible bring-up; VHF/UHF only |
| HackRF One | 20 MHz | 1 MHz–6 GHz | ~1 ms | half-duplex, 8-bit ADC |
| ADALM-Pluto | 20 MHz (56 with hack) | 70 MHz–6 GHz | ~0.5 ms | good IQ quality, cheap |
| USRP B210 | 56 MHz | 70 MHz–6 GHz | ~0.2 ms | best of the accessible options |

**None reaches 18 GHz.** The upper band needs a block downconverter, and its LO
settling time then dominates `t_settle`. That is a real constraint on any
deployed system, and it makes the retune cost *larger* on hardware than in
simulation — which strengthens rather than weakens the case for a scheduler that
economises on retunes.

### 2.1 A staged plan

**Stage 1 — L/S band, one device.** ADALM-Pluto or B210 over 1–2 GHz.
`B = 32` channels of 32 MHz, `K = 1`. Everything in the pipeline runs unchanged;
only the grid shrinks. Emitters: a signal generator on a known frequency, plus a
second generator gated to imitate a scanning beam.

**Stage 2 — measured `t_settle` and NF.** See §3. Until these are measured the
scheduling comparison is not meaningful, because `t_settle` *is* the economics.

**Stage 3 — CFAR detection chain.** Replace the analytic `Pd` with real signal
processing (§4), and re-validate `Pfa` against the design value.

**Stage 4 — wideband front end.** Block downconverter for 6–18 GHz, or a bank of
receivers. At this point `t_settle` grows and the ablation in
`eval/ablation.py::t_settle` becomes the design tool for choosing dwell length.

---

## 3. The two measurements that must come first

### 3.1 `t_settle` — measure, never assume

```
1. CW source at f1. Tune to f1, capture, confirm the tone.
2. Tune to f2 far away, then immediately back to f1.
3. Capture continuously and find when |X(f1)| returns to within 0.5 dB of its
   settled value. That interval is t_settle.
4. Repeat across the tune range: settling is worse for large jumps.
```

Set `receiver.t_settle_slots = ceil(t_settle / dt_s)`. Every scheduling
comparison in this project is conditioned on this number; a wrong value
invalidates all of them, which is why the stub's docstring says *MEASURE this,
do not assume*.

### 3.2 Noise figure — Y-factor

Calibrated noise source, ENR known. Measure output power with the source on and
off; `Y = P_on / P_off`, `F = ENR / (Y − 1)`. Set `receiver.noise_figure_db`.
Without this the sensitivity figure (metric 3) is fiction.

---

## 4. Replacing the analytic detector

The simulator draws detections from a closed-form `Pd`. Hardware needs the
actual chain, and it must be calibrated to the **same `Pfa`** so the scheduling
comparison stays apples-to-apples:

1. **Welch PSD**, `fft_size` bins, Hann window, 50 % overlap, `N` averages —
   matching `receiver.detector.fft_size` and the derived `n_integrate`.
2. **CA-CFAR** per bin with guard cells. Set the threshold multiplier from the
   design `Pfa` using the same Erlang tail the simulator uses
   (`propagation.detection_threshold`), so the two agree by construction.
3. **Cluster** adjacent above-threshold bins into single detections.
4. **Convert** bin index → absolute Hz using the tuned centre.

**Validation gate.** Terminate the input, run the chain for 10⁶ dwells, and
confirm the measured false-alarm rate matches the design `Pfa` to within a
factor of two. `tests/test_env.py::test_empirical_pfa_matches_configured` is the
simulated equivalent and should be mirrored on hardware.

---

## 5. What will be worse on real hardware, and why

Stated in advance rather than discovered later.

| Effect | Impact | Mitigation |
|---|---|---|
| **Non-Gaussian interference** — spurs, LO leakage, images, intermodulation | CFAR sees structured, not white, noise; `Pfa` rises above design | Ordered-statistic CFAR instead of CA-CFAR; a static spur mask learned during calibration |
| **AGC transients** | The first samples after a retune are unusable, effectively increasing `t_settle` | Fix the gain (no AGC) during a scan, or extend `t_settle` to cover the transient |
| **LO drift and phase noise** | Degrades the effective SNR of narrowband signals; wide phase-noise skirts mask weak neighbours | GPSDO reference; accept the SNR penalty in the link budget |
| **ADC dynamic range** (8-bit on HackRF) | A strong interferer desensitises the whole IBW — exactly the `Interferer` class, but worse | Front-end filtering; the `w5` interferer penalty already pushes the scheduler away, which helps |
| **USB/host throughput** | Caps sustained IBW and forces sample dropping | Reduce `fft_size` or the averaging count; both are config fields |
| **Real emitters are not Swerling I** | Detection statistics shift by a few dB either way | Re-fit `Pd` empirically per emitter class; the metric layer already reports empirical `Pd` per SNR bin (metric 1) |

### 5.1 Which reported numbers would move

* **Sensitivity (metric 3)** — worse by the measured NF minus the assumed 4 dB,
  plus implementation loss. Expect 3–8 dB degradation.
* **`Pfa` (metric 2)** — higher than design until spurs are masked.
* **TTFI and interception ratio** — the *ordering* of schedulers should hold,
  because a larger real `t_settle` penalises exactly the policies that hop most,
  which is the sweep. The *absolute* values will worsen for every policy.
* **Scan-period estimates** — should be unaffected: they depend on arrival
  timing, not amplitude, and the clustering step in `analysis/estimators.py` is
  robust to per-pulse detection dropouts.

---

## 6. Embedded deployment

The predictor exports to **ONNX (opset 17)** for embedded inference. The
analytic schedulers need no accelerator at all:

| Policy | Inference cost per dwell | Deployable on |
|---|---|---|
| `sequential`, `coprime_sweep` | a few arithmetic ops | any MCU |
| `ucb1`, `thompson`, `whittle` | O(B) numpy, microseconds | Cortex-A / small SoC |
| `predictor`, `ppo`, `hybrid` | one small CNN forward | SoC with NPU, or ONNX Runtime on CPU |

The strongest analytic policies (Whittle, phase-locked) run in microseconds with
no learned weights, so **the headline capability does not depend on shipping a
neural network**. That is a deliberate property of the design: the learned
agents are upside, not a dependency.

---

## 7. Bring-up checklist

- [ ] Install SoapySDR and the device driver; `SoapySDRUtil --find` sees the radio
- [ ] Implement the `TODO(hardware)` blocks in `smartscan/hal/soapy_stub.py`
- [ ] Measure `t_settle` (§3.1) → `receiver.t_settle_slots`
- [ ] Measure NF by Y-factor (§3.2) → `receiver.noise_figure_db`
- [ ] Implement and calibrate the CFAR chain (§4); validate `Pfa` on a terminated input
- [ ] Build a narrow-band config (`B`, `K`, `f_start_hz`, `f_stop_hz` for the device)
- [ ] Confirm a CW source is detected at the expected SNR — link budget sanity
- [ ] Gate a second source to imitate a scanning beam; confirm `estimate_period_ls`
      recovers the gating period
- [ ] Run `smartscan benchmark` on hardware and compare the scheduler ordering
      against the simulated leaderboard
