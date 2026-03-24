# Chat notes (2026-02-09) — Stewart Platform Physical HILS paper drafting

This document captures the key decisions, rationale, and quantitative results discussed in chat while drafting the KSAS paper and updating analysis/figures.

## 1) Core narrative decisions

- **Baseline dataset**: runs **#5–#9** are treated as the “best runs” (post mounting fix). A representative “hero run” used for time-series figures is **run #8**.
- **Workspace / saturation**: baseline runs stay within the platform workspace; **saturation/clamping is not observed** in runs #5–#9 (0% saturated time/count in existing metrics JSON).
- **Residual tracking offset explanation**:
  - The “smooth, consistent ~2°” residual seen after run #5 is argued to be dominated by **FC mounting attitude bias** (fixed offset between FC/IMU frame and platform top-plate pose frame).
  - The **correct reference for mounting bias** is **platform pose**, not the simulator attitude.
  - Mounting-bias evidence is shown by comparing **PX4 attitude vs commanded platform pose** during a steady window (e.g., `hold_0`).

## 2) What figures/charts are needed to prove the claims

To support the paper’s claims with the collected logs (`tick.csv`, `stewart_ack.csv`) and existing figure outputs under `paper/fig/`, the minimal set is:

- **Fig. 1 — Architecture**: proves the “physical closed-loop HILS” integration.
  - File: `paper/fig/Fig1_Architecture.png`
- **Fig. 2 — Pitch response overlay**: shows the trim-transition experiment and the phase-lag/behavior.
  - File: `paper/fig/Fig2_PitchResponse.png`
  - Includes a trace where X-Plane pitch is shifted by the estimated mounting bias (visual alignment cue).
- **Fig. 4 — Mounting bias evidence**: direct evidence for FC mounting bias using PX4 vs commanded pose.
  - File: `paper/fig/Fig4_MountingBias_TimeSeries.png`
  - Shows pre-fix run vs hero run, with mean bias in `hold_0`.
- **Fig. 5(b) — Latency distribution**: justifies phase lag / bandwidth limits with measured timing.
  - File: `paper/fig/Fig5b_Latency_Distribution_Baseline.png`

Optional (use if space allows):

- **Fig. 3 — Pitch tracking error (raw vs bias-corrected)**:
  - File: `paper/fig/Fig3_PitchError.png`
- **Fig. 5(a) — Latency time series**:
  - File: `paper/fig/Fig5a_Latency_TimeSeries.png`

## 3) Metric definitions (latency layers)

All time quantities are measured on the **host** using a **monotonic clock**.

- **X-Plane sample age at tick**: `t_tick_ns - xp_t_rx_ns`
- **PX4 sample age at tick**: `t_tick_ns - px4_t_rx_ns`
- **Serial RTT**: `t_ack_rx_ns - t_cmd_send_ns` (pose send → ACK receive)
- **X-Plane rx→ACK (proxy)**: `t_ack_rx_ns - xp_t_rx_ns` (latest X-Plane sample used at that tick → ACK)
- **PX4 rx→ACK (proxy)**: `t_ack_rx_ns - px4_t_rx_ns` (latest PX4 sample used at that tick → ACK)

## 4) Bias estimation method (the “mounting bias”)

**Pitch mounting bias** for a run is defined as:

- `b_pitch = mean( wrap180(px4_pitch_deg - cmd_pitch_deg) )` over the steady window:
  - `scenario_phase == "hold_0"`
  - `pid_enabled == 1`
  - after `skip_first_s = 10.0` seconds

Bias-corrected PX4-vs-X-Plane pitch error:

- `wrap180( (px4_pitch_deg - b_pitch) - xp_pitch_deg )`

Notes:

- `cmd_pitch_deg` is the **commanded platform pose** (open-loop). This is still the most sensible reference for *mounting bias* given the current hardware.
- A stronger future version would use **external platform attitude measurement** (inclinometer/vision) as the reference.

## 5) Quantitative results used in the draft

### 5.1 Existing (raw) pitch tracking error vs X-Plane (from existing `postprocess_metrics.json`)

Metric: `pitch_px4_minus_xp_deg` over the analysis window (`skip_first_s = 10 s`).

| Run | Mean (deg) | RMS (deg) | P95 abs (deg) | Max abs (deg) |
|---|---:|---:|---:|---:|
| #5 | 2.0540 | 2.0963 | 2.5246 | 4.8690 |
| #6 | 2.1277 | 2.1830 | 2.8178 | 4.3592 |
| #7 | 1.9254 | 1.9630 | 2.4756 | 3.9897 |
| #8 | 1.6338 | 1.6827 | 2.1238 | 4.8438 |
| #9 | 2.0831 | 2.1427 | 2.7336 | 4.6308 |

### 5.2 Bias + bias-removed pitch error (computed directly from `tick.csv` for runs #5–#9)

Bias computed in `hold_0` as described above; error computed over the analysis window.

| Run | b_pitch mean±std (deg) | raw RMS (PX4−X‑Plane) | raw mean | bias-corrected RMS | bias-corrected mean |
|---|---:|---:|---:|---:|---:|
| #5 | 2.0084±0.2096 | 2.0963 | 2.0540 | 0.4214 | 0.0456 |
| #6 | 1.7460±0.2234 | 2.1830 | 2.1277 | 0.6198 | 0.3817 |
| #7 | 1.8265±0.2191 | 1.9630 | 1.9254 | 0.3951 | 0.0989 |
| #8 | 1.8185±0.2387 | 1.6827 | 1.6338 | 0.4432 | -0.1847 |
| #9 | 1.7282±0.2948 | 2.1427 | 2.0831 | 0.6147 | 0.3550 |

Interpretation:

- Raw pitch error is dominated by a relatively constant offset (~1.7–2.0°), consistent with mounting bias.
- After subtracting the estimated bias, RMS drops to ~0.4–0.6° in these runs (dynamic component).

### 5.3 Latency summary (runs #5–#9, from existing `postprocess_metrics.json`)

All units ms. Saturation is 0% in these baseline runs.

| Run | Serial RTT mean | Serial RTT P95 | Serial RTT max | PX4 rx→ACK mean | PX4 rx→ACK P95 | PX4 rx→ACK max | X‑Plane rx→ACK mean | X‑Plane rx→ACK P95 | X‑Plane rx→ACK max | Saturation (% time) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #5 | 21.7627 | 33.9212 | 65.0500 | 20.6623 | 33.2395 | 63.7002 | 30.7208 | 62.4697 | 80.0690 | 0.0 |
| #6 | 21.7398 | 33.8954 | 54.4494 | 20.6083 | 33.1112 | 46.1230 | 26.7176 | 48.9724 | 79.8520 | 0.0 |
| #7 | 21.5830 | 33.7010 | 50.0613 | 20.4293 | 32.9395 | 42.3585 | 26.4289 | 48.8591 | 65.1295 | 0.0 |
| #8 | 21.8969 | 33.5583 | 72.5507 | 20.7169 | 33.0030 | 66.0645 | 31.2639 | 62.7551 | 97.5157 | 0.0 |
| #9 | 21.6695 | 33.8642 | 81.4045 | 20.5575 | 33.0931 | 80.8785 | 29.3779 | 49.5338 | 104.9433 | 0.0 |

## 6) Implementation notes referenced during drafting

- **Host tick rate**: default is 50 Hz (even if older text/comments mention 100 Hz). Ensure the paper states the correct rates used in experiments.
- **Speed PID / throttle**: best-run metadata indicates speed-hold PID was enabled; paper should avoid implying throttle is constant unless you explicitly disabled it.

## 7) Files involved

- Paper draft: `paper/Paper_draft.md`
- Figure outputs: `paper/fig/Fig1_Architecture.png`, `paper/fig/Fig2_PitchResponse.png`, `paper/fig/Fig3_PitchError.png`, `paper/fig/Fig4_MountingBias_TimeSeries.png`, `paper/fig/Fig5a_Latency_TimeSeries.png`, `paper/fig/Fig5b_Latency_Distribution_Baseline.png`
- Figure generator: `paper/Fig_Gen.py` (reads CSV logs directly; estimates mounting bias from PX4 vs commanded pose in `hold_0`)
- Postprocess script: `postprocess_run.py` (metrics JSON + plots)

