# ToDo (Stewart Platform Physical HILS)

**Canonical task tracker** (this file supersedes the TODO section that used to live in `README.md`).

Priorities:
- **P0**: Blocking for “physical closed-loop HILS” demo + KSAS-required artifacts
- **P1**: Strengthens validity/quantitative results
- **P2**: Nice-to-have / robustness polish

## P0 — Do next (in dependency order)

### P0.1 — Define + automate the baseline scenario (enables everything below)
- [X] **Implement trim transition test scenario**
  - [X] Scripted profile: hold → step (0→+5° pitch) → hold → return (optional)
  - [X] Repeat N=3–5 runs automatically with consistent initialization
  - [X] Save run name + scenario parameters in run_meta.json
- [X] **Run the primary experiment: trim transition (baseline)**
  - [X] Scenario definition (initial condition + input command; e.g., pitch step 0° → +5°)
  - [X] Logging: X-Plane attitude, PX4 attitude estimate, actuator outputs
  - [X] Repeat 3–5 times for conference-grade results

### P0.2 — Collect the conference baseline dataset (depends on P0.1)
- [X] **Trim transition dataset (conference baseline)**
  - [X] Run pitch step (e.g., 0°→+5°) for N=3–5 repeats with identical initial conditions
  - [X] Export one “best” run for figures + full set for statistics
  - [x] Best-looking run (pitch overlay + lowest `pitch_px4_minus_xp_deg` RMS among runs #5–#9): `logs/run-20260208-210808 #8/`

### P0.3 — Post-process + compute paper numbers (depends on P0.2)
- [ ] **Time alignment method (so “error” is meaningful)**
  - [ ] Estimate effective delay between X-Plane and PX4 attitude (e.g., cross-correlation) and compensate before computing stats
  - [ ] Report both “raw” and “aligned” error summary (at least in internal plots)
- [ ] **Add per-run PX4:stewart mounting bias to Postprocessing_run.py**
  - [ ] Estimate roll/pitch bias using the `hold_0` window (or another steady window) per run
  - [ ] Report both raw and bias-corrected tracking metrics (mean/RMS/max/std)
- [X] **Edit post-processing to show saturation statistics**
  - [X] % time saturated (sat flag)
  - [X] α distribution (histogram + summary stats)
  - [X] Error conditioned on α=1 vs α<1 (mean/RMS/max/std)
- [ ] **Metric pack (numbers to quote in text)**
  - [ ] Attitude error per axis: mean / RMS / max / std (after time alignment)
  - [ ] End-to-end latency: mean / RMS / max / std
  - [ ] Saturation rate: % time saturated + distribution of α
- [ ] **Saturation impact analysis**
  - [ ] Condition metrics on α=1 vs α<1 (error/latency differences)
  - [ ] Plot α(t) alongside command magnitude to show graceful degradation

### P0.4 — Turn results into conference artifacts (depends on P0.2–P0.3)
- [ ] **Figure pack (minimum set for abstract/paper)**
  - [ ] Fig. 1: system block diagram (protocols + rates annotated)
  - [ ] Fig. 2: time series overlay — X-Plane attitude vs PX4 attitude (+ error plot)
  - [ ] Fig. 3: latency breakdown (end-to-end + per-layer if available)
  - [ ] Table 1: update rates + interfaces (UDP/USB serial/MAVLink) + payload formats
- [ ] **Define the “result narrative” in 3 claims (one paragraph)**
  - [ ] Physical closed-loop HILS is achieved (sim → motion → IMU/FC → sim)
  - [ ] Tracking fidelity is quantified (error stats) within feasible envelope
  - [ ] Limitations are explained by latency + saturation + actuation limits (not hand-wavy)
- [ ] **Produce KSAS-required artifacts**
  - [ ] Scenario description write-up
  - [ ] Side-by-side visualization (X-Plane capture + real platform video)
  - [ ] Time-series plots (angles + angular rates if available)
  - [ ] Limitations/special findings: saturation, latency/jitter, missing aerodynamics, anomalies

### P0 — Completed prerequisites (already done)
- [X] **Implement PX4 ↔ Simulink MAVLink USB interface**
  - ~~[ ] Parse actuator outputs (primary)~~
  - [X] Parse attitude estimate (using custom PID controller instead)
  - ~~[ ] Parse arming/mode/status (recommended)~~
- [X] **Close the loop end-to-end**
  - [X] Route PX4 actuator outputs → Simulink → inject into X-Plane controls
  - [X] Verify “vehicle responds” acceptance criterion with PX4 in the loop
- [X] **Measure platform actuation range (roll/pitch/yaw) + define safe envelope**
  - [X] Document limits in degrees
  - [X] Implement + document clamp behavior when command exceeds range
  - [X] Ensure clamping is logged for later analysis (ACK `sat` + `alpha` + command pairing)
- [X] **Latency measurement (end-to-end, host clock)**
  - [X] X-Plane → host sample age
  - [X] PX4 → host sample age
  - [X] Host → ESP32 serial RTT (pose send → ACK rx)
  - [X] X-Plane RX → ACK (end-to-end proxy)
  - [X] PX4 RX → ACK (end-to-end proxy)
  - [X] Produce latency time series + summary stats (mean/RMS/max/std)
- [X] **Error + latency plotting/analysis pipeline**
  - [X] Angles vs time; (if available) angular rates vs time
  - [X] Tracking error time series + stats (mean/RMS/max/std)
  - [X] Latency plots + stats (end-to-end proxies)
- [X] **Logging checklist (minimum signals + timestamps)**
  - [X] Timestamp every stream at the receiver (host) with a common timebase
  - [X] Log X-Plane attitude (roll/pitch/yaw), PX4 attitude estimate
  - [X] Log platform command (pose/attitude) sent to ESP32
  - [X] Log saturation/clamp state: flag + scaling factor α (via ESP32 ACK)
  - [X] Log dropped packets / parse-fail counters (robustness) (at least host-side)

## P1 — Validation + credibility improvements

- [ ] **IK accuracy test (proxy via PX4 IMU since no encoders)**
  - [ ] Static pose holds at multiple setpoints
  - [ ] Quantify repeatability/hysteresis
  - [ ] Document methodology + results
- [ ] **Document system architecture + comms + timing (Q&A readiness)**
  - [ ] Block diagram of X-Plane / Simulink / Stewart MCU / PX4 data flow
  - [ ] Protocol summary (UDP, USB serial, MAVLink over USB serial)
  - [ ] Identify update rates: Simulink `dt/Ts`, effective command rate, X-Plane rate, MAVLink rate
  - [ ] Coordinate/rotation representation and transforms (Euler/quaternion etc.)
  - [ ] Initial calibration: reference attitude definition + procedure
- [ ] **Optional experiment: controller comparison (PID vs sliding mode, etc.)**
  - [ ] Define metrics: overshoot, settling time, steady-state error
  - [ ] Choose implementation approach (outer-loop in Simulink vs full surface-command controller)
  - [ ] Produce quantitative plots + side-by-side physical behavior evidence

### P1 — Extra analyses (high insight, low risk)
- [ ] **Repeatability / hysteresis**
  - [ ] Repeat same trim transition N times; plot ensemble mean ± std band
  - [ ] Report run-to-run variation (std, max-min)
- [ ] **Cross-axis coupling**
  - [ ] Command pure pitch/roll; quantify leakage into other axes (coupling matrix estimate)
  - [ ] Compare coupling in mid-range vs near workspace boundary
- [ ] **Bandwidth characterization (optional but strong)**
  - [ ] Small-amplitude sine input sweep (no saturation); estimate gain/phase vs frequency
  - [ ] Quote effective attitude bandwidth / phase lag
- [ ] **Robustness stats**
  - [ ] Report packet drop, parse-fail, watchdog counts over long run (e.g., 10–30 min)

## P2 — Nice-to-have / robustness

- [ ] **Rotation axis / center-of-rotation offset feature**
  - [ ] Implement configurable rotation center offset in kinematic mapping
  - [ ] Document calibration procedure
- [ ] **Add lightweight status flags (no encoders)**
  - [ ] Command clamp / saturation indicator
  - [ ] “Near limit” indicator
  - [ ] Watchdog / update timeout indicator
  - [ ] Log flags into dataset

