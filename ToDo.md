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
- [X] **Add per-run PX4:stewart mounting bias to Postprocessing_run.py**
  - [X] Estimate roll/pitch bias btw PX4 and platform pose using the `hold_0` window (or another steady window) per run
  - [X] Report both raw and bias-corrected tracking metrics (mean/RMS/max/std)
- [X] **Edit post-processing to show saturation statistics**
  - [X] % time saturated (sat flag)
  - [X] α distribution (histogram + summary stats)
  - [X] Error conditioned on α=1 vs α<1 (mean/RMS/max/std)
- [ ] **Metric pack (numbers to quote in text)**
  - [ ] Attitude error per axis: mean / RMS / max / std (after time alignment)
  - [X] End-to-end latency: mean / RMS / max / std
  - [X] Saturation rate: % time saturated + distribution of α
- [X] **Saturation impact analysis**
  - [X] Condition metrics on α=1 vs α<1 (error/latency differences)
  - [X] Plot α(t) alongside command magnitude to show graceful degradation

### P0.4 — Turn results into conference artifacts (depends on P0.2–P0.3)
- [ ] **Figure pack (minimum set for abstract/paper)**
  - [X] Fig. 1: system block diagram (protocols + rates annotated)
  - [X] Fig. 2: time series overlay — X-Plane attitude vs PX4 attitude
  - [X] Fig. 3: pitch tracking error time series (kept as a separate figure rather than appended to Fig. 2)
  - [ ] Latency breakdown figure: only latency time-series/distribution figures exist (`Fig5a`, `Fig5b`); no per-layer breakdown figure found
  - [ ] Table 1: update rates + interfaces (UDP/USB serial/MAVLink) + payload formats as a dedicated table
  - [ ] Figure source data in repo: `paper/fig/FigGen_manifest.json` points to `/Volumes/SHGP31-5/...`, and `logs/` is absent in this checkout
- [ ] **Define the “result narrative” in 3 claims (one paragraph)**
  - [ ] Physical closed-loop HILS is achieved (sim → motion → IMU/FC → sim)
  - [ ] Tracking fidelity is quantified (error stats) within feasible envelope
  - [ ] Limitations are explained by latency + saturation + actuation limits (not hand-wavy)
- [ ] **Produce KSAS-required artifacts**
  - [ ] Scenario description write-up
  - [ ] Side-by-side visualization (X-Plane capture + real platform video)
  - [ ] Time-series plots (angles + angular rates if available)
  - [ ] Limitations/special findings: saturation, latency/jitter, missing aerodynamics, anomalies

### P0.5 — Presentation revision per PI feedback (2026-03-20)

See `doc/Prof_feedback_2026-03-20_presentation.md` for full text.

- [ ] **Decide framing direction** (PI says either is fine — pick whichever is easier to discuss with audience)
- [ ] **Storytelling focus: move detail into reference tables**
  - [ ] Consolidate specs (working area, z position, saturator, rates/interfaces) into summary table slide(s)
  - [ ] Strip narrated detail from script; oral delivery should focus on project story + audience interest
- [ ] **Add photos + scenario-driven demo video**
  - [ ] Record trim-transition run video with X-Plane side-by-side
  - [ ] Video should narrate specific scenario motion, not just "look it moves"
  - [ ] Add hardware photos to slides where currently missing
- [ ] **Fix font / readability**
  - [ ] Switch to bolder, projector-robust font in `generate_presentation.py`
  - [ ] Increase base font sizes; only non-essential info stays small
- [ ] **Split slide 5 (system block diagram) into per-block detail slides**
  - [ ] Each block (X-Plane, Host/Simulink, Stewart Platform, PX4) gets its own slide
  - [ ] Each sub-slide: representative photo + data exchanged with other blocks
- [ ] **Improve graph legibility**
  - [ ] Increase line width in `Fig_Gen.py`
  - [ ] Add markers / dashed lines where traces overlap (especially Fig2 pitch overlay, Fig5a latency)
- [ ] **Add Korean title to cover slide**

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
  - [X] Static pose holds at multiple setpoints
  - [ ] Quantify repeatability/hysteresis
  - [ ] Document methodology + results
- [ ] **Document system architecture + comms + timing (Q&A readiness)**
  - [X] Block diagram of X-Plane / Simulink / Stewart MCU / PX4 data flow
  - [X] Protocol summary (UDP, USB serial, MAVLink over USB serial)
  - [X] Identify update rates: Simulink `dt/Ts`, effective command rate, X-Plane rate, MAVLink rate
  - [ ] Coordinate/rotation representation and transforms (Euler/quaternion etc.)
  - [ ] Initial calibration: reference attitude definition + procedure
- [ ] **Optional experiment: controller comparison (PID vs sliding mode, etc.)**
  - [ ] Define metrics: overshoot, settling time, steady-state error
  - [ ] Choose implementation approach (outer-loop in Simulink vs full surface-command controller)
  - [ ] Produce quantitative plots + side-by-side physical behavior evidence

### P1 — Extra analyses (high insight, low risk)
- [ ] **Repeatability / hysteresis**
  - [x] Repeat same trim transition N times; plot ensemble mean ± std band
  - [ ] Report run-to-run variation (std, max-min)
- [ ] **Cross-axis coupling**
  - [ ] Command pure pitch/roll; quantify leakage into other axes (coupling matrix estimate)
  - [ ] Compare coupling in mid-range vs near workspace boundary
- [ ] **Bandwidth characterization (optional but strong)**
  - [ ] Implement single-axis sine-sweep bandwidth measurement script
  - [ ] Use small unsaturated amplitude (about 0.5-2 deg) and sweep roughly 0.1-5 Hz
  - [ ] Log command and measured attitude on a common host clock (`cmd_pitch -> px4_pitch`, optional external reference)
  - [ ] Estimate gain/phase vs frequency and report effective attitude bandwidth (-3 dB) plus representative phase lag (e.g. 0.5 / 1 / 2 Hz)
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
- [ ] **Fix saturator
 - [ ] use span instead of worst angle

