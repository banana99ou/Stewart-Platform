# ToDo (Stewart Platform Physical HILS)

Priorities:
- **P0**: Blocking for “physical closed-loop HILS” demo + KSAS-required artifacts
- **P1**: Strengthens validity/quantitative results
- **P2**: Nice-to-have / robustness polish

## P0 — Must do next (conference blockers)

- [ ] **Implement PX4 ↔ Simulink MAVLink USB interface**
  - [ ] Parse actuator outputs (primary)
  - [ ] Parse attitude estimate (recommended)
  - [ ] Parse arming/mode/status (recommended)
- [ ] **Close the loop end-to-end**
  - [ ] Route PX4 actuator outputs → Simulink → inject into X-Plane controls
  - [ ] Verify “vehicle responds” acceptance criterion with PX4 in the loop
- [ ] **Measure platform actuation range (roll/pitch/yaw) + define safe envelope**
  - [ ] Document limits in degrees
  - [ ] Implement + document clamp behavior when command exceeds range
  - [ ] Ensure clamping is logged for later analysis
- [ ] **Latency measurement (per-layer + end-to-end)**
  - [ ] X-Plane → Simulink input latency
  - [ ] Simulink compute latency
  - [ ] Simulink → Stewart command latency
  - [ ] Stewart motion → PX4 IMU response latency
  - [ ] PX4 actuator output → Simulink receive latency
  - [ ] Simulink injection → X-Plane response latency
  - [ ] Produce latency time series + summary stats (mean/RMS/max/std)
- [ ] **Error + latency plotting/analysis pipeline**
  - [ ] Angles vs time; (if available) angular rates vs time
  - [ ] Tracking error time series + stats (mean/RMS/max/std)
  - [ ] Latency plots + stats (per-layer + end-to-end estimate)
- [ ] **Run the primary experiment: trim transition (baseline)**
  - [ ] Scenario definition (initial condition + input command; e.g., pitch step 0° → +5°)
  - [ ] Logging: X-Plane attitude, PX4 attitude estimate, actuator outputs
  - [ ] Repeat 3–5 times for conference-grade results
- [ ] **Produce KSAS-required artifacts**
  - [ ] Scenario description write-up
  - [ ] Side-by-side visualization (X-Plane capture + real platform video)
  - [ ] Time-series plots (angles + angular rates if available)
  - [ ] Limitations/special findings: saturation, latency/jitter, missing aerodynamics, anomalies

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

## P2 — Nice-to-have / robustness

- [ ] **Rotation axis / center-of-rotation offset feature**
  - [ ] Implement configurable rotation center offset in kinematic mapping
  - [ ] Document calibration procedure
- [ ] **Add lightweight status flags (no encoders)**
  - [ ] Command clamp / saturation indicator
  - [ ] “Near limit” indicator
  - [ ] Watchdog / update timeout indicator
  - [ ] Log flags into dataset

