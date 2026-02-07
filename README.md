# project_spec.md — Stewart Platform Physical HILS (KSAS Spring Conference)

## 0) Publication Target
This project is being developed for **presentation at the KSAS (Korean Society for Aeronautical & Space Sciences) Spring Conference (춘계 항공우주학회)**. The build and experiments are scoped to produce a conference-grade demo + quantitative results (tracking error, latency, limitations).

---

## 1) Purpose and Differentiation
### 1.1 Purpose
Build a **physical Hardware-in-the-Loop Simulation (HILS)** testbed where **X-Plane-simulated vehicle attitude/motion** is realized physically by a **Stewart platform**, so that a **real Flight Controller (Pix32 v6, PX4)** mounted on it experiences **real IMU excitation**.

### 1.2 Differentiation vs Conventional HILS
This system differentiates itself by having **an actual flight controller and IMU experience real motion**:
- **Conventional HILS:** FC receives simulated sensor inputs; attitude is typically verified via display/logs only.
- **This system:** simulated attitude is realized physically → **FC IMU senses real motion** → enables more realistic closed-loop behavior (within platform limits).
- **Limits:** attitude range is bounded; aerodynamic forces are not physically reproduced → complements, not replaces, conventional HILS.

### 1.3 Value Proposition
- More intuitive physical monitoring of attitude during controller validation
- Training/demo/education potential
- Preliminary validation tool before real flight testing
- Extensible to broader “vehicle” class (e.g., aircraft/UAV and future extension)

---

## 2) System Components
### 2.1 X-Plane
- Real-time vehicle dynamics simulation
- Provides attitude/motion signals and accepts control inputs

### 2.2 Simulink Interface Layer (single model file)
A **single Simulink model** integrates:
- X-Plane UDP interface (state extraction + control injection)
- Stewart command generation + platform control
- PX4 MAVLink interface (USB)
- Logging hooks (tracking error + latency)

**Simulink sample time (`dt` / `Ts`)**: the discrete execution/update interval of the Simulink model (how often blocks run and I/O updates).

### 2.3 Stewart Platform (RC-servo actuators)
- Executes commanded pose within mechanical/servo constraints
- **Current hardware has no encoder/pose feedback** (RC airplane grade servos).
- Low priority future work (status flags like clamp/near-limit/watchdog) is tracked in `ToDo.md`.

### 2.4 Flight Controller: Pix32 v6 with PX4
- Mounted on platform
- Uses onboard IMU + estimator
- Outputs control surface commands
- Communicates via USB serial MAVLink to Simulink host

---

## 3) Interfaces and Protocols (explicit)
| Link | Physical | Protocol | Notes |
|---|---|---|---|
| X-Plane ↔ Simulink | Ethernet/loopback | **UDP** | X-Plane state out + control in |
| Simulink ↔ Stewart MCU | USB | **USB serial** | command + (optional) status |
| PX4 (Pix32 v6) ↔ Simulink | USB | **MAVLink over USB serial** | actuator outputs + attitude + status |

---

## 4) High-Level Data Flow (Closed Loop)
1. X-Plane → (UDP) → Simulink: simulated attitude/motion
2. Simulink → (USB serial) → Stewart MCU: pose command → Stewart moves
3. Stewart motion → PX4 IMU senses real motion → PX4 computes actuator outputs
4. PX4 → (MAVLink USB) → Simulink: actuator outputs (+ attitude estimate)
5. Simulink → (UDP) → X-Plane: inject control surface commands
6. Repeat

---

## 5) Conference Paper Requirements (must include)
The KSAS paper/presentation must include:

1) **Scenario description**
   - initial condition
   - input command (e.g., pitch step 0° → +5°)

2) **Side-by-side visualization**
   - X-Plane capture
   - real platform video (recommended side-by-side layout)

3) **Time series plots**
   - angles (roll/pitch/yaw as applicable)
   - angular rates (p/q/r if available)

4) **Special findings / limitations**
   - range saturation behavior
   - latency effects / jitter
   - missing physical aerodynamics
   - any observed anomalies

---

## 6) Experiments
### 6.1 Primary: Trim transition (Option A baseline)
- Example: pitch reference step 0° → +5°
- Log:
  - X-Plane attitude (simulation)
  - PX4 attitude estimate (physical IMU proxy)
  - actuator outputs (elevator/etc.)
- Report tracking error stats (Mean / RMS / Max / Std)

### 6.2 Optional: Controller comparison experiment (Option B)
Optional controller comparison experiment: **compare PID vs sliding mode control** (and/or vector field guidance, if applicable) using:
- **overshoot**
- **settling time**
- **steady-state error**
(plus physical behavior/video evidence)

Implementation note (two feasible approaches):
- Outer-loop comparison in Simulink (PX4 remains inner stabilization), OR
- Full surface-command controllers in Simulink using PX4 IMU state (higher risk)

---

## 7) Timing, Rates, and Latency Measurement (required)
### 7.1 Definitions
- `dt` / `Ts`: Simulink base sample time = 100Hz
- `f_cmd`: effective Stewart command update rate (often ≈ 1/dt, but must be measured)
- `f_xplane`: X-Plane update/output rate
- `f_mav`: MAVLink message/actuator output rate

### 7.2 Latency measurement plan (must-do for conference)
Implementation status + any refinements are tracked in `ToDo.md`. Measurement targets:
- X-Plane → Simulink input latency
- Simulink compute latency
- Simulink → Stewart command latency
- Stewart motion → PX4 IMU response latency
- PX4 actuator output → Simulink receive latency
- Simulink injection → X-Plane response latency

Deliverables:
- per-layer latency time series + summary stats (mean/RMS/max/std)
- end-to-end loop delay estimate

---

## 8) Current Status
### 8.1 Working
- Stewart platform ↔ Simulink ↔ X-Plane integration is working.

### 8.2 Not yet implemented / missing artifacts
This section easily goes stale; the canonical, up-to-date status and remaining work live in `ToDo.md`.

---

## 9) TODOs / Task tracking
The canonical TODO list (with priorities + completion status) is maintained in `ToDo.md`.

---

## Tracking error tester (PX4 + Stewart, yaw-zeroed)

If you want to test for Euler coupling (e.g. “pitch doesn’t look like pitch when yaw is high”), use:

```bash
python tracking_error_tester.py --stewart-com COM13 --px4-com COM9
```

What it does:
- Connects to PX4 MAVLink `ATTITUDE`
- Captures the first PX4 yaw as **yaw0**, then logs **yaw_rel = wrap(yaw - yaw0)**
- Commands roll/pitch steps at multiple yaw angles
- Writes `summary.csv` + raw samples under `logs/run-*/`

### 9.5 Rotation-axis offset (MEDIUM)
- implement configurable rotation center offset in kinematic mapping
- document calibration procedure

### 9.6 Saturation/status flags (LOW)
- command clamp/near-limit flag
- watchdog timeout flag
- log into dataset

---

## 10) Acceptance Criteria
### A) Bring-up (minimum viable)
1. PC detects Pix32 v6 over USB reliably.
2. Simulink receives PX4 actuator outputs in real time.
3. Simulink injects those outputs into X-Plane and the vehicle responds.

### B) KSAS conference-grade
1. Stable trim-hold segment
2. Trim transition succeeds (target 3–5 repeats)
3. Required paper artifacts produced:
   - scenario description
   - side-by-side video
   - angles + angular rates plots
   - tracking error + latency plots + limitations discussion
4. Platform range documented; saturation behavior shown

# narrative
Claim 1 (integration): “We built a physically closed-loop HILS where simulator dynamics drive real motion, a real FC/IMU responds, and its outputs affect the simulator again.”
Claim 2 (fidelity): “Within the platform’s feasible workspace, the physical motion tracks the simulator command with quantifiable error, and we can explain the error by latency, saturation, and actuator limits.”
Claim 3 (usefulness): “This testbed is useful for (i) pre-flight validation, (ii) education/demo, and (iii) extending to other aerospace vehicles (space/reentry) because the loop structure + evaluation method generalize.”