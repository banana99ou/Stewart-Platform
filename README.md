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

### 2.2 Host interface layer (Python baseline + optional Simulink model)
**Baseline (used for current datasets):** `hils_host.py` (single-process Python host) integrates:
- X-Plane UDP interface (state receive + control injection)
- Stewart MCU serial interface (pose command + ACK/status)
- PX4 MAVLink interface (ATTITUDE stream for physical IMU/estimator)
- Logging + post-processing hooks (tracking error, latency, saturation)

**Optional / legacy:** `matlab/XplaneControlInterface.slx` (Simulink model) exists as an alternate integration path.

### 2.3 Stewart Platform (RC-servo actuators)
- Executes commanded pose within mechanical/servo constraints
- **Current hardware has no encoder/pose feedback** (RC airplane grade servos).
- Low priority future work (status flags like clamp/near-limit/watchdog) is tracked in `ToDo.md`.

### 2.4 Flight Controller: Pix32 v6 with PX4
- Mounted on platform
- Uses onboard IMU + estimator
- Communicates via USB serial MAVLink to the host

**Control-structure note (important):**
- In the current baseline, PX4 is used primarily as a **physical IMU/estimator** (ATTITUDE feedback).
- **Control injection into X-Plane is performed by a custom host-side PID** (see `hils_host.py`), not by PX4’s onboard attitude controller.

---

## 3) Interfaces and Protocols (explicit)
| Link | Physical | Protocol | Notes |
|---|---|---|---|
| X-Plane ↔ Host (`hils_host.py`) | Ethernet/loopback | **UDP** | X-Plane state out + control in |
| Host ↔ Stewart MCU | USB | **USB serial** | pose command + ACK/status |
| PX4 (Pix32 v6) ↔ Host | USB | **MAVLink over USB serial** | attitude estimate (ATTITUDE) + (optional) status |

---

## 3.1) X-Plane UDP: ports + packet format (critical)

### 3.1.1 Default ports (Python host baseline)
- **X-Plane → Host (state)**: `127.0.0.1:49004` (`--xplane-rx-port 49004`)
- **Host → X-Plane (controls)**: `127.0.0.1:49000` (`--xplane-tx-port 49000`)

### 3.1.2 X-Plane settings checklist (known-good)
In **X-Plane → Settings → Data Output → Network configuration**:
- Enable at least one **DATA output row** that contains attitude (commonly group 17).
- Set destination to **`127.0.0.1:49004`** (host RX port).

In **X-Plane → Settings → Network → UDP ports**:
- Set **Port we receive on** (and **legacy**) to the host TX port (commonly **49000**).
- Set **Port we send from (legacy)** to a value that works on your machine.
  - If Windows reports bind/port conflicts, make this **different** from the host RX port (e.g. 49005).

### 3.1.3 Packet formats (what actually mattered for integration)

**X-Plane DATA record layout** (little-endian):
- **record** = `int32 index` + `8 * float32` (36 bytes)
- `-999.0` means “no command / ignore” for that field.

**Host → X-Plane control injection** (what broke vs Simulink until fixed):
- Header must be **`DATA0`** (ASCII `'0'` = byte **48**), not `DATA\\0`.
- The packet is:
  - `b"DATA0"` (5 bytes) +
  - record index **8** (surfaces) +
  - record index **25** (throttle)

This matches the Simulink Byte Pack stream:
`68 65 84 65 48 ...` (ASCII `DATA0`).

**X-Plane → Host state receive**:
- The host accepts payloads starting with `b"DATA"` and auto-detects header length **4 vs 5** bytes by checking whether the remainder is a multiple of 36.

## 4) High-Level Data Flow (Closed Loop)
1. X-Plane → (UDP) → Host: simulated attitude/motion
2. Host → (USB serial) → Stewart MCU: pose command → Stewart moves
3. Stewart motion → PX4 IMU senses real motion → PX4 estimates attitude
4. PX4 → (MAVLink USB) → Host: attitude estimate (physical IMU proxy)
5. Host → (UDP) → X-Plane: inject control surface commands (host-side PID using PX4 attitude)
6. Repeat

---

## 4.1 Reproducibility / versions (dataset provenance)
To reproduce the results, please check out commit `49247330b62a792a53e6f2d933b44067f9b338b4`—this is the exact code version used to generate all reported figures and metrics.

## 4.2 Python environment
- **Python**: 3.8.3 (Windows; `C:\\Python38\\python.exe`)
- **Dependencies**: see `requirements.txt`

```bash
pip install -r requirements.txt
```

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

#### 6.1.0 Baseline sim scenario (as-tested; include in paper)
- **X-Plane**: 12.3.3-r1
- **Aircraft**: Cirrus Vision SF50
- **Failure**: none
- **CG**: -1.2 inches
- **Total payload**: 724.5 lbs
- **Total fuel weight**: 577.3 lbs
- **Location / start condition**: Kaaeloa (PHJR), Runway 04R, 10 nm approach
- **Weather**: clear
- **Time**: Jan 26th, 18:21 local
- **Controls**: no joystick, no autopilot
- **UDP output rate**: 50 packets/sec
- **Video**: 3 side-by-side recordings captured (X-Plane + platform)
- **Best run (for figures)**: `logs/run-20260208-210808 #8/`

#### 6.1.1 Test protocol (field checklist)
This project’s baseline “trim transition” dataset is intended to be repeatable enough for conference figures, even if the X-Plane reset is manual.

- **Pre-flight setup**
  - Connect hardware: Stewart platform (ESP32), PX4 (Pix32 v6), host PC.
  - Boot X-Plane and load the desired initial condition (e.g., “Hawaii / Honolulu 10 nm approach” scenario).
  - Ensure X-Plane UDP output is enabled at the expected rate (commonly 50 Hz) and configured to send `DATA*` to the host RX port.

- **Run protocol (manual repeat — simplest / what we actually use most often)**
  - Pause the sim (optional; depends on your X-Plane UDP behavior).
  - Start the host:

```bash
python hils_host.py --scenario trim-transition --scenario-warmup-s 10 --scenario-hold-s 10 --scenario-step-pitch-deg 5
```

  - Unpause the sim.
  - Observe: the host holds pitch setpoint at 0°, steps to +5°, then returns to 0° (time-based), and exits cleanly with logs.
  - Reset the sim back to the same initial condition.
  - Re-run the same command for N=3–5 repeats (each repeat is a fresh program run and a fresh log folder).

- **Run protocol (scripted repeats — untested)** \
If you want the host to manage repeat folders in a single session, use `--scenario-repeats` and optionally wait for manual X-Plane reset between repeats:

```bash
python hils_host.py --scenario trim-transition --scenario-warmup-s 10 --scenario-hold-s 10 --scenario-step-pitch-deg 5 --scenario-repeats 5 --repeat-wait-enter
```

- **Expected outputs (per run folder under `logs/run-*/`)**
  - `tick.csv`: host-tick snapshots (includes scenario phase + pitch setpoint when scenario mode is used)
  - `xplane_att.csv`: raw X-Plane receive samples
  - `px4_att.csv`: raw PX4 receive samples
  - `stewart_ack.csv`: per-command ACK/status from the ESP32 paired with sent commands
  - `run_meta.json`: run configuration + scenario parameters
  - `integrity.json`: lightweight integrity summary (tick gaps, sample presence, ACK health)
  - Postprocess outputs (if enabled): `postprocess_*.png`, `postprocess_*_metrics.json`

- **Minimum sanity checks (per repeat)**
  - X-Plane + PX4 streams are present (not empty).
  - ESP32 ACKs are present (`ack_total > 0`), and queue depth does not grow without bound.
  - The pitch setpoint step is visible in the logs/plots (`sp_pitch_deg`).

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
- `dt` / `Ts`: Simulink base sample time (if using Simulink path; often 100 Hz)
- `f_tick`: Python host control tick (`hils_host.py --tick-hz`, current default **50 Hz**)
- `f_xplane`: X-Plane DATA output rate (commonly **50 packets/sec** in current datasets)
- `f_pose`: effective Stewart pose update rate (≈ host tick when the host sends one pose per tick; bounded by serial + IK compute)
- `f_pwm`: servo PWM rate on ESP32 (**50 Hz**, set via `setPeriodHertz(50)` in firmware)
- `f_mav`: PX4 MAVLink ATTITUDE stream request rate (`--px4-att-hz`, default **100 Hz**)

### 7.2 Latency measurement plan (must-do for conference)
Implementation status + any refinements are tracked in `ToDo.md`. Measurement targets:
- X-Plane → host input latency
- Host compute latency
- Host → Stewart command latency
- Stewart motion → PX4 IMU response latency
- PX4 attitude output → host receive latency
- Host injection → X-Plane response latency

Deliverables:
- per-layer latency time series + summary stats (mean/RMS/max/std)
- end-to-end loop delay estimate

---

## 8) TODOs / Task tracking
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

### 8.5 Rotation-axis offset (MEDIUM)
- implement configurable rotation center offset in kinematic mapping
- document calibration procedure

### 8.6 Saturation/status flags (LOW)
- command clamp/near-limit flag
- watchdog timeout flag
- log into dataset

---

## 9) Acceptance Criteria
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