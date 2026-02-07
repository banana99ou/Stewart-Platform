## Making the Stewart Platform Accurate (practical calibration + validation)

This platform is **open-loop** (RC servos, no encoders). So “accuracy” mainly depends on:

- **PWM endpoint calibration** (servo travel scaling)
- **per-servo center + direction** (zero bias and mirroring)
- correct **geometry constants** (anchors, horn, linkage length)
- mechanical stiffness / low backlash
- measuring achieved attitude with an **external reference** (inclinometer / IMU / camera)

If any of these are off, the IK math can be perfect and the platform will still be wrong.

---

## 0) Define the accuracy target

For conference-grade results, pick at least one:

- **Static attitude error** (roll/pitch/yaw): mean / RMS / max over a 2–5 s hold
- **Repeatability**: command the same pose N times, report std-dev + hysteresis
- **Dynamic tracking**: step/sine (phase lag + RMS)

---

## 1) Calibration order (do this first → saves days)

1. **Servo PWM endpoints** (scale): make sure the servo actually reaches its intended travel.
2. **Servo centers** (bias): calibrate `SERVO_CENTER_DEG[]` with the real horn/jig.
3. **Servo signs** (mirrors): set `SERVO_SIGN[]` so +model angle is physically consistent.
4. Only then: geometry fine-tuning / range / saturation.

Why: if PWM endpoints are too narrow, a +10° command might physically produce only +5–6° even though IK reports OK.

---

## 2) PWM endpoint calibration (the “my 10° command becomes 5°” trap)

In Arduino/ESP32 `Servo.write(0..180)` is just a **mapping** into the microsecond pulse range you pass to `attach(min_us, max_us)`.
It does **not** guarantee 180° of mechanical travel.

### Procedure (safe)

- **Disconnect linkages** (or remove upper plate) so the servo is unloaded.
- Start with conservative endpoints, then expand slowly:
  - Example safe-ish start: 900–2100 µs
  - Common full-range target (varies): ~500–2500 µs
- Increase range until:
  - travel stops increasing meaningfully, OR
  - you hit a hard stop (buzzing/stall/heating) → back off immediately.

After changing PWM endpoints, **recalibrate** `SERVO_CENTER_DEG[]` because the meaning of `write(deg)` changes.

---

## 3) Per-servo zero bias and direction

The firmware supports per-servo calibration:

- `SERVO_CENTER_DEG[i]`: absolute `servo.write()` degrees corresponding to model 0°
- `SERVO_SIGN[i]`: +1 or -1 to fix mirrored installation

### Practical method (recommended)

- Unloaded (best):
  - Use serial commands:
    - `SET i deg` → absolute `servo.write(deg)`
    - `SWEEP i [start end step delay_ms]` → find the exact mechanical zero
    - `MODEL i a_deg` → command in model-angle space (center + sign)
  - Align each horn to your 0° reference using a printed jig / angle gauge.
  - Record the `deg` and set it in `SERVO_CENTER_DEG[]`.
  - Verify direction: `MODEL i +10` should move the horn in the expected direction; otherwise flip `SERVO_SIGN[i]`.

- Assembled (ok for ~1–3° goals):
  - Command a level pose, measure roll/pitch, and nudge a few `SERVO_CENTER_DEG[]` entries by ±1° until level.
  - Always approach the final pose from the **same direction** (reduces backlash effects).

---

## 4) Measuring platform attitude (roll/pitch/yaw)

### Roll/Pitch (recommended)

- **Digital inclinometer** on the top plate.
  - Always place it at the same spot and orientation.
  - Zero it on the same surface the base sits on (as you did).

### Yaw (harder)

- Magnetometer yaw can be biased by servos/wires and local fields.
- For ~3° yaw accuracy, prefer optical/reference methods:
  - printed protractor ring + pointer, or
  - AprilTag/ArUco on the plate + a fixed camera.

---

## 5) Validation test set (minimal but meaningful)

For roll/pitch accuracy:

- `0 0 0 0 0 0` (level)
- `0 0 0 +5 0 0`, `0 0 0 -5 0 0`
- `0 0 0 0 +5 0`, `0 0 0 0 -5 0`
- repeat each pose 10× (measure repeatability + hysteresis)

If measured error depends on whether you approached the pose from +/−, that’s backlash/stiction/torque dependence (normal for RC servos).

