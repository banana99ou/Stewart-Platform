#include <ESP32Servo.h>
#include <math.h>
#include <stdio.h>
#include "geometry.h"

// ---------------- Servo HW ----------------

constexpr uint8_t NUM_SERVOS = 6;
constexpr int PWM_PULSE_MIN_US = 1000;
constexpr int PWM_PULSE_MAX_US = 2000;

const uint8_t servo_pins[NUM_SERVOS] = {4, 13, 16, 17, 18, 19};
Servo servos[NUM_SERVOS];

static inline int clampDeg(int d) {
  if (d < 0) return 0;
  if (d > 180) return 180;
  return d;
}

// ---------------- Math helpers ----------------

const float DEG2RAD = 3.14159265358979323846f / 180.0f;

static inline Vec3 v3(float x, float y, float z) { Vec3 r = {x, y, z}; return r; }

static inline Vec3 vAdd(const Vec3 &a, const Vec3 &b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 vSub(const Vec3 &a, const Vec3 &b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 vScale(const Vec3 &a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline float vDot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 vCross(const Vec3 &a, const Vec3 &b) {
  return v3(a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
}
static inline float vNorm(const Vec3 &a) { return sqrtf(vDot(a, a)); }
static inline Vec3 vNormed(const Vec3 &a) {
  float n = vNorm(a);
  if (n < 1e-6f) return v3(0, 0, 0);
  return vScale(a, 1.0f / n);
}

static inline Vec3 rotateZ(const Vec3 &p, float deg) {
  float rad = deg * DEG2RAD;
  float c = cosf(rad), s = sinf(rad);
  return v3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

static inline Vec3 rotateX(const Vec3 &p, float deg) {
  float rad = deg * DEG2RAD;
  float c = cosf(rad), s = sinf(rad);
  return v3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

static inline Vec3 rotateY(const Vec3 &p, float deg) {
  float rad = deg * DEG2RAD;
  float c = cosf(rad), s = sinf(rad);
  return v3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

static inline Vec3 translate(const Vec3 &p, float dx, float dy, float dz) {
  return v3(p.x + dx, p.y + dy, p.z + dz);
}

static inline Vec3 mirrorX(const Vec3 &p) {
  return v3(-p.x, p.y, p.z);
}

static inline Vec3 mat3Mul(const Mat3 &R, const Vec3 &v) {
  return v3(
    R.m[0][0] * v.x + R.m[0][1] * v.y + R.m[0][2] * v.z,
    R.m[1][0] * v.x + R.m[1][1] * v.y + R.m[1][2] * v.z,
    R.m[2][0] * v.x + R.m[2][1] * v.y + R.m[2][2] * v.z
  );
}

static inline Mat3 mat3Transpose(const Mat3 &R) {
  Mat3 Rt;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      Rt.m[i][j] = R.m[j][i];
  return Rt;
}

// ---------------- Geometry constants ----------------

const float LINK_LENGTH = 162.21128f; // mm (ball link center to center)

// Base geometry for servo 1 (same as Python)
// NOTE: name uses suffix to avoid Arduino's binary macros B0/B1.
const Vec3 B1_pos = {-50.0f, 100.0f, 10.0f};
const Vec3 H1     = {-85.0f, 108.0f + 3.3f / 2.0f, 10.0f};
const Vec3 U1     = { -7.5f, 108.0f + 3.3f / 2.0f, 152.5f};

Vec3 bottomAnchor[NUM_SERVOS];
Vec3 horn0Anchor[NUM_SERVOS];
Vec3 upperAnchor[NUM_SERVOS];

void generateCoordinates() {
  // servos 1,3,5: rotations of servo1
  float anglesZ[3] = {0.0f, 120.0f, 240.0f};
  for (int i = 0; i < 3; ++i) {
    int idx = i * 2; // 0,2,4
    bottomAnchor[idx] = rotateZ(B1_pos, anglesZ[i]);
    horn0Anchor[idx]   = rotateZ(H1, anglesZ[i]);
    upperAnchor[idx]  = rotateZ(U1, anglesZ[i]);
  }

  // servos 2,4,6: X-mirror of servo1 rotated
  Vec3 B1m = mirrorX(B1_pos);
  Vec3 H1m = mirrorX(H1);
  Vec3 U1m = mirrorX(U1);

  bottomAnchor[1] = rotateZ(B1m, 120.0f);
  horn0Anchor[1]   = rotateZ(H1m, 120.0f);
  upperAnchor[1]  = rotateZ(U1m, 120.0f);

  bottomAnchor[3] = rotateZ(B1m, 240.0f);
  horn0Anchor[3]   = rotateZ(H1m, 240.0f);
  upperAnchor[3]  = rotateZ(U1m, 240.0f);

  bottomAnchor[5] = B1m;
  horn0Anchor[5]   = H1m;
  upperAnchor[5]  = U1m;
}

// ---------------- Servo axis frames ----------------

Vec3 getServoAxis(int idx) {
  // axes lie in XY plane, radially outward
  Vec3 baseAxis = v3(0.0f, 1.0f, 0.0f);
  float angle;
  if (idx == 0 || idx == 5) {
    angle = 0.0f;
  } else if (idx == 1 || idx == 2) {
    angle = 120.0f;
  } else if (idx == 3 || idx == 4) {
    angle = -120.0f;
  } else {
    angle = 0.0f;
  }
  Vec3 a = rotateZ(baseAxis, angle);
  return vNormed(a);
}

Mat3 buildServoFrame(const Vec3 &axis) {
  Vec3 ez = axis;
  Vec3 ref = v3(0.0f, 0.0f, 1.0f);
  if (fabsf(vDot(ref, ez)) > 0.99f)
    ref = v3(1.0f, 0.0f, 0.0f);
  Vec3 ex = vNormed(vCross(ref, ez));
  Vec3 ey = vCross(ez, ex);

  Mat3 R;
  R.m[0][0] = ex.x; R.m[0][1] = ex.y; R.m[0][2] = ex.z;
  R.m[1][0] = ey.x; R.m[1][1] = ey.y; R.m[1][2] = ey.z;
  R.m[2][0] = ez.x; R.m[2][1] = ez.y; R.m[2][2] = ez.z;
  return R;
}

// ---------------- IK core ----------------

bool solveServoAngleZAxis(const Vec3 &B, const Vec3 &H0,
                          const Vec3 &Utarget, float L,
                          float &theta1Deg, float &theta2Deg) {
  Vec3 h = vSub(H0, B);
  Vec3 s = vSub(Utarget, B);

  float a = s.x * h.x + s.y * h.y;
  float b = -s.x * h.y + s.y * h.x;
  float gamma = 0.5f * (vDot(s, s) + vDot(h, h) - L * L);
  float gamma_p = gamma - s.z * h.z;

  float r = hypotf(a, b);
  if (r < 1e-6f) return false;

  float x = gamma_p / r;
  if (x < -1.0f || x > 1.0f) return false;

  float phi = atan2f(b, a);
  float d = acosf(x);

  theta1Deg = (phi + d) * 180.0f / 3.14159265358979323846f;
  theta2Deg = (phi - d) * 180.0f / 3.14159265358979323846f;
  return true;
}

bool pickSolution(float theta1, float theta2,
                  float minDeg, float maxDeg,
                  float preferDeg, float &outDeg) {
  float candidates[2];
  int n = 0;
  float ts[2] = {theta1, theta2};
  for (int i = 0; i < 2; ++i) {
    float t = ts[i];
    float tn = fmodf(t + 180.0f, 360.0f);
    if (tn < 0) tn += 360.0f;
    tn -= 180.0f;
    if (tn >= minDeg && tn <= maxDeg) {
      candidates[n++] = tn;
    }
  }
  if (n == 0) return false;
  float best = candidates[0];
  float bestDiff = fabsf(best - preferDeg);
  for (int i = 1; i < n; ++i) {
    float d = fabsf(candidates[i] - preferDeg);
    if (d < bestDiff) {
      bestDiff = d;
      best = candidates[i];
    }
  }
  outDeg = best;
  return true;
}

// Transform upper anchors by pose (same semantics as Python)
void transformUpperAnchors(const Vec3 src[NUM_SERVOS], Vec3 dst[NUM_SERVOS],
                           float dz, float rollDeg, float pitchDeg, float yawDeg,
                           float dx, float dy) {
  // center of plate
  Vec3 center = v3(0, 0, 0);
  for (int i = 0; i < NUM_SERVOS; ++i) center = vAdd(center, src[i]);
  center = vScale(center, 1.0f / NUM_SERVOS);

  Vec3 tmp[NUM_SERVOS];
  for (int i = 0; i < NUM_SERVOS; ++i) {
    tmp[i] = vSub(src[i], center);
    tmp[i] = rotateX(tmp[i], rollDeg);
    tmp[i] = rotateY(tmp[i], pitchDeg);
    tmp[i] = rotateZ(tmp[i], yawDeg);
    tmp[i] = vAdd(tmp[i], center);
    tmp[i] = translate(tmp[i], dx, dy, dz);
  }
  for (int i = 0; i < NUM_SERVOS; ++i) dst[i] = tmp[i];
}

void computeAllServoAngles(const Vec3 bottom[NUM_SERVOS],
                           const Vec3 horn0[NUM_SERVOS],
                           const Vec3 upperT[NUM_SERVOS],
                           float L, float minDeg, float maxDeg,
                           float outAngles[NUM_SERVOS],
                           bool outSuccess[NUM_SERVOS]) {
  for (int i = 0; i < NUM_SERVOS; ++i) {
    Vec3 axis = getServoAxis(i);
    Mat3 R = buildServoFrame(axis);     // global -> local
    Mat3 Rt = mat3Transpose(R);

    Vec3 B = bottom[i];
    Vec3 H0 = horn0[i];
    Vec3 U  = upperT[i];

    Vec3 hLocal = mat3Mul(R, vSub(H0, B));
    Vec3 uLocal = mat3Mul(R, vSub(U,  B));

    // Local origin in servo frame; name avoids Arduino's B0 macro.
    Vec3 originLocal = v3(0, 0, 0);
    float t1, t2;
    if (!solveServoAngleZAxis(originLocal, hLocal, uLocal, L, t1, t2)) {
      outSuccess[i] = false;
      outAngles[i] = 0.0f;
      continue;
    }

    float theta;
    if (!pickSolution(t1, t2, minDeg, maxDeg, 0.0f, theta)) {
      outSuccess[i] = false;
      outAngles[i] = 0.0f;
      continue;
    }

    // Store IK angle (relative to model 0°)
    outSuccess[i] = true;
    outAngles[i] = theta;
    // If you wanted horn positions, you’d rotate hLocal by theta and map back with Rt.
  }
}

// Apply a single pose (x,y,z,roll,pitch,yaw) via IK and drive servos
// x,y,z in mm; roll,pitch,yaw in degrees.
bool applyPoseIK(float x, float y, float z,
                 float rollDeg, float pitchDeg, float yawDeg,
                 float minDeg, float maxDeg) {
  Vec3 upperT[NUM_SERVOS];
  float angles[NUM_SERVOS];
  bool  ok[NUM_SERVOS];

  transformUpperAnchors(upperAnchor, upperT,
                        z, rollDeg, pitchDeg, yawDeg,
                        x, y);

  computeAllServoAngles(bottomAnchor, horn0Anchor, upperT,
                        LINK_LENGTH, minDeg, maxDeg,
                        angles, ok);

  bool allOk = true;
  for (int i = 0; i < NUM_SERVOS; ++i) {
    if (!ok[i]) { allOk = false; break; }
  }

  if (!allOk) {
    Serial.println("# IK failed for this pose; servos not moved.");
    return false;
  }

  // Drive servos: model 0° corresponds to 90° command.
  Serial.print("# IK angles (deg):");
  for (int i = 0; i < NUM_SERVOS; ++i) {
    int cmd = clampDeg((int)roundf(90.0f + angles[i]));
    servos[i].write(cmd);
    Serial.print(' ');
    Serial.print(angles[i], 2);
  }
  Serial.println();
  return true;
}

// ---------------- Circular motion, normal toward (0,0,50) ----------------

void demoCircularMotion() {
  const float radius_mm = 50.0f;
  const float dz        = 0.0f;
  const float baseRoll  = 0.0f;
  const float basePitch = 0.0f;
  const float baseYaw   = 0.0f;
  const int   nSteps    = 180;
  const float minDeg    = -90.0f;
  const float maxDeg    =  90.0f;
  const float tiltDeg   = 10.0f; // how much to lean outward

  Vec3 upperT[NUM_SERVOS];
  float angles[NUM_SERVOS];
  bool  ok[NUM_SERVOS];

  for (int step = 0; step < nSteps; ++step) {
    float tDeg = (360.0f / nSteps) * step;
    float rad = tDeg * DEG2RAD;
    float dx = radius_mm * cosf(rad);
    float dy = radius_mm * sinf(rad);

    // Lean plate so that its normal tilts radially outward in XY
    // (-normal points approximately toward (0,0,50)).
    float rollStep  = -tiltDeg * (dy / radius_mm);
    float pitchStep =  tiltDeg * (dx / radius_mm);
    float rollCmd   = baseRoll  + rollStep;
    float pitchCmd  = basePitch + pitchStep;

    transformUpperAnchors(upperAnchor, upperT,
                          dz, rollCmd, pitchCmd, baseYaw,
                          dx, dy);

    computeAllServoAngles(bottomAnchor, horn0Anchor, upperT,
                          LINK_LENGTH, minDeg, maxDeg,
                          angles, ok);

    // Drive servos: assume model 0° corresponds to 90° command.
    for (int i = 0; i < NUM_SERVOS; ++i) {
      if (!ok[i]) continue; // skip on failure to avoid wild motion
      int cmd = clampDeg((int)roundf(90.0f + angles[i]));
      servos[i].write(cmd);
    }

    delay(40); // ~25 Hz
  }
}

// ---------------- Serial command handling ----------------

void handleSerial() {
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  // Text commands: DEMO / DEMO CENTER / CENTER
  if (line.equalsIgnoreCase("DEMO") || line.equalsIgnoreCase("DEMO CENTER")) {
    Serial.println("# Starting circular IK demo...");
    demoCircularMotion();
    Serial.println("# Demo done.");
    return;
  }

  if (line.equalsIgnoreCase("CENTER")) {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write(90);
    Serial.println("# centered to 90");
    return;
  }

  // Numeric pose: "x y z roll pitch yaw"
  float x, y, z, rollDeg, pitchDeg, yawDeg;
  if (sscanf(line.c_str(), "%f %f %f %f %f %f",
             &x, &y, &z, &rollDeg, &pitchDeg, &yawDeg) == 6) {
    Serial.print("# Pose request: x="); Serial.print(x);
    Serial.print(" y="); Serial.print(y);
    Serial.print(" z="); Serial.print(z);
    Serial.print(" roll="); Serial.print(rollDeg);
    Serial.print(" pitch="); Serial.print(pitchDeg);
    Serial.print(" yaw="); Serial.println(yawDeg);
    applyPoseIK(x, y, z, rollDeg, pitchDeg, yawDeg,
                -90.0f, 90.0f);
    return;
  }

  Serial.println("# Unknown. Use: 'CENTER' | 'DEMO' | 'x y z roll pitch yaw'");
}

// ---------------- Arduino setup/loop ----------------

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(servo_pins[i], PWM_PULSE_MIN_US, PWM_PULSE_MAX_US);
    servos[i].write(90);
  }

  generateCoordinates();

  Serial.println("Stewart Platform IK demo ready.");
  Serial.println("Commands: CENTER | SET i angle | ALL angle | SWEEP i | DEMO");
}

void loop() {
  if (Serial.available()) handleSerial();
}