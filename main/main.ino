#include <ESP32Servo.h>
#include <math.h>
#include <stdio.h>
#include "geometry.h"

// ---------------- WiFi debug viewer (optional) ----------------
// This lets you inspect the last received serial command and parse status in a browser,
// which is useful when Simulink owns the COM port.
#ifndef STEWART_WIFI_DEBUG
#define STEWART_WIFI_DEBUG 1
#endif

// ---------------- Serial logging (optional) ----------------
// When Simulink is not reading from the ESP32, heavy Serial.print can block and cause jerky motion.
// Set to 0 for smooth high-rate control.
#ifndef STEWART_SERIAL_LOG
#define STEWART_SERIAL_LOG 1
#endif

#if STEWART_SERIAL_LOG
  #define SLOG_PRINT(x)    Serial.print(x)
  #define SLOG_PRINT2(x,y) Serial.print((x),(y))
  #define SLOG_PRINTLN(x)  Serial.println(x)
  #define SLOG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define SLOG_PRINT(x)    do {} while (0)
  #define SLOG_PRINT2(x,y) do {} while (0)
  #define SLOG_PRINTLN(x)  do {} while (0)
  #define SLOG_PRINTF(...) do {} while (0)
#endif

#if STEWART_WIFI_DEBUG
  #include <WiFi.h>
  #include <WebServer.h>
  #include <esp_system.h>

  static const char *WIFI_SSID = "iptime";
  static const char *WIFI_PASS = "Za!cW~QWdh5FC~f";
  static WebServer dbgServer(80);

  static String dbgLastLine = "";
  static bool dbgLastParseOk = false;
  static uint32_t dbgLinesTotal = 0;
  static uint32_t dbgParseOk = 0;
  static uint32_t dbgParseFail = 0;
  static uint32_t dbgLastRxMs = 0;
  static float dbgLastPose[6] = {0,0,0,0,0,0};

  static void jsonAppendEscaped(String &out, const String &in) {
    for (size_t i = 0; i < (size_t)in.length(); ++i) {
      char c = in[i];
      switch (c) {
        case '\"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
          if ((uint8_t)c < 0x20) {
            char buf[7];
            snprintf(buf, sizeof(buf), "\\u%04X", (unsigned)((uint8_t)c));
            out += buf;
          } else out += c;
      }
    }
  }

  static void jsonAddKV(String &json, const char *key, const String &value) {
    json += "\""; json += key; json += "\":\"";
    jsonAppendEscaped(json, value);
    json += "\"";
  }

  static const char* resetReasonText(esp_reset_reason_t r) {
    switch (r) {
      case ESP_RST_POWERON: return "POWERON";
      case ESP_RST_EXT:     return "EXT";
      case ESP_RST_SW:      return "SW";
      case ESP_RST_PANIC:   return "PANIC";
      case ESP_RST_INT_WDT: return "INT_WDT";
      case ESP_RST_TASK_WDT:return "TASK_WDT";
      case ESP_RST_WDT:     return "WDT";
      case ESP_RST_DEEPSLEEP:return "DEEPSLEEP";
      case ESP_RST_BROWNOUT:return "BROWNOUT";
      case ESP_RST_SDIO:    return "SDIO";
      default:              return "UNKNOWN";
    }
  }

  static void dbgHandleRoot() {
    String html;
    html.reserve(2600);
    html += "<!doctype html><html><head><meta charset='utf-8'/>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'/>";
    html += "<title>Stewart Serial Debug</title>";
    html += "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:16px;}";
    html += ".card{max-width:900px;border:1px solid #ddd;border-radius:12px;padding:16px;}";
    html += "code,pre{background:#f6f8fa;border-radius:8px;padding:10px;display:block;overflow:auto;}</style>";
    html += "</head><body><div class='card'>";
    html += "<h2>Stewart Serial Debug</h2>";
    html += "<div><b>IP:</b> <span id='ip'>...</span></div>";
    html += "<div><b>Uptime (ms):</b> <span id='up'>...</span></div>";
    html += "<div><b>Reset reason:</b> <span id='rr'>...</span></div>";
    html += "<hr/>";
    html += "<div><b>Lines total:</b> <span id='lt'>0</span></div>";
    html += "<div><b>Parse OK / Fail:</b> <span id='ok'>0</span> / <span id='fail'>0</span></div>";
    html += "<div><b>Last parse:</b> <span id='lp'>-</span></div>";
    html += "<div><b>Last RX age (ms):</b> <span id='age'>-</span></div>";
    html += "<p><b>Last line:</b></p><pre id='line'>-</pre>";
    html += "<p><b>Last pose:</b></p><pre id='pose'>-</pre>";
    html += "<button onclick='fetch(\"/clear\",{method:\"POST\"})'>Clear counters</button>";
    html += "<p style='color:#666'>Polling <code>/data</code> every 250ms.</p>";
    html += "</div><script>";
    html += "async function tick(){const r=await fetch('/data',{cache:'no-store'});";
    html += "const j=await r.json();";
    html += "ip.textContent=j.ip; up.textContent=j.uptime_ms; rr.textContent=j.reset_reason;";
    html += "lt.textContent=j.lines_total; ok.textContent=j.parse_ok; fail.textContent=j.parse_fail;";
    html += "lp.textContent=j.last_parse; age.textContent=j.last_rx_age_ms;";
    html += "line.textContent=j.last_line; pose.textContent=j.last_pose; }";
    html += "setInterval(()=>tick().catch(()=>{}),250);tick().catch(()=>{});";
    html += "</script></body></html>";
    dbgServer.send(200, "text/html; charset=utf-8", html);
  }

  static void dbgHandleData() {
    String json;
    json.reserve(900 + dbgLastLine.length());
    json += "{";
    jsonAddKV(json, "ip", (WiFi.status() == WL_CONNECTED) ? WiFi.localIP().toString() : "0.0.0.0"); json += ",";
    json += "\"uptime_ms\":" + String(millis()) + ",";
    jsonAddKV(json, "reset_reason", resetReasonText(esp_reset_reason())); json += ",";
    json += "\"lines_total\":" + String(dbgLinesTotal) + ",";
    json += "\"parse_ok\":" + String(dbgParseOk) + ",";
    json += "\"parse_fail\":" + String(dbgParseFail) + ",";
    jsonAddKV(json, "last_parse", dbgLastParseOk ? "OK" : "FAIL"); json += ",";
    uint32_t age = (dbgLastRxMs == 0) ? 0 : (millis() - dbgLastRxMs);
    json += "\"last_rx_age_ms\":" + String(age) + ",";
    jsonAddKV(json, "last_line", dbgLastLine); json += ",";
    String pose;
    pose.reserve(120);
    pose += "x=" + String(dbgLastPose[0], 3);
    pose += " y=" + String(dbgLastPose[1], 3);
    pose += " z=" + String(dbgLastPose[2], 3);
    pose += " roll=" + String(dbgLastPose[3], 3);
    pose += " pitch=" + String(dbgLastPose[4], 3);
    pose += " yaw=" + String(dbgLastPose[5], 3);
    jsonAddKV(json, "last_pose", pose);
    json += "}";
    dbgServer.send(200, "application/json; charset=utf-8", json);
  }

  static void dbgHandleClear() {
    dbgLinesTotal = 0;
    dbgParseOk = 0;
    dbgParseFail = 0;
    dbgLastLine = "";
    dbgLastParseOk = false;
    dbgLastRxMs = 0;
    for (int i = 0; i < 6; ++i) dbgLastPose[i] = 0.0f;
    dbgServer.send(200, "text/plain; charset=utf-8", "ok");
  }

  static void dbgWiFiTick() {
    static uint32_t lastKick = 0;
    static wl_status_t lastStatus = WL_IDLE_STATUS;

    // Print IP once when we successfully connect.
    wl_status_t st = WiFi.status();
    if (st == WL_CONNECTED) {
      if (lastStatus != WL_CONNECTED) {
        IPAddress ip = WiFi.localIP();
        Serial.print("# WiFi connected. IP: ");
        Serial.println(ip);
        Serial.print("# Open: http://");
        Serial.print(ip);
        Serial.println("/");
      }
      lastStatus = st;
      return;
    }
    lastStatus = st;

    if (millis() - lastKick < 3000) return;
    lastKick = millis();
    WiFi.disconnect(false, false);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
#endif

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

// Transform upper anchors by pose (same semantics as Python).
// Convention (right-hand rule, Z up):
//   - pitch: rotation about +X
//   - roll : rotation about +Y
void transformUpperAnchors(const Vec3 src[], Vec3 dst[],
                           float dz, float rollDeg, float pitchDeg, float yawDeg,
                           float dx, float dy) {
  // center of plate
  Vec3 center = v3(0, 0, 0);
  for (int i = 0; i < NUM_SERVOS; ++i) center = vAdd(center, src[i]);
  center = vScale(center, 1.0f / NUM_SERVOS);

  // convert yaw to -180 to 180 range
  if(yawDeg > 180.0f) {
    yawDeg = yawDeg - 360.0f;
  }

  Vec3 tmp[NUM_SERVOS];
  for (int i = 0; i < NUM_SERVOS; ++i) {
    tmp[i] = vSub(src[i], center);
    // Apply pitch (X), then roll (Y), then yaw (Z)
    tmp[i] = rotateX(tmp[i], -pitchDeg);
    tmp[i] = rotateY(tmp[i], -rollDeg);
    tmp[i] = rotateZ(tmp[i], yawDeg);
    tmp[i] = vAdd(tmp[i], center);
    tmp[i] = translate(tmp[i], dx, dy, dz);
  }
  for (int i = 0; i < NUM_SERVOS; ++i) dst[i] = tmp[i];
}

void computeAllServoAngles(const Vec3 bottom[],
                           const Vec3 horn0[],
                           const Vec3 upperT[],
                           float L, float minDeg, float maxDeg,
                           float outAngles[],
                           bool outSuccess[]) {
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
    SLOG_PRINTLN("# IK failed for this pose; servos not moved.");
    return false;
  }

  // Drive servos: model 0° corresponds to 90° command.
  SLOG_PRINT("# IK angles (deg):");
  for (int i = 0; i < NUM_SERVOS; ++i) {
    int cmd = clampDeg((int)roundf(90.0f + angles[i]));
    servos[i].write(cmd);
    SLOG_PRINT(' ');
    SLOG_PRINT2(angles[i], 2);
  }
  SLOG_PRINTLN("");
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
    // With our convention and small angles: n_xy ≈ (roll, -pitch).
    float rollStep  = tiltDeg * (dx / radius_mm);
    float pitchStep = -tiltDeg * (dy / radius_mm);
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
  // Deprecated: kept for compatibility with older code paths.
  // We now use a non-blocking line parser in loop().
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

#if STEWART_WIFI_DEBUG
  dbgLastRxMs = millis();
  dbgLinesTotal++;
  dbgLastLine = line;
#endif

  // Text commands: DEMO / DEMO CENTER / CENTER
  if (line.equalsIgnoreCase("DEMO") || line.equalsIgnoreCase("DEMO CENTER")) {
    SLOG_PRINTLN("# Starting circular IK demo...");
    demoCircularMotion();
    SLOG_PRINTLN("# Demo done.");
    return;
  }

  if (line.equalsIgnoreCase("CENTER")) {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write(90);
    SLOG_PRINTLN("# centered to 90");
    return;
  }

  // Numeric pose: "x y z roll pitch yaw"
  float x, y, z, rollDeg, pitchDeg, yawDeg;
  if (sscanf(line.c_str(), "%f %f %f %f %f %f",
             &x, &y, &z, &rollDeg, &pitchDeg, &yawDeg) == 6) {
#if STEWART_WIFI_DEBUG
    dbgLastParseOk = true;
    dbgParseOk++;
    dbgLastPose[0] = x;
    dbgLastPose[1] = y;
    dbgLastPose[2] = z;
    dbgLastPose[3] = rollDeg;
    dbgLastPose[4] = pitchDeg;
    dbgLastPose[5] = yawDeg;
#endif
    SLOG_PRINT("# Pose request: x="); SLOG_PRINT(x);
    SLOG_PRINT(" y="); SLOG_PRINT(y);
    SLOG_PRINT(" z="); SLOG_PRINT(z);
    SLOG_PRINT(" roll="); SLOG_PRINT(rollDeg);
    SLOG_PRINT(" pitch="); SLOG_PRINT(pitchDeg);
    SLOG_PRINT(" yaw="); SLOG_PRINTLN(yawDeg);
    applyPoseIK(x, y, z, rollDeg, pitchDeg, yawDeg,
                -90.0f, 90.0f);
    return;
  }

#if STEWART_WIFI_DEBUG
  dbgLastParseOk = false;
  dbgParseFail++;
#endif
  SLOG_PRINTLN("# Unknown. Use: 'CENTER' | 'DEMO' | 'x y z roll pitch yaw'");
}

// ---------------- Non-blocking serial line parser ----------------
static constexpr size_t RX_LINE_MAX = 160;
static char rxLine[RX_LINE_MAX];
static size_t rxLen = 0;

static void processLine(const char *cstrLine) {
  String line(cstrLine);
  line.trim();
  if (line.length() == 0) return;

#if STEWART_WIFI_DEBUG
  dbgLastRxMs = millis();
  dbgLinesTotal++;
  dbgLastLine = line;
#endif

  // Text commands: DEMO / DEMO CENTER / CENTER
  if (line.equalsIgnoreCase("DEMO") || line.equalsIgnoreCase("DEMO CENTER")) {
    SLOG_PRINTLN("# Starting circular IK demo...");
    demoCircularMotion();
    SLOG_PRINTLN("# Demo done.");
    return;
  }

  if (line.equalsIgnoreCase("CENTER")) {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write(90);
    SLOG_PRINTLN("# centered to 90");
    return;
  }

  float x, y, z, rollDeg, pitchDeg, yawDeg;
  if (sscanf(line.c_str(), "%f %f %f %f %f %f",
             &x, &y, &z, &rollDeg, &pitchDeg, &yawDeg) == 6) {
#if STEWART_WIFI_DEBUG
    dbgLastParseOk = true;
    dbgParseOk++;
    dbgLastPose[0] = x;
    dbgLastPose[1] = y;
    dbgLastPose[2] = z;
    dbgLastPose[3] = rollDeg;
    dbgLastPose[4] = pitchDeg;
    dbgLastPose[5] = yawDeg;
#endif

    applyPoseIK(x, y, z, rollDeg, pitchDeg, yawDeg, -90.0f, 90.0f);
    return;
  }

#if STEWART_WIFI_DEBUG
  dbgLastParseOk = false;
  dbgParseFail++;
#endif
  SLOG_PRINTLN("# Unknown/parse fail");
}

static void serialPollLines() {
  while (Serial.available() > 0) {
    int v = Serial.read();
    if (v < 0) break;
    char c = (char)v;

    if (c == '\r') continue;

    if (c == '\n') {
      rxLine[rxLen] = '\0';
      processLine(rxLine);
      rxLen = 0;
      continue;
    }

    if (rxLen + 1 < RX_LINE_MAX) {
      rxLine[rxLen++] = c;
    } else {
      // Overflow: drop the line to resync on next newline.
      rxLen = 0;
    }
  }
}

// ---------------- Arduino setup/loop ----------------

void setup() {
  Serial.begin(115200);
  // We use a non-blocking parser; keep default timeout (doesn't matter unless handleSerial() is used).

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

  SLOG_PRINTLN("Stewart Platform IK demo ready.");
  SLOG_PRINTLN("Commands: CENTER | SET i angle | ALL angle | SWEEP i | DEMO");

#if STEWART_WIFI_DEBUG
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  dbgServer.on("/", HTTP_GET, dbgHandleRoot);
  dbgServer.on("/data", HTTP_GET, dbgHandleData);
  dbgServer.on("/clear", HTTP_POST, dbgHandleClear);
  dbgServer.begin();
  SLOG_PRINTLN("# WiFi debug server started (port 80). Check router for IP.");
#endif
}

void loop() {
  serialPollLines();

#if STEWART_WIFI_DEBUG
  dbgWiFiTick();
  dbgServer.handleClient();
#endif
}