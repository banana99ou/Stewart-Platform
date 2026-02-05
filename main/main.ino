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
  #include <WiFiUdp.h>
  #include <esp_system.h>

  static const char *WIFI_SSID = "iptime";
  static const char *WIFI_PASS = "Za!cW~QWdh5FC~f";
  static WebServer dbgServer(80);
  static WiFiUDP dbgUdp;

  static String dbgLastLine = "";
  static bool dbgLastParseOk = false;
  static uint32_t dbgLinesTotal = 0;
  static uint32_t dbgParseOk = 0;
  static uint32_t dbgParseFail = 0;
  static uint32_t dbgLastRxMs = 0;
  static float dbgLastPose[6] = {0,0,0,0,0,0};

  // --------------- Status/ACK telemetry (Serial + UDP) ---------------
  // Goal: emit a small machine-parsable message per pose command so you can
  // compute end-to-end latency and saturation/IK outcomes.
  //
  // Design notes:
  // - UDP is lossy; that's fine for metrics, but don't rely on it for control.
  // - Keep messages short to avoid impacting motion (Serial can block if host isn't reading).
  #ifndef STEWART_STATUS_SERIAL
  #define STEWART_STATUS_SERIAL 1
  #endif

  #ifndef STEWART_STATUS_UDP
  #define STEWART_STATUS_UDP 1
  #endif

  // UDP destination:
  // - If STATUS_UDP_HOST is a valid IP string, use that.
  // - Else (empty/invalid), send to WiFi.broadcastIP().
  static const char *STATUS_UDP_HOST = ""; // e.g. "192.168.0.10" (PC) or "" for broadcast
  static const uint16_t STATUS_UDP_PORT = 14551;

  enum StatusFlags : uint32_t {
    ST_CMD_POSE        = 1u << 0,
    ST_PARSE_OK        = 1u << 1,
    ST_PARSE_FAIL      = 1u << 2,
    ST_SAT_APPLIED     = 1u << 3,
    ST_IK_OK           = 1u << 4,
    ST_IK_FAIL         = 1u << 5,
    ST_RX_OVERFLOW     = 1u << 6,
    ST_WIFI_CONNECTED  = 1u << 7,
    ST_Z_CLAMPED       = 1u << 8,
  };

  static uint32_t gPoseSeq = 0; // increments on each parsed pose command

  static IPAddress statusUdpDest() {
    if (STATUS_UDP_HOST && STATUS_UDP_HOST[0] != '\0') {
      IPAddress ip;
      if (ip.fromString(STATUS_UDP_HOST)) return ip;
    }
    return (WiFi.status() == WL_CONNECTED) ? WiFi.broadcastIP() : IPAddress(255,255,255,255);
  }

  static void emitStatusLine(const char *cstr) {
  #if STEWART_STATUS_SERIAL
    Serial.println(cstr);
  #endif

  #if STEWART_STATUS_UDP
    if (WiFi.status() == WL_CONNECTED) {
      IPAddress dst = statusUdpDest();
      if (dbgUdp.beginPacket(dst, STATUS_UDP_PORT)) {
        dbgUdp.write((const uint8_t*)cstr, strlen(cstr));
        dbgUdp.endPacket();
      }
    }
  #endif
  }

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

  static const char* statusWordFrom(uint32_t flags) {
    if (flags & ST_RX_OVERFLOW) return "RX_OVERFLOW";
    if (flags & ST_PARSE_FAIL)  return "PARSE_FAIL";
    if (flags & ST_IK_FAIL)     return "IK_FAIL";
    if (flags & ST_SAT_APPLIED) return "SAT";
    return "OK";
  }

  static void flagsToTokens(uint32_t flags, char *out, size_t outCap) {
    // Produces a compact token list like: "POSE|WIFI|SAT|IK_OK"
    // Always null-terminated.
    if (outCap == 0) return;
    out[0] = '\0';

    auto add = [&](const char *tok) {
      if (!tok || !tok[0]) return;
      size_t len = strlen(out);
      size_t tlen = strlen(tok);
      if (len == 0) {
        if (tlen + 1 > outCap) return;
        memcpy(out, tok, tlen + 1);
      } else {
        if (len + 1 + tlen + 1 > outCap) return;
        out[len] = '|';
        memcpy(out + len + 1, tok, tlen + 1);
      }
    };

    if (flags & ST_CMD_POSE)       add("POSE");
    if (flags & ST_PARSE_OK)       add("PARSE_OK");
    if (flags & ST_PARSE_FAIL)     add("PARSE_FAIL");
    if (flags & ST_RX_OVERFLOW)    add("RX_OVERFLOW");
    if (flags & ST_WIFI_CONNECTED) add("WIFI");
    if (flags & ST_SAT_APPLIED)    add("SAT");
    if (flags & ST_IK_OK)          add("IK_OK");
    if (flags & ST_IK_FAIL)        add("IK_FAIL");
    if (flags & ST_Z_CLAMPED)      add("Z_CLAMP");
  }
#endif

// ---------------- Servo HW ----------------

constexpr uint8_t NUM_SERVOS = 6;
constexpr int PWM_PULSE_MIN_US = 500;
constexpr int PWM_PULSE_MAX_US = 2450;

const uint8_t servo_pins[NUM_SERVOS] = {4, 13, 16, 17, 18, 19};
Servo servos[NUM_SERVOS];

// ---------------- Geometry globals (forward declarations) ----------------
// These are defined later, but referenced by helper functions above their definitions.
// Keeping declarations here also avoids Arduino's auto-prototype step tripping over
// undeclared identifiers in function signatures/bodies.
extern const float LINK_LENGTH; // mm (ball link center to center)
extern Vec3 bottomAnchor[NUM_SERVOS];
extern Vec3 horn0Anchor[NUM_SERVOS];
extern Vec3 upperAnchor[NUM_SERVOS];

// ---------------- Servo calibration ----------------
// The IK solver outputs angles in degrees around the modeled "0°" horn pose.
// Real hardware needs per-servo calibration:
// - center offset (what write(deg) corresponds to modeled 0°)
// - sign (some servos may be mirrored in installation)
//
// Defaults preserve current behavior: model 0° -> write(90), positive angle adds degrees.
static float SERVO_CENTER_DEG[NUM_SERVOS] = {92, 90, 92, 88, 88, 89};
static float SERVO_SIGN[NUM_SERVOS]       = { 1,  1,  1,  1,  1,  1};

static inline int clampDeg(int d) {
  if (d < 0) return 0;
  if (d > 180) return 180;
  return d;
}

// ---------------- Math helpers ----------------

const float DEG2RAD = 3.14159265358979323846f / 180.0f;
const float RAD2DEG = 180.0f / 3.14159265358979323846f;

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

static inline Mat3 mat3MulMat(const Mat3 &A, const Mat3 &B) {
  Mat3 C;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float s = 0.0f;
      for (int k = 0; k < 3; ++k) s += A.m[i][k] * B.m[k][j];
      C.m[i][j] = s;
    }
  }
  return C;
}

static inline Mat3 mat3Identity() {
  Mat3 I;
  I.m[0][0] = 1; I.m[0][1] = 0; I.m[0][2] = 0;
  I.m[1][0] = 0; I.m[1][1] = 1; I.m[1][2] = 0;
  I.m[2][0] = 0; I.m[2][1] = 0; I.m[2][2] = 1;
  return I;
}

static inline Mat3 mat3Transpose(const Mat3 &R) {
  Mat3 Rt;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      Rt.m[i][j] = R.m[j][i];
  return Rt;
}

// ---------------- Rotation-magnitude saturator ----------------
// Saturate overall rotation magnitude by scaling in axis-angle (rotation-vector) space,
// while preserving the rotation axis. This avoids the "clamp each Euler axis" artifact.
//
// IMPORTANT: This uses the same Euler convention as transformUpperAnchors():
//   p' = Rz(yaw) * Ry(-roll) * Rx(-pitch) * p
//
// Adaptive saturation: max rotation magnitude depends strongly on Z.
// We embed a z->maxMagDeg lookup from a Python reachability sweep and interpolate.
//
// Notes:
// - The values are "inscribed sphere radius" in rotation-vector space (conservative).
// - We clamp z into the table range before lookup.
#ifndef STEWART_ROT_SAT_ENABLE
#define STEWART_ROT_SAT_ENABLE 1
#endif

// Reachability sweep results (worst-case max rotation magnitude vs z) for z in [-22, +25] mm, step 1mm.
// Generated from `rot_boundary_z_sweep.npz` (n_dirs=300, tol=0.25deg).
static const float ROT_SAT_WORST_DEG[48] = {
  0.000000f, 0.468750f, 0.937500f, 1.406250f, 1.875000f, 2.343750f, 2.812500f, 3.281250f,
  3.750000f, 4.218750f, 4.687500f, 5.156250f, 5.390625f, 5.859375f, 6.328125f, 6.796875f,
  7.265625f, 7.968750f, 8.437500f, 8.906250f, 9.375000f, 9.843750f, 10.312500f, 10.781250f,
  11.250000f, 11.718750f, 12.187500f, 12.656250f, 13.125000f, 13.828125f, 14.296875f, 14.765625f,
  15.234375f, 15.703125f, 16.171875f, 16.875000f, 17.343750f, 17.812500f, 18.281250f, 18.750000f,
  19.453125f, 19.921875f, 20.390625f, 20.859375f, 21.562500f, 22.031250f, 22.500000f, 23.203125f
};
static const float ROT_SAT_Z_MIN_MM = -22.0f;
static const float ROT_SAT_Z_STEP_MM = 1.0f;
static const float ROT_SAT_Z_MAX_MM = ROT_SAT_Z_MIN_MM + ROT_SAT_Z_STEP_MM * (float)(sizeof(ROT_SAT_WORST_DEG) / sizeof(ROT_SAT_WORST_DEG[0]) - 1);

// Optional safety margin (deg) subtracted from the interpolated limit.
static const float ROT_SAT_MARGIN_DEG = 0.5f;

static inline float rotSatLimitDegForZ(float zMm, bool &zClamped) {
  zClamped = false;
  float z0 = zMm;
  float zc = zMm;
  if (zc < ROT_SAT_Z_MIN_MM) { zc = ROT_SAT_Z_MIN_MM; zClamped = true; }
  if (zc > ROT_SAT_Z_MAX_MM) { zc = ROT_SAT_Z_MAX_MM; zClamped = true; }

  // Index in table
  float t = (zc - ROT_SAT_Z_MIN_MM) / ROT_SAT_Z_STEP_MM;
  int i0 = (int)floorf(t);
  float frac = t - (float)i0;
  int n = (int)(sizeof(ROT_SAT_WORST_DEG) / sizeof(ROT_SAT_WORST_DEG[0]));
  if (i0 < 0) { i0 = 0; frac = 0.0f; }
  if (i0 >= n - 1) { i0 = n - 1; frac = 0.0f; }
  int i1 = (i0 < n - 1) ? (i0 + 1) : i0;

  float a = ROT_SAT_WORST_DEG[i0];
  float b = ROT_SAT_WORST_DEG[i1];
  float lim = a + frac * (b - a);
  lim = fmaxf(0.0f, lim - ROT_SAT_MARGIN_DEG);

  (void)z0; // for debugging if needed
  return lim;
}

static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static inline float wrapDeg180(float d) {
  // Wrap to (-180, 180]
  while (d > 180.0f) d -= 360.0f;
  while (d <= -180.0f) d += 360.0f;
  return d;
}

static inline Mat3 rotXMat(float deg) {
  float r = deg * DEG2RAD;
  float c = cosf(r), s = sinf(r);
  Mat3 R;
  R.m[0][0] = 1; R.m[0][1] = 0;  R.m[0][2] = 0;
  R.m[1][0] = 0; R.m[1][1] = c;  R.m[1][2] = -s;
  R.m[2][0] = 0; R.m[2][1] = s;  R.m[2][2] = c;
  return R;
}

static inline Mat3 rotYMat(float deg) {
  float r = deg * DEG2RAD;
  float c = cosf(r), s = sinf(r);
  Mat3 R;
  R.m[0][0] = c;  R.m[0][1] = 0; R.m[0][2] = s;
  R.m[1][0] = 0;  R.m[1][1] = 1; R.m[1][2] = 0;
  R.m[2][0] = -s; R.m[2][1] = 0; R.m[2][2] = c;
  return R;
}

static inline Mat3 rotZMat(float deg) {
  float r = deg * DEG2RAD;
  float c = cosf(r), s = sinf(r);
  Mat3 R;
  R.m[0][0] = c;  R.m[0][1] = -s; R.m[0][2] = 0;
  R.m[1][0] = s;  R.m[1][1] = c;  R.m[1][2] = 0;
  R.m[2][0] = 0;  R.m[2][1] = 0;  R.m[2][2] = 1;
  return R;
}

static inline Mat3 eulerToMat_Arduino(float rollDeg, float pitchDeg, float yawDeg) {
  // Matches transformUpperAnchors(): rotateX(-pitch), then rotateY(-roll), then rotateZ(yaw).
  Mat3 Rx = rotXMat(-pitchDeg);
  Mat3 Ry = rotYMat(-rollDeg);
  Mat3 Rz = rotZMat(yawDeg);
  return mat3MulMat(Rz, mat3MulMat(Ry, Rx));
}

static inline void matToEuler_Arduino(const Mat3 &R, float &rollDeg, float &pitchDeg, float &yawDeg) {
  // Invert R = Rz(yaw) * Ry(-roll) * Rx(-pitch)
  float alpha = atan2f(R.m[2][1], R.m[2][2]);               // alpha = -pitch (rad)
  float beta  = asinf(clampf(-R.m[2][0], -1.0f, 1.0f));     // beta  = -roll  (rad)
  float gamma = atan2f(R.m[1][0], R.m[0][0]);               // gamma = yaw    (rad)
  pitchDeg = (-alpha) * RAD2DEG;
  rollDeg  = (-beta)  * RAD2DEG;
  yawDeg   = (gamma)  * RAD2DEG;
  yawDeg   = wrapDeg180(yawDeg);
}

static inline Mat3 axisAngleToMat(const Vec3 &axisUnit, float angleRad) {
  // Rodrigues: R = I + sin(th)K + (1-cos(th))K^2
  float th = angleRad;
  if (fabsf(th) < 1e-12f) return mat3Identity();
  Vec3 k = axisUnit;
  float kx = k.x, ky = k.y, kz = k.z;

  Mat3 K;
  K.m[0][0] = 0;   K.m[0][1] = -kz; K.m[0][2] = ky;
  K.m[1][0] = kz;  K.m[1][1] = 0;   K.m[1][2] = -kx;
  K.m[2][0] = -ky; K.m[2][1] = kx;  K.m[2][2] = 0;

  Mat3 K2 = mat3MulMat(K, K);
  float s = sinf(th);
  float c = cosf(th);

  Mat3 R = mat3Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R.m[i][j] = R.m[i][j] + s * K.m[i][j] + (1.0f - c) * K2.m[i][j];
    }
  }
  return R;
}

static inline void matToAxisAngle(const Mat3 &R, Vec3 &axisUnit, float &angleRad) {
  // angle in [0, pi]
  float tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  float c = clampf((tr - 1.0f) * 0.5f, -1.0f, 1.0f);
  float th = acosf(c);
  angleRad = th;

  if (th < 1e-8f) {
    axisUnit = v3(1, 0, 0);
    angleRad = 0.0f;
    return;
  }

  float s = sinf(th);
  if (fabsf(s) > 1e-6f) {
    float inv = 1.0f / (2.0f * s);
    axisUnit = vNormed(v3(
      (R.m[2][1] - R.m[1][2]) * inv,
      (R.m[0][2] - R.m[2][0]) * inv,
      (R.m[1][0] - R.m[0][1]) * inv
    ));
    return;
  }

  // Near pi: diagonal-based extraction
  float A00 = (R.m[0][0] + 1.0f) * 0.5f;
  float A11 = (R.m[1][1] + 1.0f) * 0.5f;
  float A22 = (R.m[2][2] + 1.0f) * 0.5f;
  float kx = sqrtf(fmaxf(A00, 0.0f));
  float ky = sqrtf(fmaxf(A11, 0.0f));
  float kz = sqrtf(fmaxf(A22, 0.0f));
  if ((R.m[2][1] - R.m[1][2]) < 0.0f) kx = -kx;
  if ((R.m[0][2] - R.m[2][0]) < 0.0f) ky = -ky;
  if ((R.m[1][0] - R.m[0][1]) < 0.0f) kz = -kz;
  axisUnit = vNormed(v3(kx, ky, kz));
}

static inline void transformUpperAnchorsMat(const Vec3 src[], Vec3 dst[],
                                           const Mat3 &R,
                                           float dz, float dx, float dy) {
  // Rotate about center (same as transformUpperAnchors), then translate.
  Vec3 center = v3(0, 0, 0);
  for (int i = 0; i < NUM_SERVOS; ++i) center = vAdd(center, src[i]);
  center = vScale(center, 1.0f / NUM_SERVOS);

  for (int i = 0; i < NUM_SERVOS; ++i) {
    Vec3 p = vSub(src[i], center);
    p = mat3Mul(R, p);
    p = vAdd(p, center);
    dst[i] = translate(p, dx, dy, dz);
  }
}

// Last-pose status fields (for telemetry emission in processLine()).
static bool gLastSatApplied = false;
static float gLastSatMagInDeg = 0.0f;
static float gLastSatMagOutDeg = 0.0f;
static uint8_t gLastIkOkMask = 0; // bit i = leg i ok
static float gLastReqRoll = 0.0f, gLastReqPitch = 0.0f, gLastReqYaw = 0.0f;
static float gLastAppliedRoll = 0.0f, gLastAppliedPitch = 0.0f, gLastAppliedYaw = 0.0f;
static float gLastSatLimitDeg = 0.0f;
static bool gLastZClamped = false;

static inline bool solvePoseIKNoMove(
  float x, float y, float z,
  const Mat3 &Rpose,
  float minDeg, float maxDeg,
  float *outAngles,
  bool *outOk
) {
  Vec3 upperT[NUM_SERVOS];
  transformUpperAnchorsMat(upperAnchor, upperT, Rpose, z, x, y);
  computeAllServoAngles(bottomAnchor, horn0Anchor, upperT, LINK_LENGTH, minDeg, maxDeg, outAngles, outOk);
  for (int i = 0; i < NUM_SERVOS; ++i) {
    if (!outOk[i]) return false;
  }
  return true;
}

static inline Mat3 axisAngleFromEuler_Arduino(float rollDeg, float pitchDeg, float yawDeg,
                                              Vec3 &axisUnit, float &angleRad) {
  Mat3 R = eulerToMat_Arduino(rollDeg, pitchDeg, yawDeg);
  matToAxisAngle(R, axisUnit, angleRad);
  return R;
}

static inline bool saturateRotationByIKDirection(
  float x, float y, float z,
  float &rollDeg, float &pitchDeg, float &yawDeg,
  float minDeg, float maxDeg,
  float &magInDeg, float &magOutDeg,
  float &guessLimitDeg, bool &zClamped,
  uint8_t &okMaskOut,
  float *outAngles
) {
  // Build requested rotation (axis-angle)
  Vec3 axis;
  float th;
  Mat3 Rreq = axisAngleFromEuler_Arduino(rollDeg, pitchDeg, yawDeg, axis, th);
  magInDeg = th * RAD2DEG;

  // z-table is used only as an *initial guess* for bracketing; it is NOT a hard cap.
  guessLimitDeg = rotSatLimitDegForZ(z, zClamped);

  // First try requested pose directly
  bool ok[NUM_SERVOS];
  if (solvePoseIKNoMove(x, y, z, Rreq, minDeg, maxDeg, outAngles, ok)) {
    okMaskOut = 0;
    for (int i = 0; i < NUM_SERVOS; ++i) if (ok[i]) okMaskOut |= (uint8_t)(1u << i);
    magOutDeg = magInDeg;
    return false; // no saturation applied
  }

  // If requested fails, binary-search along the same axis-angle direction (scale angle).
  // We need a known-good lower bound.
  float anglesTmp[NUM_SERVOS];
  bool ok0[NUM_SERVOS];
  Mat3 R0 = mat3Identity();
  if (!solvePoseIKNoMove(x, y, z, R0, minDeg, maxDeg, anglesTmp, ok0)) {
    // Even zero-rotation failed at this translation/z: no saturation can fix.
    okMaskOut = 0;
    for (int i = 0; i < NUM_SERVOS; ++i) if (ok0[i]) okMaskOut |= (uint8_t)(1u << i);
    magOutDeg = 0.0f;
    return false;
  }

  float hi = th;
  float lo = 0.0f;

  // If we have a nonzero guess, try it to get a tighter bracket.
  float guessRad = guessLimitDeg * DEG2RAD;
  if (guessRad > 1e-6f && guessRad < hi) {
    Mat3 Rg = axisAngleToMat(axis, guessRad);
    bool okg[NUM_SERVOS];
    if (solvePoseIKNoMove(x, y, z, Rg, minDeg, maxDeg, anglesTmp, okg)) {
      lo = guessRad;
    } else {
      hi = guessRad;
    }
  }

  // Fixed-iteration bisection (keeps runtime bounded)
  for (int it = 0; it < 7; ++it) { // ~0.8% resolution on [0,hi]
    float mid = 0.5f * (lo + hi);
    Mat3 Rm = axisAngleToMat(axis, mid);
    bool okm[NUM_SERVOS];
    if (solvePoseIKNoMove(x, y, z, Rm, minDeg, maxDeg, anglesTmp, okm)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  // Use lo as applied angle
  Mat3 Rs = axisAngleToMat(axis, lo);
  bool oks[NUM_SERVOS];
  bool okAll = solvePoseIKNoMove(x, y, z, Rs, minDeg, maxDeg, outAngles, oks);
  okMaskOut = 0;
  for (int i = 0; i < NUM_SERVOS; ++i) if (oks[i]) okMaskOut |= (uint8_t)(1u << i);

  float rDeg = lo * RAD2DEG;
  magOutDeg = rDeg;
  matToEuler_Arduino(Rs, rollDeg, pitchDeg, yawDeg);
  return okAll; // saturation applied (and should be OK)
}

// ---------------- Geometry constants ----------------

const float LINK_LENGTH = 127.0f; // mm (ball link center to center)

// Base geometry for servo 1 (same as Python)
// NOTE: name uses suffix to avoid Arduino's binary macros B0/B1.
const Vec3 B1_pos = {-50.0f, 100.0f, 10.0f};
const Vec3 H1     = {-97.5f, 108.0f + 3.3f / 2.0f, 10.0f};
const Vec3 U1     = { -7.5f, 108.0f + 3.3f / 2.0f, 99.60469f};

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

// Solve for the two possible servo angles (about Z axis) to reach Utarget,
// given base anchor B, horn anchor H0 (at 0 deg), and target upper anchor Utarget.
// L is the linkage length. Outputs: theta1Deg/theta2Deg in degrees.
//
// Returns true if there is a solution, false if unreachable.
bool solveServoAngleZAxis(const Vec3 &B, const Vec3 &H0,
                          const Vec3 &Utarget, float L,
                          float &theta1Deg, float &theta2Deg) {
  // "h": horn offset vector (relative to base anchor)
  Vec3 h = vSub(H0, B);
  // "s": desired end effector position relative to base anchor
  Vec3 s = vSub(Utarget, B);

  // Compute geometric terms a, b (in-plane projections)
  float a = s.x * h.x + s.y * h.y;         // dot product for alignment
  float b = -s.x * h.y + s.y * h.x;        // orthogonal (for atan2)
  // Half the squared distance difference to encode linkage constraint
  float gamma = 0.5f * (vDot(s, s) + vDot(h, h) - L * L);
  // Apply Z (height) difference, ignoring planar linkage error due to height difference
  float gamma_p = gamma - s.z * h.z;

  // r is the effective length, used for arc-cosine solution
  float r = hypotf(a, b);
  if (r < 1e-6f) return false;  // linkage collapsed, numerically unstable

  // x is the cosine of the angle between two vectors, must be within physical range
  float x = gamma_p / r;
  if (x < -1.0f || x > 1.0f) return false; // unattainable

  float phi = atan2f(b, a);    // base argument
  float d = acosf(x);          // solution offset

  // Two possible solutions (elbow up/down)
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
      Serial.println("IK failed for servo " + String(i) + " with hLocal " + String(hLocal.x) + ", " + String(hLocal.y) + ", " + String(hLocal.z) + " and uLocal " + String(uLocal.x) + ", " + String(uLocal.y) + ", " + String(uLocal.z));
      continue;
    }

    float theta;
    if (!pickSolution(t1, t2, minDeg, maxDeg, 0.0f, theta)) {
      outSuccess[i] = false;
      outAngles[i] = 0.0f;
      Serial.println("IK failed for servo " + String(i) + " with angles " + String(t1) + " and " + String(t2));
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
  float angles[NUM_SERVOS];
  bool ok[NUM_SERVOS];

  // Default telemetry values
  gLastSatApplied = false;
  gLastSatMagInDeg = 0.0f;
  gLastSatMagOutDeg = 0.0f;
  gLastSatLimitDeg = 0.0f;
  gLastZClamped = false;

#if STEWART_ROT_SAT_ENABLE
  {
    float mi = 0.0f, mo = 0.0f, lim = 0.0f;
    bool zClamped = false;
    uint8_t okMask = 0;
    bool satApplied = saturateRotationByIKDirection(
      x, y, z,
      rollDeg, pitchDeg, yawDeg,
      minDeg, maxDeg,
      mi, mo,
      lim, zClamped,
      okMask,
      angles
    );
    gLastSatApplied = satApplied;
    gLastSatMagInDeg = mi;
    gLastSatMagOutDeg = mo;
    gLastSatLimitDeg = lim;
    gLastZClamped = zClamped;
    gLastIkOkMask = okMask;
  }
#else
  {
    // No saturation: just solve once
    Mat3 R = eulerToMat_Arduino(rollDeg, pitchDeg, yawDeg);
    bool okAll = solvePoseIKNoMove(x, y, z, R, minDeg, maxDeg, angles, ok);
    uint8_t okMask = 0;
    for (int i = 0; i < NUM_SERVOS; ++i) if (ok[i]) okMask |= (uint8_t)(1u << i);
    gLastIkOkMask = okMask;
    if (!okAll) {
      SLOG_PRINTLN("# IK failed for this pose; servos not moved.");
      return false;
    }
  }
#endif

  // Save what was actually applied to IK (after saturation)
  gLastAppliedRoll = rollDeg;
  gLastAppliedPitch = pitchDeg;
  gLastAppliedYaw = yawDeg;

  // If any leg failed, don't move servos.
  if (gLastIkOkMask != 0x3F) {
    SLOG_PRINTLN("# IK failed for this pose; servos not moved.");
    return false;
  }

  // Drive servos: model 0° corresponds to 90° command.
  SLOG_PRINT("# IK angles (deg):");
  for (int i = 0; i < NUM_SERVOS; ++i) {
    int cmd = clampDeg((int)roundf(SERVO_CENTER_DEG[i] + SERVO_SIGN[i] * angles[i]));
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
      int cmd = clampDeg((int)roundf(SERVO_CENTER_DEG[i] + SERVO_SIGN[i] * angles[i]));
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
    for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write((int)roundf(SERVO_CENTER_DEG[i]));
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

static void processLine(const char *cstrLine, uint32_t rxUs) {
  String line(cstrLine);
  line.trim();
  if (line.length() == 0) return;

#if 1
  // Uppercase copy for command matching (keeps original `line` for sscanf parsing).
  String u = line;
  u.toUpperCase();
#endif

#if STEWART_WIFI_DEBUG
  dbgLastRxMs = millis();
  dbgLinesTotal++;
  dbgLastLine = line;
#endif

  // Text commands: DEMO / DEMO CENTER / CENTER
  if (u == "DEMO" || u == "DEMO CENTER") {
    SLOG_PRINTLN("# Starting circular IK demo...");
    demoCircularMotion();
    SLOG_PRINTLN("# Demo done.");
    return;
  }

  if (u == "CENTER") {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write(clampDeg((int)roundf(SERVO_CENTER_DEG[i])));
    SLOG_PRINTLN("# centered to SERVO_CENTER_DEG[]");
    return;
  }

  // Calibration helpers:
  //   SET i deg     -> set servo i (1..6) to absolute write(deg) position
  //   ALL deg       -> set all servos to absolute write(deg)
  //   MODEL i a_deg -> set servo i to SERVO_CENTER_DEG[i] + SERVO_SIGN[i] * a_deg (model angle)
  //   SWEEP i [start end step delay_ms] -> sweep servo i through a range (absolute degrees)
  if (u.startsWith("SET ")) {
    int idx1 = 0;
    float deg = 0.0f;
    if (sscanf(line.c_str(), "SET %d %f", &idx1, &deg) == 2 && idx1 >= 1 && idx1 <= (int)NUM_SERVOS) {
      int cmd = clampDeg((int)roundf(deg));
      servos[idx1 - 1].write(cmd);
      SLOG_PRINTF("# SET servo %d -> %d deg\n", idx1, cmd);
      return;
    }
    SLOG_PRINTLN("# Usage: SET i deg   (i=1..6, deg=0..180)");
    return;
  }

  if (u.startsWith("ALL ")) {
    float deg = 0.0f;
    if (sscanf(line.c_str(), "ALL %f", &deg) == 1) {
      int cmd = clampDeg((int)roundf(deg));
      for (uint8_t i = 0; i < NUM_SERVOS; i++) servos[i].write(cmd);
      SLOG_PRINTF("# ALL -> %d deg\n", cmd);
      return;
    }
    SLOG_PRINTLN("# Usage: ALL deg   (deg=0..180)");
    return;
  }

  if (u.startsWith("MODEL ")) {
    int idx1 = 0;
    float aDeg = 0.0f;
    if (sscanf(line.c_str(), "MODEL %d %f", &idx1, &aDeg) == 2 && idx1 >= 1 && idx1 <= (int)NUM_SERVOS) {
      int cmd = clampDeg((int)roundf(SERVO_CENTER_DEG[idx1 - 1] + SERVO_SIGN[idx1 - 1] * aDeg));
      servos[idx1 - 1].write(cmd);
      SLOG_PRINTF("# MODEL servo %d angle=%.2f -> %d deg\n", idx1, aDeg, cmd);
      return;
    }
    SLOG_PRINTLN("# Usage: MODEL i angle_deg   (i=1..6, angle in model degrees)");
    return;
  }

  if (u.startsWith("SWEEP ")) {
    int idx1 = 0;
    float startDeg = 30.0f;
    float endDeg   = 150.0f;
    float stepDeg  = 1.0f;
    int delayMs    = 25;

    int n = sscanf(line.c_str(), "SWEEP %d %f %f %f %d", &idx1, &startDeg, &endDeg, &stepDeg, &delayMs);
    if (n >= 1 && idx1 >= 1 && idx1 <= (int)NUM_SERVOS) {
      if (stepDeg == 0.0f) stepDeg = 1.0f;
      // Ensure step sign matches direction.
      if ((endDeg - startDeg) > 0 && stepDeg < 0) stepDeg = -stepDeg;
      if ((endDeg - startDeg) < 0 && stepDeg > 0) stepDeg = -stepDeg;

      SLOG_PRINTF("# SWEEP servo %d: start=%.1f end=%.1f step=%.2f delay=%dms\n",
                  idx1, startDeg, endDeg, stepDeg, delayMs);

      for (float d = startDeg; (stepDeg > 0) ? (d <= endDeg) : (d >= endDeg); d += stepDeg) {
        int cmd = clampDeg((int)roundf(d));
        servos[idx1 - 1].write(cmd);
        delay((uint32_t)max(0, delayMs));
        // Allow WiFi debug server to stay responsive during sweep.
#if STEWART_WIFI_DEBUG
        dbgServer.handleClient();
#endif
      }
      SLOG_PRINTLN("# SWEEP done.");
      return;
    }
    SLOG_PRINTLN("# Usage: SWEEP i [start end step delay_ms]   (i=1..6)");
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

    uint32_t seq = ++gPoseSeq;
    uint32_t tApplyStart = micros();
    // Save requested (pre-saturation) for telemetry.
    gLastReqRoll = rollDeg;
    gLastReqPitch = pitchDeg;
    gLastReqYaw = yawDeg;
    bool ikOk = applyPoseIK(x, y, z, rollDeg, pitchDeg, yawDeg, -90.0f, 90.0f);
    uint32_t tDone = micros();

#if STEWART_WIFI_DEBUG
    // Human-readable ACK/status line (still easy to parse):
    // ACK seq=<n> status=<OK|SAT|IK_FAIL|PARSE_FAIL|RX_OVERFLOW> flags=<TOKENS>
    //     rx_us=<t> done_us=<t> dt_us=<t> ik_mask=0xXX sat=<0|1> lim=<deg> mag=<in>-><out>
    //     req=<x,y,z,r,p,y> applied_rpy=<r,p,y>
    uint32_t flags = ST_CMD_POSE | ST_PARSE_OK;
    if (WiFi.status() == WL_CONNECTED) flags |= ST_WIFI_CONNECTED;
    if (gLastSatApplied) flags |= ST_SAT_APPLIED;
    if (gLastZClamped) flags |= ST_Z_CLAMPED;
    if (ikOk) flags |= ST_IK_OK;
    else flags |= ST_IK_FAIL;

    char toks[80];
    flagsToTokens(flags, toks, sizeof(toks));

    char buf[300];
    snprintf(
      buf, sizeof(buf),
      "ACK seq=%lu status=%s flags=%s rx_us=%lu done_us=%lu dt_us=%lu ik_mask=0x%02X sat=%d lim=%.3f mag=%.3f->%.3f req=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f applied_rpy=%.3f,%.3f,%.3f",
      (unsigned long)seq,
      statusWordFrom(flags),
      toks,
      (unsigned long)rxUs,
      (unsigned long)tDone,
      (unsigned long)(tDone - rxUs),
      (unsigned)gLastIkOkMask,
      gLastSatApplied ? 1 : 0,
      (double)gLastSatLimitDeg,
      (double)gLastSatMagInDeg,
      (double)gLastSatMagOutDeg,
      (double)x, (double)y, (double)z,
      (double)gLastReqRoll, (double)gLastReqPitch, (double)gLastReqYaw,
      (double)gLastAppliedRoll, (double)gLastAppliedPitch, (double)gLastAppliedYaw
    );
    emitStatusLine(buf);
#endif
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
      uint32_t rxUs = micros();
      processLine(rxLine, rxUs);
      rxLen = 0;
      continue;
    }

    if (rxLen + 1 < RX_LINE_MAX) {
      rxLine[rxLen++] = c;
    } else {
      // Overflow: drop the line to resync on next newline.
      rxLen = 0;
#if STEWART_WIFI_DEBUG
      uint32_t flags = ST_PARSE_FAIL | ST_RX_OVERFLOW;
      if (WiFi.status() == WL_CONNECTED) flags |= ST_WIFI_CONNECTED;
      uint32_t now = micros();
      char toks[80];
      flagsToTokens(flags, toks, sizeof(toks));
      char buf[220];
      snprintf(
        buf, sizeof(buf),
        "ACK seq=0 status=%s flags=%s rx_us=%lu done_us=%lu dt_us=0 ik_mask=0x00 sat=0 lim=0 mag=0->0 req=0,0,0,0,0,0 applied_rpy=0,0,0",
        statusWordFrom(flags),
        toks,
        (unsigned long)now,
        (unsigned long)now
      );
      emitStatusLine(buf);
#endif
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
  dbgUdp.begin(STATUS_UDP_PORT); // also fine when sending broadcast; enables stack
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