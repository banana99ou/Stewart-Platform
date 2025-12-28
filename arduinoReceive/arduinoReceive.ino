/*
  ESP32 Serial -> WiFi Web Viewer

  Purpose:
  - Simulink occupies the serial port, so you can't use Serial Monitor to verify data.
  - This sketch reads incoming Serial bytes (from Simulink) and exposes them via a
    simple webpage served by the ESP32 over Wi-Fi.

  What you get on the webpage:
  - Total bytes received
  - Last byte (dec/hex)
  - Last N bytes (hex)
  - “Decoded float” every 4 bytes (little-endian, as-is)

  Notes:
  - This does NOT change your Simulink->ESP32 serial link; it only adds visibility.
  - Float decoding assumes the sender packs IEEE754 float into 4 bytes.
*/

#include <WiFi.h>
#include <WebServer.h>

// ---------------- Debug logging ----------------
// Set to 0 to silence all debug prints.
#ifndef DEBUG_SERIAL
#define DEBUG_SERIAL 1
#endif

// Set to 1 to print every incoming byte (can be very spammy).
#ifndef DEBUG_VERBOSE_BYTES
#define DEBUG_VERBOSE_BYTES 0
#endif

#if DEBUG_SERIAL
  #define DBG_PRINTLN(x)  Serial.println(x)
  #define DBG_PRINT(x)    Serial.print(x)
  #define DBG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define DBG_PRINTLN(x)  do {} while (0)
  #define DBG_PRINT(x)    do {} while (0)
  #define DBG_PRINTF(...) do {} while (0)
#endif

// ---------------- WiFi config ----------------
static const char *WIFI_SSID = "iptime";
static const char *WIFI_PASS = "Za!cW~QWdh5FC~f";

// ---------------- Web server ----------------
static WebServer server(80);

// ---------------- Serial capture ----------------
static constexpr size_t RING_SIZE = 256;
static uint8_t ringBuf[RING_SIZE];
static size_t ringHead = 0;

static uint32_t totalBytes = 0;
static bool hasLastByte = false;
static uint8_t lastByte = 0;

typedef union {
  float number;
  uint8_t bytes[4];
} FLOATUNION_t;

static uint8_t floatAcc[4];
static uint8_t floatAccIdx = 0;
static bool hasLastFloat = false;
static float lastFloat = 0.0f;
static uint8_t lastFloatBytes[4] = {0, 0, 0, 0};
static uint32_t floatsDecoded = 0;

// Keep a short history of decoded floats for easier debugging.
static constexpr size_t FLOAT_RING_SIZE = 24;
static float floatRing[FLOAT_RING_SIZE];
static uint8_t floatRingBytes[FLOAT_RING_SIZE][4];
static size_t floatRingHead = 0;
static uint32_t floatRingCount = 0;

static String hexByte(uint8_t b) {
  const char *hex = "0123456789ABCDEF";
  String s;
  s.reserve(2);
  s += hex[(b >> 4) & 0x0F];
  s += hex[b & 0x0F];
  return s;
}

static String lastNBytesHex(size_t n) {
  if (n == 0) return String("-");
  if (n > RING_SIZE) n = RING_SIZE;
  String out;
  out.reserve(n * 3);

  // Oldest -> newest among the last n bytes
  size_t start = (ringHead + RING_SIZE - n) % RING_SIZE;
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = ringBuf[(start + i) % RING_SIZE];
    out += hexByte(b);
    if (i + 1 < n) out += ' ';
  }
  return out;
}

static String lastNBytesDec(size_t n) {
  if (n == 0) return String("-");
  if (n > RING_SIZE) n = RING_SIZE;
  String out;
  // Rough reserve: up to 3 digits + space each
  out.reserve(n * 4);

  size_t start = (ringHead + RING_SIZE - n) % RING_SIZE;
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = ringBuf[(start + i) % RING_SIZE];
    out += String((unsigned)b);
    if (i + 1 < n) out += ' ';
  }
  return out;
}

static String lastNBytesAscii(size_t n) {
  if (n == 0) return String("-");
  if (n > RING_SIZE) n = RING_SIZE;
  String out;
  out.reserve(n);

  size_t start = (ringHead + RING_SIZE - n) % RING_SIZE;
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = ringBuf[(start + i) % RING_SIZE];
    if (b == '\n') out += "\\n";
    else if (b == '\r') out += "\\r";
    else if (b == '\t') out += "\\t";
    else if (b >= 32 && b <= 126) out += (char)b;
    else out += '.';
  }
  return out;
}

static String lastNFloatsText(size_t n) {
  if (floatRingCount == 0) return String("-");
  if (n > FLOAT_RING_SIZE) n = FLOAT_RING_SIZE;
  if (n > (size_t)floatRingCount) n = (size_t)floatRingCount;

  String out;
  out.reserve(n * 56);

  // Newest -> oldest
  for (size_t i = 0; i < n; ++i) {
    size_t idx = (floatRingHead + FLOAT_RING_SIZE - 1 - i) % FLOAT_RING_SIZE;
    out += String(floatRing[idx], 6);
    out += "  [";
    out += hexByte(floatRingBytes[idx][0]); out += ' ';
    out += hexByte(floatRingBytes[idx][1]); out += ' ';
    out += hexByte(floatRingBytes[idx][2]); out += ' ';
    out += hexByte(floatRingBytes[idx][3]);
    out += "]";
    if (i + 1 < n) out += '\n';
  }
  return out;
}

// JSON string escaping (critical: raw newlines/backslashes/quotes will break JSON parsing)
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
        // Escape other control chars as \u00XX
        if ((uint8_t)c < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04X", (unsigned)((uint8_t)c));
          out += buf;
        } else {
          out += c;
        }
    }
  }
}

static void jsonAddKV(String &json, const char *key, const String &value) {
  json += "\"";
  json += key;
  json += "\":\"";
  jsonAppendEscaped(json, value);
  json += "\"";
}

static void handleRoot() {
  String html;
  html.reserve(3600);
  html += "<!doctype html><html><head><meta charset='utf-8'/>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'/>";
  html += "<title>ESP32 Serial Viewer</title>";
  html += "<style>";
  html += "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:16px;}";
  html += ".card{max-width:900px;border:1px solid #ddd;border-radius:12px;padding:16px;}";
  html += "code,pre{background:#f6f8fa;border-radius:8px;padding:10px;display:block;overflow:auto;}";
  html += ".row{display:flex;gap:12px;flex-wrap:wrap;}";
  html += ".k{min-width:220px;}";
  html += "button{padding:10px 14px;border-radius:10px;border:1px solid #ccc;background:#fff;cursor:pointer;}";
  html += "</style></head><body>";
  html += "<div class='card'>";
  html += "<h2>ESP32 Serial Viewer</h2>";
  html += "<div class='row'>";
  html += "<div class='k'><b>Wi-Fi:</b> <span id='wifi'>...</span></div>";
  html += "<div class='k'><b>IP:</b> <span id='ip'>...</span></div>";
  html += "<div class='k'><b>Uptime (ms):</b> <span id='up'>...</span></div>";
  html += "</div>";
  html += "<hr/>";
  html += "<div class='row'>";
  html += "<div class='k'><b>Total bytes:</b> <span id='bytes'>0</span></div>";
  html += "<div class='k'><b>Last byte:</b> <span id='lastb'>-</span></div>";
  html += "<div class='k'><b>Floats decoded:</b> <span id='floats'>0</span></div>";
  html += "</div>";
  html += "<p><b>Last bytes (hex):</b></p><pre id='hex'>-</pre>";
  html += "<p><b>Last bytes (decimal):</b></p><pre id='dec'>-</pre>";
  html += "<p><b>Last bytes (ASCII, non-printable shown as '.'): </b></p><pre id='asc'>-</pre>";
  html += "<p><b>Last float:</b> <span id='lf'>-</span> <span id='lfb'></span></p>";
  html += "<p><b>Last 12 floats (newest first):</b></p><pre id='fl'>-</pre>";
  html += "<div class='row'>";
  html += "<button onclick='clearStats()'>Clear counters</button>";
  html += "<button onclick='location.reload()'>Reload page</button>";
  html += "</div>";
  html += "<p style='color:#666'>Polling <code>/data</code> every 250ms.</p>";
  html += "</div>";
  html += "<script>";
  html += "async function tick(){";
  html += "const r=await fetch('/data',{cache:'no-store'});";
  html += "const j=await r.json();";
  html += "document.getElementById('wifi').textContent=j.wifi;";
  html += "document.getElementById('ip').textContent=j.ip;";
  html += "document.getElementById('up').textContent=j.uptime_ms;";
  html += "document.getElementById('bytes').textContent=j.total_bytes;";
  html += "document.getElementById('lastb').textContent=j.last_byte;";
  html += "document.getElementById('floats').textContent=j.floats_decoded;";
  html += "document.getElementById('hex').textContent=j.last_64_hex;";
  html += "document.getElementById('dec').textContent=j.last_64_dec;";
  html += "document.getElementById('asc').textContent=j.last_64_ascii;";
  html += "document.getElementById('lf').textContent=j.last_float;";
  html += "document.getElementById('lfb').textContent=j.last_float_bytes;";
  html += "document.getElementById('fl').textContent=j.last_floats;";
  html += "}";
  html += "setInterval(()=>tick().catch(()=>{}),250);tick().catch(()=>{});";
  html += "async function clearStats(){await fetch('/clear',{method:'POST'});}";
  html += "</script></body></html>";

  server.send(200, "text/html; charset=utf-8", html);
}

static void handleData() {
  String wifi = (WiFi.status() == WL_CONNECTED) ? "connected" : "disconnected";
  String ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP().toString() : "0.0.0.0";

  String lastB;
  if (hasLastByte) {
    lastB.reserve(32);
    lastB += String((int)lastByte);
    lastB += " (0x";
    lastB += hexByte(lastByte);
    lastB += ")";
  } else {
    lastB = "-";
  }

  String lf = hasLastFloat ? String(lastFloat, 6) : String("-");
  String lfb;
  if (hasLastFloat) {
    lfb.reserve(32);
    lfb += " [";
    lfb += hexByte(lastFloatBytes[0]); lfb += ' ';
    lfb += hexByte(lastFloatBytes[1]); lfb += ' ';
    lfb += hexByte(lastFloatBytes[2]); lfb += ' ';
    lfb += hexByte(lastFloatBytes[3]);
    lfb += "]";
  } else {
    lfb = "";
  }

  size_t n = 64;
  if (totalBytes < n) n = (size_t)totalBytes;
  String hex64 = lastNBytesHex(n);
  String dec64 = lastNBytesDec(n);
  String asc64 = lastNBytesAscii(n);
  String lastFloats = lastNFloatsText(12);

  String json;
  json.reserve(800 + hex64.length() + dec64.length() + asc64.length() + lastFloats.length());
  json += "{";
  jsonAddKV(json, "wifi", wifi); json += ",";
  jsonAddKV(json, "ip", ip); json += ",";
  json += "\"uptime_ms\":" + String(millis()) + ",";
  json += "\"total_bytes\":" + String(totalBytes) + ",";
  jsonAddKV(json, "last_byte", lastB); json += ",";
  json += "\"floats_decoded\":" + String(floatsDecoded) + ",";
  jsonAddKV(json, "last_64_hex", hex64); json += ",";
  jsonAddKV(json, "last_64_dec", dec64); json += ",";
  jsonAddKV(json, "last_64_ascii", asc64); json += ",";
  jsonAddKV(json, "last_float", lf); json += ",";
  jsonAddKV(json, "last_float_bytes", lfb); json += ",";
  jsonAddKV(json, "last_floats", lastFloats);
  json += "}";

  server.send(200, "application/json; charset=utf-8", json);
}

static void handleClear() {
  totalBytes = 0;
  floatsDecoded = 0;
  hasLastByte = false;
  hasLastFloat = false;
  floatAccIdx = 0;
  for (size_t i = 0; i < RING_SIZE; ++i) ringBuf[i] = 0;
  floatRingHead = 0;
  floatRingCount = 0;
  DBG_PRINTLN("[web] clear counters");
  server.send(200, "text/plain; charset=utf-8", "ok");
}

static void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  DBG_PRINTLN("[wifi] not connected; (re)connecting...");
  WiFi.disconnect(false, false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

static void readSerialNonBlocking() {
  uint32_t nThisLoop = 0;
  while (Serial.available() > 0) {
    int b = Serial.read();
    if (b < 0) break;
    uint8_t ub = (uint8_t)b;

    ringBuf[ringHead] = ub;
    ringHead = (ringHead + 1) % RING_SIZE;

    lastByte = ub;
    hasLastByte = true;
    totalBytes++;
    nThisLoop++;

#if DEBUG_SERIAL && DEBUG_VERBOSE_BYTES
    DBG_PRINTF("[rx] byte=%u (0x%s)\n", (unsigned)ub, hexByte(ub).c_str());
#endif

    // Decode float every 4 bytes (as received).
    floatAcc[floatAccIdx++] = ub;
    if (floatAccIdx >= 4) {
      FLOATUNION_t u;
      u.bytes[0] = floatAcc[0];
      u.bytes[1] = floatAcc[1];
      u.bytes[2] = floatAcc[2];
      u.bytes[3] = floatAcc[3];
      lastFloat = u.number;
      lastFloatBytes[0] = floatAcc[0];
      lastFloatBytes[1] = floatAcc[1];
      lastFloatBytes[2] = floatAcc[2];
      lastFloatBytes[3] = floatAcc[3];
      hasLastFloat = true;
      floatsDecoded++;
      floatAccIdx = 0;

      floatRing[floatRingHead] = lastFloat;
      floatRingBytes[floatRingHead][0] = lastFloatBytes[0];
      floatRingBytes[floatRingHead][1] = lastFloatBytes[1];
      floatRingBytes[floatRingHead][2] = lastFloatBytes[2];
      floatRingBytes[floatRingHead][3] = lastFloatBytes[3];
      floatRingHead = (floatRingHead + 1) % FLOAT_RING_SIZE;
      floatRingCount++;

#if DEBUG_SERIAL
      DBG_PRINTF("[rx] float=%0.6f bytes=[%s %s %s %s]\n",
                 lastFloat,
                 hexByte(lastFloatBytes[0]).c_str(),
                 hexByte(lastFloatBytes[1]).c_str(),
                 hexByte(lastFloatBytes[2]).c_str(),
                 hexByte(lastFloatBytes[3]).c_str());
#endif
    }
  }

#if DEBUG_SERIAL
  // Rate-limited rx stats (avoid spamming when data rate is high).
  static uint32_t lastStatsMs = 0;
  if (nThisLoop > 0 && (millis() - lastStatsMs) > 1000) {
    lastStatsMs = millis();
    if (hasLastByte) {
      DBG_PRINTF("[rx] totalBytes=%lu floats=%lu lastByte=%u (0x%s)\n",
                 (unsigned long)totalBytes,
                 (unsigned long)floatsDecoded,
                 (unsigned)lastByte,
                 hexByte(lastByte).c_str());
    } else {
      DBG_PRINTF("[rx] totalBytes=%lu floats=%lu\n",
                 (unsigned long)totalBytes,
                 (unsigned long)floatsDecoded);
    }
  }
#endif
}

void setup() {
  // Serial is REQUIRED for RX (incoming bytes), regardless of debug printing.
  Serial.begin(115200);

  // Debug prints (TX) share Serial. If Simulink is also receiving from the ESP32,
  // debug output can corrupt that receive stream; in that case set DEBUG_SERIAL=0.
  DBG_PRINTLN();
  DBG_PRINTLN("ESP32 Serial->WiFi Viewer booting...");

  // Wi-Fi + web server.
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  DBG_PRINTF("[wifi] connecting to SSID '%s'...\n", WIFI_SSID);

  server.on("/", HTTP_GET, handleRoot);
  server.on("/data", HTTP_GET, handleData);
  server.on("/clear", HTTP_POST, handleClear);
  server.begin();
  DBG_PRINTLN("[web] server started on port 80");
}

void loop() {
  // Keep Wi-Fi alive (non-blocking reconnect attempts).
  static uint32_t lastWiFiKick = 0;
  if (millis() - lastWiFiKick > 3000) {
    lastWiFiKick = millis();
    ensureWiFi();
  }

  // Print one-time "connected" message once we have an IP.
#if DEBUG_SERIAL
  static bool printedWiFiUp = false;
  if (!printedWiFiUp && WiFi.status() == WL_CONNECTED) {
    printedWiFiUp = true;
    DBG_PRINTF("[wifi] connected. IP=%s RSSI=%d dBm\n",
               WiFi.localIP().toString().c_str(),
               WiFi.RSSI());
    DBG_PRINTLN("[web] open http://<ip>/ in your browser");
  }
  if (printedWiFiUp && WiFi.status() != WL_CONNECTED) {
    printedWiFiUp = false;
    DBG_PRINTLN("[wifi] disconnected");
  }
#endif

  readSerialNonBlocking();
  server.handleClient();
  delay(2);
}