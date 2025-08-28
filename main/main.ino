/*
 * ESP32 Stewart Platform IK
 * - Pins preserved: legs = {4,13,16,17,18,19}, reserves {22,23,25,26}
 * - Servo convention: command = -90..+90 (up..down), we clamp to ±45
 * - Even legs (0,2,4) inverted mechanically
 * - Serial input: roll pitch yaw x y z   [deg, mm]
 */

#include <ESP32Servo.h>
#include <math.h>

/* ================= Consts / config ================= */
constexpr uint8_t  N_LEG = 6;
constexpr float    DEG2RAD = 3.14159265358979323846f / 180.f;
constexpr float    RAD2DEG = 180.f / 3.14159265358979323846f;
constexpr float    CMD_LIMIT = 45.0f;       // clamp ±45°
constexpr int      PWM_MIN_US = 1000;
constexpr int      PWM_MAX_US = 2000;

/* geometry (mm) */
constexpr float linkLen   = 162.21128f;     // rod c-c length |l|
constexpr float hornLen   = 35.0f;
constexpr float hornOff   = 11.3f;
constexpr float hSq       = hornLen*hornLen + hornOff*hornOff;

/* servo axis and horn base vector (z-axis rotation) */
constexpr float u[3]      = {0.f, 0.f, 1.f};
constexpr float Hlocal[3] = {hornOff, hornLen, 0.f};

/* CAD seed anchors (rotate by 60° to make 6) */
constexpr float P1[3] = {-50.f, 100.f, 10.f};      // bottom (servo) anchor
constexpr float A1[3] = { -7.5f, 111.3f, 152.5f};  // platform anchor

/* pins */
const uint8_t legPins[N_LEG] = {4,13,16,17,18,19};
const uint8_t ctrlPins[4]    = {22,23,25,26};

/* even legs inverted due to mounting: 0,2,4 */
const bool invertLeg[N_LEG]  = {true,false,true,false,true,false};

/* optional small mechanical trims (deg), start at 0 */
int servoTrim[N_LEG] = {0,0,0,0,0,0};

/* ================= State ================= */
Servo leg[N_LEG];
Servo ctrl[4];

float P[N_LEG][3], A[N_LEG][3], H[N_LEG][3];
float lastThetaDeg[N_LEG];   // absolute servo angle (0..180) solution
float zeroOffsetDeg[N_LEG];  // IK angle at neutral pose (used as 0 command)

/* ================= Helpers ================= */
static inline float clampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }

void Rz_fill(float deg, float R[3][3]){
  float r = deg * DEG2RAD, c = cosf(r), s = sinf(r);
  R[0][0]=c;  R[0][1]=-s; R[0][2]=0;
  R[1][0]=s;  R[1][1]= c; R[1][2]=0;
  R[2][0]=0;  R[2][1]= 0; R[2][2]=1;
}

void buildR_ZYX(float roll,float pitch,float yaw, float R[3][3]){
  float cr = cosf(roll),  sr = sinf(roll);
  float cp = cosf(pitch), sp = sinf(pitch);
  float cy = cosf(yaw),   sy = sinf(yaw);
  R[0][0]=cy*cp;                R[0][1]=cy*sp*sr - sy*cr;   R[0][2]=cy*sp*cr + sy*sr;
  R[1][0]=sy*cp;                R[1][1]=sy*sp*sr + cy*cr;   R[1][2]=sy*sp*cr - cy*sr;
  R[2][0]=-sp;                  R[2][1]=cp*sr;              R[2][2]=cp*cr;
}

void genAnchors(){
  for(uint8_t k=0;k<N_LEG;k++){
    float Rz[3][3]; Rz_fill(60.f*k, Rz);
    // Pk = Rz * P1
    P[k][0] = Rz[0][0]*P1[0] + Rz[0][1]*P1[1];
    P[k][1] = Rz[1][0]*P1[0] + Rz[1][1]*P1[1];
    P[k][2] = P1[2];
    // Ak = Rz * A1
    A[k][0] = Rz[0][0]*A1[0] + Rz[0][1]*A1[1];
    A[k][1] = Rz[1][0]*A1[0] + Rz[1][1]*A1[1];
    A[k][2] = A1[2];
    // Hk = Rz * Hlocal (horn initially radial)
    H[k][0] = Rz[0][0]*Hlocal[0] + Rz[0][1]*Hlocal[1];
    H[k][1] = Rz[1][0]*Hlocal[0] + Rz[1][1]*Hlocal[1];
    H[k][2] = 0.f;
    lastThetaDeg[k] = NAN;
    zeroOffsetDeg[k]= 0.f;
  }
}

/* returns true if solution found; thetaDegAbs in [0,180] */
bool ikSolveOne(uint8_t i, const float R[3][3], const float T[3], float &thetaDegAbs){
  // s = R*A + T - P
  float s[3];
  for (uint8_t j=0;j<3;j++)
    s[j] = R[j][0]*A[i][0] + R[j][1]*A[i][1] + R[j][2]*A[i][2] + T[j] - P[i][j];

  // a cosθ + b sinθ = gamma  (c=0 because Hz=0, u=z)
  float a = s[0]*H[i][0] + s[1]*H[i][1];
  float b = s[0]*(-H[i][1]) + s[1]*H[i][0];
  float sSq = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  float gamma = 0.5f * (sSq + hSq - linkLen*linkLen);

  float r = sqrtf(a*a + b*b);
  if (r < 1e-6f){
    // Degenerate (shouldn’t happen in normal geometry)
    return false;
  }

  float phi = atan2f(b,a);
  float x   = clampf(gamma / r, -1.f, 1.f);
  float d   = acosf(x);

  float cand1 = (phi + d) * RAD2DEG;
  float cand2 = (phi - d) * RAD2DEG;

  // normalize to [0,360) then fold to [0,180] since horn repeats every 180°
  auto fold180 = [](float th){
    // bring to [-180,180)
    while (th >= 180.f) th -= 360.f;
    while (th <  -180.f) th += 360.f;
    // mirror negative angles
    if (th < 0.f) th = -th;
    return th; // [0,180]
  };

  cand1 = fold180(cand1);
  cand2 = fold180(cand2);

  // choose closest to lastTheta if available
  float pick = cand1;
  if (!isnan(lastThetaDeg[i])){
    float d1 = fabsf(cand1 - lastThetaDeg[i]);
    float d2 = fabsf(cand2 - lastThetaDeg[i]);
    pick = (d2 < d1) ? cand2 : cand1;
  } else {
    // heuristic: prefer the smaller angle initially
    pick = (cand1 < cand2) ? cand1 : cand2;
  }

  pick = clampf(pick, 0.f, 180.f);
  thetaDegAbs = pick;
  return true;
}

/* map command (−90..+90) to servo.write(0..180) with inversion & trim */
int toServoWriteDeg(uint8_t i, float cmdDeg){
  cmdDeg = clampf(cmdDeg, -CMD_LIMIT, +CMD_LIMIT);
  float std = 90.f + cmdDeg + servoTrim[i];     // neutral → 90°
  int val = (int)roundf(std);
  if (invertLeg[i]) val = 180 - val;            // mirror for inverted legs
  val = (val < 0) ? 0 : (val > 180 ? 180 : val);
  return val;
}

/* ================= Setup / Loop ================= */
void setup(){
  Serial.begin(115200);
  // Allocate LEDC timers for ESP32Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  genAnchors();

  // Attach leg servos
  for(uint8_t i=0;i<N_LEG;i++){
    leg[i].setPeriodHertz(50);
    leg[i].attach(legPins[i], PWM_MIN_US, PWM_MAX_US);
    leg[i].write(90); // hold neutral pulse
  }
  // Attach control-surface servos and center
  for (uint8_t k=0;k<4;k++){
    ctrl[k].setPeriodHertz(50);
    ctrl[k].attach(ctrlPins[k], PWM_MIN_US, PWM_MAX_US);
    ctrl[k].write(90);
  }

  // Compute IK neutral offsets at CAD home pose (R=I, T=0)
  float Rn[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
  float Tn[3] = {0,0,0};
  bool okAll = true;
  for(uint8_t i=0;i<N_LEG;i++){
    float th;
    bool ok = ikSolveOne(i, Rn, Tn, th);
    if(!ok){ okAll = false; th = 90.f; }
    zeroOffsetDeg[i] = th;   // store absolute angle at neutral
    lastThetaDeg[i]  = th;
    // Command neutral (0 cmd → 90 std)
    leg[i].write( toServoWriteDeg(i, 0.f) );
  }

  Serial.println("boot");
  Serial.print("# Neutral IK offsets (deg): ");
  for(uint8_t i=0;i<N_LEG;i++){ Serial.print(zeroOffsetDeg[i],1); Serial.print(i==N_LEG-1?'\n':' '); }
  if(!okAll) Serial.println("# WARN: some neutral IK solves failed; check geometry.");
  Serial.println("# Send: roll pitch yaw x y z  (deg mm)");
}

void loop(){
  if(Serial.available()){
    float roll  = Serial.parseFloat();
    float pitch = Serial.parseFloat();
    float yaw   = Serial.parseFloat();
    float Tx    = Serial.parseFloat();
    float Ty    = Serial.parseFloat();
    float Tz    = Serial.parseFloat();
    Serial.readStringUntil('\n'); // flush rest of line

    float R[3][3]; buildR_ZYX(roll*DEG2RAD, pitch*DEG2RAD, yaw*DEG2RAD, R);
    float T[3] = {Tx, Ty, Tz};

    // Solve each leg, convert to command relative to neutral, clamp to ±45°, write
    for(uint8_t i=0;i<N_LEG;i++){
      float thAbs;
      if(ikSolveOne(i, R, T, thAbs)){
        lastThetaDeg[i] = thAbs;
        float cmd = thAbs - zeroOffsetDeg[i];      // deg, where 0 = neutral
        cmd = //!clampf(cmd, -CMD_LIMIT, +CMD_LIMIT); // safety limit
        leg[i].write( toServoWriteDeg(i, cmd) );
        Serial.print(cmd,1);                       // report commanded angle
      }else{
        // keep last command; print NaN to flag
        Serial.print("nan");
      }
      Serial.print(i==N_LEG-1? '\n' : '\t');
    }
  }
}
