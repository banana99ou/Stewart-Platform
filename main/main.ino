/* 6-DOF Stewart platform
* --------------------------------
* Send on Serial (115 200 baud) a single line:
*      roll  pitch  yaw  x  y  z
* Units: degrees, millimetres.
*
* The sketch solves the crank-rocker inverse kinematics for
* six identical legs and writes the angles to servos.
*
* HW: ESP32 board, 6× SG-90/MG-90 etc.
*/

#include <ESP32Servo.h>
#include <math.h>

constexpr uint8_t  N_LEG = 6;
constexpr float    DEG2RAD = 3.14159265358979323846f / 180.f;
constexpr float    RAD2DEG = 180.f / 3.14159265358979323846f;

/* ------- geometry (millimetres) -------- */
constexpr float linkLen   = 162.21128f;          // rod c-c length  |l|
constexpr float hornLen   = 35.0f;               // horn physical length
constexpr float hornOff   = 11.3f;               // radial offset
constexpr float hSq       = hornLen*hornLen + hornOff*hornOff;   // |H|²

/* servo-axis – all along +Z (change if needed) */
constexpr float u[3] = {0.f, 0.f, 1.f};

/* horn tip vector in *servo* frame for θ=0 (change per leg if needed) */
constexpr float Hlocal[3] = {hornOff, hornLen, 0.f};

/* ---- CAD anchor points ------------------------------------------------- */
/* seed leg (#1) -— rotated in 60° steps about Z to make the hexagon        */
constexpr float P1[3] = {-50.f, 100.f, 10.f};            // bottom anchor
constexpr float A1[3] = { -7.5f, 111.3f, 152.5f};        // platform anchor

float P[N_LEG][3];
float A[N_LEG][3];
float H[N_LEG][3];

/* servo objects / pins */
Servo servo[N_LEG];
constexpr uint8_t servoPins[N_LEG] = {4,13,16,17,18,19};

int Controlsurfaces[4] = {22,23,25,26};
Servo servo6;
Servo servo7;
Servo servo8;
Servo servo9;

/* ----------------------------------------------------------------------- */

void makeAnchors() {
  for (uint8_t k=0;k<N_LEG;k++) {
    float ang = DEG2RAD * 60.f * k;
    float c = cosf(ang), s = sinf(ang);

    // rotate seed anchors in XY
    P[k][0] =  c*P1[0] - s*P1[1];
    P[k][1] =  s*P1[0] + c*P1[1];
    P[k][2] =  P1[2];

    A[k][0] =  c*A1[0] - s*A1[1];
    A[k][1] =  s*A1[0] + c*A1[1];
    A[k][2] =  A1[2];

    /* horn-tip vector, assumed to start radially outward */
    H[k][0] =  c*Hlocal[0] - s*Hlocal[1];
    H[k][1] =  s*Hlocal[0] + c*Hlocal[1];
    H[k][2] =  0.f;
  }
}

/* rotation matrix R = Rz(yaw) · Ry(pitch) · Rx(roll) (ZYX convention) */
void buildR(float roll,float pitch,float yaw, float R[3][3]) {
  float cr = cosf(roll),  sr = sinf(roll);
  float cp = cosf(pitch), sp = sinf(pitch);
  float cy = cosf(yaw),   sy = sinf(yaw);

  R[0][0]=cy*cp;                R[0][1]=cy*sp*sr - sy*cr;   R[0][2]=cy*sp*cr + sy*sr;
  R[1][0]=sy*cp;                R[1][1]=sy*sp*sr + cy*cr;   R[1][2]=sy*sp*cr - cy*sr;
  R[2][0]=-sp;                  R[2][1]=cp*sr;              R[2][2]=cp*cr;
}

/* IK for one platform pose ------------------------------------------------*/
void solve(const float R[3][3], const float T[3]) {
  for(uint8_t i=0;i<N_LEG;i++) {
    /* s_i = R*A_i + T − P_i */
    float s[3];
    for(uint8_t j=0;j<3;j++)
      s[j] = R[j][0]*A[i][0] + R[j][1]*A[i][1] + R[j][2]*A[i][2] + T[j] - P[i][j];

    /* a_i, b_i, c_i for the sin-cos form (u = [0 0 1]) */
    float a = s[0]*H[i][0] + s[1]*H[i][1];          // s·H⊥
    float b = s[0]*(-H[i][1]) + s[1]*H[i][0];       // s·(u×H)
    float c = s[2]*H[i][2];                         // (sz)(Hz)  (=0 here)

    float sSq   = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
    float gamma = (sSq + hSq - linkLen*linkLen)*0.5f;
    float rhs   = gamma - c;
    float rMag  = sqrtf(a*a + b*b);

    float thetaDeg = 90.f;      // fallback mid-stroke

    if(rMag > 1e-4f) {
      float phi   = atan2f(b,a);
      float ratio = rhs / rMag;
      ratio = constrain(ratio, -1.f, 1.f);
      float delta = acosf(ratio);

      float t1 = phi + delta;
      float t2 = phi - delta;
      float t1d = t1*RAD2DEG;
      float t2d = t2*RAD2DEG;

      /* pick whichever lies inside the servo limits 0-180° */
      if( 0.f<=t1d && t1d<=180.f )      thetaDeg = t1d;
      else if( 0.f<=t2d && t2d<=180.f ) thetaDeg = t2d;
    }

    servo[i].write(thetaDeg);               // drive motor
    Serial.print(thetaDeg,1);               // echo
    Serial.print(i==N_LEG-1? '\n':'\t');
  }
}

/* ----------------------------------------------------------------------- */
void setup() {
  makeAnchors();
 
  Serial.begin(115200);
  Serial.println(F("# Send: roll pitch yaw x y z  (deg mm)"));
 
  for(uint8_t i=0;i<N_LEG;i++)
    servo[i].attach(servoPins[i]); 
  
  servo6.attach(22);
  servo7.attach(23);
  servo8.attach(25);
  servo9.attach(26);
}

void loop() {
  if(Serial.available()) {
    float roll  = Serial.parseFloat();    // degrees
    float pitch = Serial.parseFloat();
    float yaw   = Serial.parseFloat();
    float Tx    = Serial.parseFloat();    // mm
    float Ty    = Serial.parseFloat();
    float Tz    = Serial.parseFloat();
    Serial.readStringUntil('\n');         // clear rest of line

    float R[3][3];
    buildR(roll*DEG2RAD, pitch*DEG2RAD, yaw*DEG2RAD, R);
    float T[3] = {Tx,Ty,Tz};

    solve(R,T);                           // do the IK
  }
}
 

// void setup() {
//     Serial.begin(115200);

//     // analogWriteFreq(500);

//     servo0.attach(4);
//     servo1.attach(13);
//     servo2.attach(16);
//     servo3.attach(17);
//     servo4.attach(18);
//     servo5.attach(19);

//     servo6.attach(22);
//     servo7.attach(23);
//     servo8.attach(25);
//     servo9.attach(26);

//     Serial.println("boot");
// }

// void loop() {
//   // servo0.write(0);
//   for (pos = 30 * 2; pos <= 60 * 2; pos += 1) { // goes from 0 degrees to 180 degrees in steps of 1 degree
//     // analogWrite(4, pos);
//     servo0.write(pos);
// 		servo1.write(pos);    // tell servo to go to position in variable 'pos'
//     servo2.write(pos);
//     servo3.write(pos);
//     servo4.write(pos);
//     servo5.write(pos);

//     servo6.write(pos);
//     servo7.write(pos);
//     servo8.write(pos);
//     servo9.write(pos);
// 		delay(15);             // waits 15ms for the servo to reach the position
//     Serial.println(pos);
// 	}
//   delay(500);
// 	for (pos = 60 * 2; pos >= 30 * 2; pos -= 1) { // goes from 180 degrees to 0 degrees
//     // analogWrite(4, pos);
//     servo0.write(pos);
// 		servo1.write(pos);    // tell servo to go to position in variable 'pos'
//     servo2.write(pos);
//     servo3.write(pos);
//     servo4.write(pos);
//     servo5.write(pos);

//     servo6.write(pos);
//     servo7.write(pos);
//     servo8.write(pos);
//     servo9.write(pos);
// 		delay(15);             // waits 15ms for the servo to reach the position
//     Serial.println(pos);
// 	}
//   delay(500);
// }