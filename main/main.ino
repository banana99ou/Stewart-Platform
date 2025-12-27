/* Stewart Platform — Full IK, LH coordinates, ESP32 + ESP32Servo
 * - P/A/H are hard-coded (LH frame).
 * - Builder uses LH logic by negating roll & yaw inside the ZYX rotation.
 * - Command space: -90..+90 (0 = neutral). Clamp ±45 for safety.
 *
 * Serial commands:
 *   POSE  roll pitch yaw   x y z      -> solve & MOVE ALL legs
 *   IKALL roll pitch yaw   x y z      -> solve & PRINT ALL legs (no motion)
 *   IK i  roll pitch yaw   x y z      -> solve & print one leg
 *   MOVE i roll pitch yaw  x y z      -> solve & move one leg
 *   CENTER                           -> set all legs to cmd=0
 *   INV i                            -> toggle inversion on leg i
 */

 #include <ESP32Servo.h>
 #include <math.h>
 
 /* ================= Configuration ================= */
 constexpr uint8_t  NUM_LEGS = 6;
 constexpr float    COMMAND_LIMIT_DEG = 45.0f;     // clamp command to ±45°
 constexpr float    BRANCH_HYSTERESIS_DEG = 2.0f;  // neutral bias margin
 
 constexpr int      PWM_PULSE_MIN_US = 1000;
 constexpr int      PWM_PULSE_MAX_US = 2000;
 
 const uint8_t      leg_servo_pins[NUM_LEGS] = {4,13,16,17,18,19};
 
 /* Inversion tested OK on your hardware: {1,3,5} inverted */
 bool               leg_is_inverted[NUM_LEGS] = {false, false, false, false, false, false};
 
 /* Optional small mechanical trims per leg (deg) */
 int                per_leg_trim_degrees[NUM_LEGS] = {0,0,0,0,0,0};
 
 /* ================= Geometry (mm) ================= */
 constexpr float link_length_mm                 = 162.21128f; // |l|
 constexpr float horn_length_mm                 = 35.0f;
 constexpr float horn_offset_mm                 = 11.3f;
 constexpr float horn_vec_len_sq_mm2            = horn_length_mm*horn_length_mm
                                                + horn_offset_mm*horn_offset_mm;
 
 /* Bottom (servo) anchors P[i] — LH frame */
 float bottom_anchor_mm[NUM_LEGS][3] = {
   { -50.000000f, 100.000000f, 10.000000f },
   { -111.602540f,  -6.698730f, 10.000000f },
   { -61.602540f, -93.301270f, 10.000000f },
   {  61.602540f,  -93.301270f, 10.000000f },
   { 111.602540f,   -6.698730f, 10.000000f },
   {  50.000000f, 100.000000f, 10.000000f },
 };
 
 /* Platform anchors A[i] — LH frame */
 float platform_anchor_mm[NUM_LEGS][3] = {
   {  -7.500000f, 111.300000f, 152.500000f },
   { -100.138627f, -49.154809f, 152.500000f },
   { -92.638627f, -62.145191f, 152.500000f },
   {  92.638627f, -62.145191f, 152.500000f },
   { 100.138627f, -49.154809f, 152.500000f },
   {   7.500000f, 111.300000f, 152.500000f },
 };
 
 /* Horn base vectors H[i] at θ=0 (XY plane) — LH frame */
 float horn_base_vector_mm[NUM_LEGS][3] = {
   { -11.300000f, 35.000000f, 0.000000f },
   { 11.300000f, 35.000000f, 0.000000f },
   { -35.960889f,  -7.713913f, 0.000000f },
   { -35.960889f,  -7.713913f, 0.000000f },
   { 24.660889f,  -27.286087f, 0.000000f },
   { 24.660889f,  -27.286087f, 0.000000f },
 };
 
 /* ================= Runtime state ================= */
 Servo leg_servo[NUM_LEGS];
 
 /* Raw horn angle (deg) at neutral pose (R=I, T=0) per leg */
 float neutral_horn_angle_deg[NUM_LEGS];
 
 /* Last commanded relative angle (deg) per leg (for continuity) */
 float last_command_deg[NUM_LEGS];
 
 /* ================= Helpers ================= */
 static inline float clampf(float v, float lo, float hi){
   return v < lo ? lo : (v > hi ? hi : v);
 }
 
 /* Unwrap angle (deg) to be within ±180 of a reference (deg) */
 static inline float unwrap_deg_near(float angle_deg, float ref_deg){
   while (angle_deg - ref_deg > 180.f)  angle_deg -= 360.f;
   while (angle_deg - ref_deg < -180.f) angle_deg += 360.f;
   return angle_deg;
 }
 
 /* --- LH ZYX rotation: negate roll & yaw compared to RH builder --- */
 void build_rotation_matrix_zyx(float roll_rad, float pitch_rad, float yaw_rad, float R[3][3]){
   const float cr = cosf(-roll_rad),  sr = sinf(-roll_rad);
   const float cp = cosf( pitch_rad), sp = sinf( pitch_rad);
   const float cy = cosf( -yaw_rad),  sy = sinf( -yaw_rad);
 
   R[0][0]=cy*cp;                R[0][1]=cy*sp*sr - sy*cr;   R[0][2]=cy*sp*cr + sy*sr;
   R[1][0]=sy*cp;                R[1][1]=sy*sp*sr + cy*cr;   R[1][2]=sy*sp*cr - cy*sr;
   R[2][0]=-sp;                  R[2][1]=cp*sr;              R[2][2]=cp*cr;
 }
 
 /* ================= IK: two solutions for one leg ================= */
 bool compute_two_horn_angle_solutions_for_leg(
   uint8_t leg_index,
   const float R[3][3],
   const float T_mm[3],
   float &theta1_deg, float &theta2_deg,
   // debug/inspection:
   float s_vec_mm[3], float &scalar_a, float &scalar_b, float &scalar_gamma,
   float &radius_r, float &ratio_x, float &phi_deg, float &d_deg
 ){
   // s = R*A + T − P
   float s[3];
   for (uint8_t j=0; j<3; ++j){
     s[j] = R[j][0]*platform_anchor_mm[leg_index][0]
          + R[j][1]*platform_anchor_mm[leg_index][1]
          + R[j][2]*platform_anchor_mm[leg_index][2]
          + T_mm[j]
          - bottom_anchor_mm[leg_index][j];
   }
   s_vec_mm[0]=s[0]; s_vec_mm[1]=s[1]; s_vec_mm[2]=s[2];
 
   // a cosθ + b sinθ = gamma  (u = ẑ, H.z = 0 ⇒ c=0)
   scalar_a = s[0]*horn_base_vector_mm[leg_index][0]
            + s[1]*horn_base_vector_mm[leg_index][1];
 
   // Keep RH "ẑ × H" term here (θ positive CCW); if you want θ positive CW, swap signs per earlier note.
   scalar_b = s[0]*(-horn_base_vector_mm[leg_index][1])
            + s[1]*( horn_base_vector_mm[leg_index][0]);
 
   const float s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
   scalar_gamma = 0.5f*(s2 + horn_vec_len_sq_mm2 - link_length_mm*link_length_mm);
 
   radius_r = sqrtf(scalar_a*scalar_a + scalar_b*scalar_b);
   if (radius_r < 1e-9f) return false;
 
   ratio_x = scalar_gamma / radius_r;                 // may exceed [-1,1] slightly
   const float x_clamped = clampf(ratio_x, -1.f, 1.f);
 
   const float phi_rad = atan2f(scalar_b, scalar_a);
   const float d_rad   = acosf(x_clamped);
 
   phi_deg   = phi_rad * RAD_TO_DEG;
   d_deg     = d_rad   * RAD_TO_DEG;
   theta1_deg = (phi_rad + d_rad) * RAD_TO_DEG;
   theta2_deg = (phi_rad - d_rad) * RAD_TO_DEG;
   return true;
 }
 
 /* Map command (−90..+90) to Servo.write(0..180) with inversion & trim */
 int map_command_to_servo_write(uint8_t leg_index, float command_deg){
   const float clamped = clampf(command_deg, -COMMAND_LIMIT_DEG, +COMMAND_LIMIT_DEG);
   float write_deg = 90.f + clamped + per_leg_trim_degrees[leg_index]; // neutral → 90°
   int w = (int)roundf(write_deg);
   if (leg_is_inverted[leg_index]) w = 180 - w;
   if (w < 0) w = 0; else if (w > 180) w = 180;
   return w;
 }
 
 void write_leg(uint8_t leg_index, float command_deg){
   int w = map_command_to_servo_write(leg_index, command_deg);
   leg_servo[leg_index].write(w);
 }
 
 /* ================= Neutral calibration ================= */
 void calibrate_neutral_pose(){
   const float R_I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
   const float T_0[3]    = {0,0,0};
 
   for (uint8_t i=0; i<NUM_LEGS; ++i){
     float s[3], a,b,gamma,r,x,phi,d, t1,t2;
     if (!compute_two_horn_angle_solutions_for_leg(i, R_I, T_0,
           t1,t2, s,a,b,gamma,r,x,phi,d))
     { t1=0; t2=180; }
 
     auto fold180 = [](float th){
       while (th >= 180.f) th -= 360.f;
       while (th <  -180.f) th += 360.f;
       return th;
     };
     t1 = fold180(t1); t2 = fold180(t2);
     neutral_horn_angle_deg[i] = (fabsf(t1) <= fabsf(t2)) ? t1 : t2;
     last_command_deg[i] = 0.f;
   }
 
   Serial.print("# neutral_horn_angle_deg: ");
   for (uint8_t i=0;i<NUM_LEGS;i++){ Serial.print(neutral_horn_angle_deg[i],2); Serial.print(i==NUM_LEGS-1?'\n':' '); }
 }
 
 /* ================= Solve & (optionally) move ALL legs ================= */
 void solve_pose_for_all_legs(const float R[3][3], const float T_mm[3],
                              bool do_move, bool verbose){
   for (uint8_t i=0; i<NUM_LEGS; ++i){
     float s[3], a,b,gamma,r,x,phi,d, t1,t2;
     bool ok = compute_two_horn_angle_solutions_for_leg(i, R, T_mm,
                 t1,t2, s,a,b,gamma,r,x,phi,d);
     if (!ok){
       if (verbose) { Serial.print("leg "); Serial.print(i); Serial.println(": IK FAIL"); }
       continue;
     }
 
     // commands relative to neutral, with continuity
     float cmd1 = unwrap_deg_near(t1 - neutral_horn_angle_deg[i], last_command_deg[i]);
     float cmd2 = unwrap_deg_near(t2 - neutral_horn_angle_deg[i], last_command_deg[i]);
 
     // branch select: neutral bias + small hysteresis; else continuity
     float cmd_pick;
     if (fabsf(cmd1) + BRANCH_HYSTERESIS_DEG < fabsf(cmd2))       cmd_pick = cmd1;
     else if (fabsf(cmd2) + BRANCH_HYSTERESIS_DEG < fabsf(cmd1))  cmd_pick = cmd2;
     else cmd_pick = (fabsf(cmd1 - last_command_deg[i]) <= fabsf(cmd2 - last_command_deg[i])) ? cmd1 : cmd2;
 
     float cmd_clamped = cmd_pick; //clampf(cmd_pick, -COMMAND_LIMIT_DEG, +COMMAND_LIMIT_DEG);
 
     if (verbose){
       Serial.print("leg "); Serial.print(i);
       Serial.print(": cmd="); Serial.print(cmd_clamped,2);
       Serial.print("  (t1="); Serial.print(t1,2);
       Serial.print(", t2="); Serial.print(t2,2);
       Serial.print(", t0="); Serial.print(neutral_horn_angle_deg[i],2);
       Serial.print(", x=");  Serial.print(x,3);
       Serial.println(")");
     }
 
     if (do_move){
       write_leg(i, cmd_clamped);
       last_command_deg[i] = cmd_clamped;
     }
   }
 }
 
 /* ================= Serial command handler ================= */
 void handle_serial(){
   String line = Serial.readStringUntil('\n'); line.trim();
   if (line.length()==0) return;
 
   if (line.equalsIgnoreCase("CENTER")){
     for (uint8_t i=0;i<NUM_LEGS;i++){ write_leg(i, 0.f); last_command_deg[i]=0.f; }
     Serial.println("# centered");
     return;
   }
   if (line.startsWith("INV ")){
     int i = line.substring(4).toInt();
     if (i>=0 && i<NUM_LEGS){
       leg_is_inverted[i] = !leg_is_inverted[i];
       Serial.print("# invert["); Serial.print(i); Serial.print("] = ");
       Serial.println(leg_is_inverted[i] ? "true":"false");
     }
     return;
   }
 
   // Tokenize: head + six floats
   String head; int sp1 = line.indexOf(' ');
   if (sp1>0) head = line.substring(0, sp1);
   auto need6 = [](){ Serial.println("# need 6 numbers: roll pitch yaw x y z"); };
   auto parse6 = [&](int from, float out[6])->bool{
     int k=0;
     while (k<6){
       int sp = line.indexOf(' ', from);
       String tok = (sp<0) ? line.substring(from) : line.substring(from, sp);
       tok.trim(); if(tok.length()==0) return false;
       out[k++] = tok.toFloat();
       if (sp<0) break; from = sp+1;
     }
     return k==6;
   };
 
   if (head.equalsIgnoreCase("POSE") || head.equalsIgnoreCase("IKALL")
       || head.equalsIgnoreCase("IK") || head.equalsIgnoreCase("MOVE")){
     float v[6]; if (!parse6(sp1+1, v)) { need6(); return; }
 
     const float roll_rad  = v[0] * DEG_TO_RAD;
     const float pitch_rad = v[1] * DEG_TO_RAD;
     const float yaw_rad   = v[2] * DEG_TO_RAD;
 
     // NOTE: This preserves your last test mapping: {-x, +y, -z}.
     // If you want pure LH everywhere, change to: { v[3], v[4], v[5] }.
     const float T_mm[3]   = { -v[3],  v[4],  -v[5] };
 
     float R[3][3]; build_rotation_matrix_zyx(roll_rad, pitch_rad, yaw_rad, R);
 
     if (head.equalsIgnoreCase("POSE")){
       solve_pose_for_all_legs(R, T_mm, /*move=*/true,  /*verbose=*/true);
       return;
     }
     if (head.equalsIgnoreCase("IKALL")){
       solve_pose_for_all_legs(R, T_mm, /*move=*/false, /*verbose=*/true);
       return;
     }
     // Single-leg modes kept for focused debugging:
     if (head.equalsIgnoreCase("IK") || head.equalsIgnoreCase("MOVE")){
       // Expect an index before the six floats → reparse with index
       // Format: "IK i roll pitch yaw x y z" / "MOVE i ..."
       int sp2 = line.indexOf(' ', sp1+1);
       if (sp2<0){ Serial.println("# IK/MOVE: need index + 6 numbers"); return; }
       int leg_index = line.substring(sp1+1, sp2).toInt();
       float vv[6]; if (!parse6(sp2+1, vv)) { need6(); return; }
 
       const float rr = vv[0] * DEG_TO_RAD;
       const float pp = vv[1] * DEG_TO_RAD;
       const float yy = vv[2] * DEG_TO_RAD;
       const float TT[3] = { -vv[3], vv[4], -vv[5] };
       float RR[3][3]; build_rotation_matrix_zyx(rr, pp, yy, RR);
 
       float s[3], a,b,gamma,r,x,phi,d, t1,t2;
       if (!compute_two_horn_angle_solutions_for_leg(leg_index, RR, TT, t1,t2, s,a,b,gamma,r,x,phi,d)){
         Serial.println("# IK FAIL");
         return;
       }
       float cmd1 = unwrap_deg_near(t1 - neutral_horn_angle_deg[leg_index], last_command_deg[leg_index]);
       float cmd2 = unwrap_deg_near(t2 - neutral_horn_angle_deg[leg_index], last_command_deg[leg_index]);
 
       float pick;
       if (fabsf(cmd1) + BRANCH_HYSTERESIS_DEG < fabsf(cmd2))       pick = cmd1;
       else if (fabsf(cmd2) + BRANCH_HYSTERESIS_DEG < fabsf(cmd1))  pick = cmd2;
       else pick = (fabsf(cmd1 - last_command_deg[leg_index]) <= fabsf(cmd2 - last_command_deg[leg_index])) ? cmd1 : cmd2;
 
       float pickC = clampf(pick, -COMMAND_LIMIT_DEG, +COMMAND_LIMIT_DEG);
       int w = map_command_to_servo_write(leg_index, pickC);
 
       Serial.print("LEG "); Serial.println(leg_index);
       Serial.print("s = ["); Serial.print(s[0],3); Serial.print(", "); Serial.print(s[1],3); Serial.print(", "); Serial.print(s[2],3); Serial.println("]");
       Serial.print("a="); Serial.print(a,6); Serial.print(" b="); Serial.print(b,6);
       Serial.print(" gamma="); Serial.print(gamma,6); Serial.print(" r="); Serial.print(r,6);
       Serial.print(" x="); Serial.println(x,6);
       Serial.print("t1="); Serial.print(t1,3); Serial.print(" t2="); Serial.println(t2,3);
       Serial.print("t0="); Serial.print(neutral_horn_angle_deg[leg_index],3);
       Serial.print(" cmd1="); Serial.print(cmd1,3); Serial.print(" cmd2="); Serial.println(cmd2,3);
       Serial.print("pick="); Serial.print(pick,3); Serial.print(" clamped="); Serial.println(pickC,3);
       Serial.print("-> writeDeg="); Serial.println(w);
 
       if (head.equalsIgnoreCase("MOVE")){
         write_leg(leg_index, pickC);
         last_command_deg[leg_index] = pickC;
       }
       return;
     }
   }
 
   Serial.println("# Unknown. Use: POSE | IKALL | IK i ... | MOVE i ... | CENTER | INV i");
 }
 
 /* ================= setup / loop ================= */
 void setup(){
   Serial.begin(115200);
 
   ESP32PWM::allocateTimer(0);
   ESP32PWM::allocateTimer(1);
   ESP32PWM::allocateTimer(2);
   ESP32PWM::allocateTimer(3);
 
   for (uint8_t i=0;i<NUM_LEGS;i++){
     leg_servo[i].setPeriodHertz(50);
     leg_servo[i].attach(leg_servo_pins[i], PWM_PULSE_MIN_US, PWM_PULSE_MAX_US);
     leg_servo[i].write(90); // neutral hold
   }
 
   calibrate_neutral_pose();
 
   Serial.println("Full IK ready.");
   Serial.println("POSE r p y  x y z   -> move all");
   Serial.println("IKALL r p y  x y z  -> print all");
   Serial.println("IK i r p y  x y z   | MOVE i r p y  x y z");
   Serial.println("CENTER | INV i");
 }
 
 void loop(){
   if (Serial.available()) handle_serial();
 }
 