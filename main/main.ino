// Minimal ESP32 + 6xServo hardware test (no IK)

#include <ESP32Servo.h>

constexpr uint8_t NUM_SERVOS = 6;
constexpr int PWM_PULSE_MIN_US = 1000;
constexpr int PWM_PULSE_MAX_US = 2000;

const uint8_t servo_pins[NUM_SERVOS] = {4, 13, 16, 17, 18, 19};
Servo servos[NUM_SERVOS];

// Helper: constrain to [0,180]
static inline int clampDeg(int d){
  if (d < 0) return 0;
  if (d > 180) return 180;
  return d;
}

// Commands:
//   CENTER                 -> all servos to 90°
//   SET i angle            -> set servo i (0..5) to angle (0..180)
//   SWEEP i                -> sweep servo i between 60° and 120° once
//   ALL angle              -> set all servos to angle (0..180)
void handleSerial(){
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("CENTER")){
    for (uint8_t i=0;i<NUM_SERVOS;i++) servos[i].write(90);
    Serial.println("# centered to 90");
    return;
  }

  if (line.startsWith("SET ")){
    int sp = line.indexOf(' ', 4);
    if (sp < 0){ Serial.println("# usage: SET i angle"); return; }
    int i = line.substring(4, sp).toInt();
    int ang = line.substring(sp+1).toInt();
    if (i < 0 || i >= NUM_SERVOS){ Serial.println("# bad index"); return; }
    ang = clampDeg(ang);
    servos[i].write(ang);
    Serial.print("# SET "); Serial.print(i); Serial.print(" -> "); Serial.println(ang);
    return;
  }

  if (line.startsWith("ALL ")){
    int ang = line.substring(4).toInt();
    ang = clampDeg(ang);
    for (uint8_t i=0;i<NUM_SERVOS;i++) servos[i].write(ang);
    Serial.print("# ALL -> "); Serial.println(ang);
      return;
    }

  if (line.startsWith("SWEEP ")){
    int i = line.substring(6).toInt();
    if (i < 0 || i >= NUM_SERVOS){ Serial.println("# bad index"); return; }
    for (int a=60; a<=120; a+=2){ servos[i].write(a); delay(15); }
    for (int a=120; a>=60; a-=2){ servos[i].write(a); delay(15); }
    Serial.print("# SWEEP done on "); Serial.println(i);
      return;
  }

  Serial.println("# Unknown. Use: CENTER | SET i angle | ALL angle | SWEEP i");
}

void setup(){
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (uint8_t i=0;i<NUM_SERVOS;i++){
    servos[i].setPeriodHertz(50);
    servos[i].attach(servo_pins[i], PWM_PULSE_MIN_US, PWM_PULSE_MAX_US);
    servos[i].write(90);
  }

  Serial.println("Servo HW test ready.");
  Serial.println("Commands: CENTER | SET i angle | ALL angle | SWEEP i");
}

void loop(){
  if (Serial.available()) handleSerial();
}
