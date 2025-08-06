#include <ESP32Servo.h>

int pos = 0;

Servo servo0;
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;

Servo servo6;
Servo servo7;
Servo servo8;
Servo servo9;


void setup() {
    Serial.begin(115200);

    // analogWriteFreq(500);

    servo0.attach(4);
    servo1.attach(13);
    servo2.attach(16);
    servo3.attach(17);
    servo4.attach(18);
    servo5.attach(19);

    servo6.attach(22);
    servo7.attach(23);
    servo8.attach(25);
    servo9.attach(26);

    Serial.println("boot");
}

void loop() {
  // servo0.write(0);
  for (pos = 30 * 2; pos <= 60 * 2; pos += 1) { // goes from 0 degrees to 180 degrees in steps of 1 degree
    // analogWrite(4, pos);
    servo0.write(pos);
		servo1.write(pos);    // tell servo to go to position in variable 'pos'
    servo2.write(pos);
    servo3.write(pos);
    servo4.write(pos);
    servo5.write(pos);

    servo6.write(pos);
    servo7.write(pos);
    servo8.write(pos);
    servo9.write(pos);
		delay(15);             // waits 15ms for the servo to reach the position
    Serial.println(pos);
	}
  delay(500);
	for (pos = 60 * 2; pos >= 30 * 2; pos -= 1) { // goes from 180 degrees to 0 degrees
    // analogWrite(4, pos);
    servo0.write(pos);
		servo1.write(pos);    // tell servo to go to position in variable 'pos'
    servo2.write(pos);
    servo3.write(pos);
    servo4.write(pos);
    servo5.write(pos);

    servo6.write(pos);
    servo7.write(pos);
    servo8.write(pos);
    servo9.write(pos);
		delay(15);             // waits 15ms for the servo to reach the position
    Serial.println(pos);
	}
  delay(500);
}