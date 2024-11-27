#include <util/atomic.h>

#define ENCA 2      // Encoder pinA
#define ENCB 3      // Encoder pinB
#define PWM 11      // Motor PWM pin
#define IN2 6       // Motor controller pin2
#define IN1 7       // Motor controller pin1

volatile int posi = 0; // Position variable
long prevT = 0;

// PID constants for initial test
float kp = 75;
float ki = 3.75;  
float kd = 2.5;

// PID variables
float integral = 0;
float prevError = 0;

// Windup limit for the integral term
float integralLimit = 75; // You can adjust this limit based on system testing

void setup() {
  Serial.begin(115200);

  // ENCODER
  pinMode(ENCA, INPUT);
  pinMode(ENCB, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA), readEncoder, RISING);

  // DC MOTOR
  pinMode(PWM, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
}

void loop() {
  // Set target position (adjust for your test)
  int targ = 2530;

  // Time difference (in seconds)
  long currT = micros();
  float deltaT = ((float)(currT - prevT)) / 1.0e6;
  prevT = currT;

  int pos = 0;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pos = posi;
  }

  // Error calculation
  int error = targ - pos;

  // PID Control calculations
  float P = kp * error;

  // Accumulate integral and apply windup limit
  integral += error * deltaT;

  // Apply the windup limit to the integral term
  if (integral > integralLimit) {
    integral = integralLimit;
  } else if (integral < -integralLimit) {
    integral = -integralLimit;
  }

  float I = ki * integral;
  float derivative = (error - prevError) / deltaT;
  float D = kd * derivative;
  prevError = error;

  // PID output
  float output = P + I + D;
  if (output > 255) output = 255;
  if (output < -255) output = -255;

  // Motor control
  int dir = (output < 0) ? 1 : -1;
  int pwr = abs(output);

  if (pos >= 2535) {
    dir = 0;
    pwr = 0;
  }

  setMotor(dir, pwr, PWM, IN1, IN2);

  // Log data for MATLAB (time, position, error, output)
  Serial.print(millis());
  Serial.print(",");
  Serial.print(pos);
  Serial.print(",");
  Serial.print(error);
  Serial.print(",");
  Serial.println(output);
}

void setMotor(int dir, int pwmVal, int pwm, int in1, int in2) {
  analogWrite(pwm, pwmVal);
  if (dir == 1) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else if (dir == -1) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}

void readEncoder() {
  int b = digitalRead(ENCB);
  if (b == HIGH) {
    posi++;
  } else {
    posi--;
  }
}