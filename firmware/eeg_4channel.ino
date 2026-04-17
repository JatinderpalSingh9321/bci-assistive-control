/*
 * BCI Assistive Control — 4-Channel EEG Streaming Firmware
 * Group No. 7 | Upside Down Labs BioAmp EXG Pill
 *
 * Streams 4 analog EEG channels at 250 Hz over USB serial.
 *
 * Channel Assignment:
 *   A0 → C3  (left motor cortex)   — referenced to left earlobe (A1)
 *   A1 → C4  (right motor cortex)  — referenced to right earlobe (A2)
 *   A2 → Fp1 (left forehead)       — referenced to left earlobe (A1)
 *   A3 → Fp2 (right forehead)      — referenced to right earlobe (A2)
 *
 * Electrode Wiring (per BioAmp EXG Pill):
 *   Each channel needs its own BioAmp EXG Pill (or multiplexed setup):
 *     Channel 1: EXG Pill OUT → A0, IN+ → C3, IN- → A1 earlobe, GND → shared
 *     Channel 2: EXG Pill OUT → A1, IN+ → C4, IN- → A2 earlobe, GND → shared
 *     Channel 3: EXG Pill OUT → A2, IN+ → Fp1, IN- → A1 earlobe, GND → shared
 *     Channel 4: EXG Pill OUT → A3, IN+ → Fp2, IN- → A2 earlobe, GND → shared
 *
 * Minimal Setup (2 BioAmp pills):
 *   Pill 1: C3-C4 bipolar → A0 (motor imagery)
 *   Pill 2: Fp1-Fp2 bipolar → A2 (EOG/blink)
 *   (Use A1, A3 for optional individual channels)
 *
 * Protocol:
 *   Line format: c3,c4,fp1,fp2  (comma-separated 10-bit ADC values)
 *   250 lines per second (250 Hz)
 *
 * Arduino R4 Minima — 6 analog inputs available, using 4.
 */

const int PIN_C3   = A0;
const int PIN_C4   = A1;
const int PIN_FP1  = A2;
const int PIN_FP2  = A3;

const int SAMPLE_RATE = 250;
const long DELAY_US   = 1000000L / SAMPLE_RATE;
const long BAUD_RATE  = 115200;

const int LED_PIN = 13;
unsigned long sampleCount = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(10);  // 10-bit (0-1023)
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  while (!Serial) {
    ;  // Wait for USB serial
  }

  // Header — identifies this as 4-channel firmware
  Serial.println("# BCI EEG Stream v2.0 (4-Channel)");
  Serial.println("# Sample Rate: 250 Hz");
  Serial.println("# Channels: C3,C4,Fp1,Fp2");
  Serial.println("# Format: <c3>,<c4>,<fp1>,<fp2>");
  Serial.println("# START");
}

void loop() {
  unsigned long t_start = micros();

  // Read all 4 channels as fast as possible
  int c3  = analogRead(PIN_C3);
  int c4  = analogRead(PIN_C4);
  int fp1 = analogRead(PIN_FP1);
  int fp2 = analogRead(PIN_FP2);

  // Send CSV line
  Serial.print(c3);
  Serial.print(',');
  Serial.print(c4);
  Serial.print(',');
  Serial.print(fp1);
  Serial.print(',');
  Serial.println(fp2);

  // LED heartbeat (toggle every second)
  sampleCount++;
  if (sampleCount % SAMPLE_RATE == 0) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  // Busy-wait for precise 250 Hz timing
  while (micros() - t_start < DELAY_US) {
    // Tight loop
  }
}
