/*
 * BCI Assistive Control — EEG Streaming Firmware
 * Group No. 7 | Upside Down Labs BioAmp EXG Pill
 *
 * Streams analog EEG data from BioAmp EXG Pill at 250 Hz
 * over USB serial at 115200 baud.
 *
 * Wiring:
 *   BioAmp EXG Pill VCC  → Arduino 5V (or 3.3V)
 *   BioAmp EXG Pill GND  → Arduino GND
 *   BioAmp EXG Pill OUT  → Arduino A0
 *   Electrode IN+         → Scalp electrode
 *   Electrode IN-         → Reference (earlobe)
 *   Electrode GND         → Ground (other earlobe / Fpz)
 *
 * Protocol:
 *   Each line: single integer (0–1023), 10-bit ADC value
 *   Timestamp maintained via busy-wait loop for precise 250 Hz
 */

const int EEG_PIN     = A0;      // Analog input pin
const int SAMPLE_RATE = 250;     // Sampling rate in Hz
const long DELAY_US   = 1000000L / SAMPLE_RATE;  // Microseconds per sample
const long BAUD_RATE  = 115200;  // Serial baud rate

// Optional: LED blink to indicate streaming is active
const int LED_PIN     = 13;
unsigned long sampleCount = 0;

void setup() {
  Serial.begin(BAUD_RATE);

  // Default ADC resolution is 10-bit (0-1023) to remain compatible with data processing scripts
  // ARDUINO R4 OPTIMIZATION: Do not use analogReference(EXTERNAL) here
  analogReadResolution(10);
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Wait for serial connection
  while (!Serial) {
    ; // Wait for USB serial port to connect (needed for Leonardo/Micro)
  }

  // Send header to identify firmware
  Serial.println("# BCI EEG Stream v1.0");
  Serial.println("# Sample Rate: 250 Hz");
  Serial.println("# Format: <adc_value>");
  Serial.println("# START");
}

void loop() {
  unsigned long t_start = micros();

  // Read 10-bit ADC value (0–1023)
  int val = analogRead(EEG_PIN);

  // Send over serial
  Serial.println(val);

  // Toggle LED every second to show activity
  sampleCount++;
  if (sampleCount % SAMPLE_RATE == 0) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  // Busy-wait for precise timing (maintains exact 250 Hz)
  while (micros() - t_start < DELAY_US) {
    // Tight loop — ensures consistent sample rate
  }
}
