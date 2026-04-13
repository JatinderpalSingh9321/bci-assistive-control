# BCI Assistive Control — Implementation Plan
## Upside Down Labs Kit + 1D-CNN + Flask + Real-Time Dashboard
### Group No. 7 | Last Semester Major Project

---

## Executive Summary

**Objective:** Build a complete, working non-invasive BCI system using the Upside Down Labs bio-signal acquisition kit, a 1D-CNN deep learning pipeline, and a real-time web interface — deployable for paralyzed patient assistive control.

| Component | Technology |
|-----------|-----------|
| **Hardware** | Upside Down Labs BioAmp EXG Pill / EEG kit |
| **Signal Processing** | Python, MNE, SciPy |
| **Deep Learning** | TensorFlow / Keras (1D-CNN) |
| **Backend API** | Flask (REST) |
| **Frontend UI** | HTML/JS real-time dashboard |
| **Timeline** | 8 weeks |
| **Expected Accuracy** | 81–87% (LOSO cross-validation) |

---

## Part 1: Understanding the Upside Down Labs Kit

### 1.1 What is Upside Down Labs?

Upside Down Labs is an open-hardware neurotechnology company that produces affordable biosignal acquisition tools designed for research, education, and DIY neurotech. Their kits are fully open-source (hardware + firmware) and interface directly with Python via USB/serial.

**Key Advantages for This Project:**
- Cost: ₹3,000–₹5,000 (vs. ₹15,00,000+ for clinical EEG systems)
- Open hardware: full schematics, no proprietary lock-in
- Python-compatible: reads directly via `pyserial`
- Active community: forums, example code, and tutorials available

**Relevant Kit Specifications:**
- Analog front-end: instrumentation amplifier optimized for μV-range biosignals
- Sampling rate: configurable (typically 250–500 Hz via Arduino/Raspberry Pi)
- Electrode types: gel or dry electrodes (gel recommended for EEG)
- Connectivity: USB serial to laptop
- Compatible boards: Arduino Uno/Nano, Raspberry Pi, ESP32

### 1.2 Kit Contents Checklist

```
Upside Down Labs EEG Setup:
├── BioAmp EXG Pill (main amplifier board)
├── Gel electrodes × 8 (reusable, disposable gel pads)
├── Reference electrode (clip for earlobe)
├── Ground electrode (clip for other earlobe)
├── Snap-to-lead wires
├── Arduino Uno or Nano (for ADC + USB serial)
├── USB-A to USB-B cable
├── Electrode gel (Ten20 or similar)
└── Electrode cap or headband (for consistent placement)
```

### 1.3 Electrode Placement for Motor Imagery

Follow the 10-20 international system, focusing on channels most informative for motor imagery:

```
Central (Primary Motor Cortex):
  IN+ (Left side):   C3
  IN- (Right side):  C4
  
Reference/Ground:    Fpz (forehead midline) or Earlobe

Motor Imagery Pattern:
  LEFT hand  → ERD at C3
  RIGHT hand → ERD at C4
```

---

## Part 2: Hardware Setup

### 2.1 Wiring the BioAmp EXG Pill

```
BioAmp EXG Pill → Arduino Uno:
  VCC  →  5V (or 3.3V depending on kit version)
  GND  →  GND
  OUT  →  A0 (analog input)

Electrodes → BioAmp:
  IN+  →  EEG electrode (scalp)
  IN-  →  Reference (earlobe)
  GND  →  Ground (other earlobe or Fpz)
```

### 2.2 Arduino Firmware (Data Streaming)

Flash the following sketch to stream EEG at 250 Hz over serial:

```cpp
// Arduino sketch: stream 8-channel EEG at 250 Hz
// If using multiplexer for 8 channels, extend accordingly

const int SAMPLE_RATE = 250;  // Hz
const int DELAY_US = 1000000 / SAMPLE_RATE;  // microseconds

void setup() {
  Serial.begin(115200);
  analogReference(EXTERNAL);  // use 3.3V AREF for better resolution
}

void loop() {
  unsigned long t = micros();
  int val = analogRead(A0);  // 0–1023 (10-bit ADC)
  Serial.println(val);
  while (micros() - t < DELAY_US);  // precise timing
}
```

### 2.3 Software Installation

```bash
# Hardware interface
pip install pyserial

# Signal processing
pip install numpy scipy mne

# Machine learning
pip install scikit-learn tensorflow

# Visualization
pip install matplotlib seaborn

# API
pip install flask flask-cors

# Verify
python -c "import serial, mne, tensorflow; print('All dependencies OK')"
```

### 2.4 Hardware Calibration

**Step 1: Verify Serial Connection**
```python
import serial
import serial.tools.list_ports

# List available ports
ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p.device, p.description)

# Connect (adjust port as needed: COM3 on Windows, /dev/ttyUSB0 on Linux)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
print("Connected:", ser.isOpen())
```

**Step 2: Check Signal Quality**
```python
import numpy as np
import time

def check_signal_quality(ser, duration=10, fs=250):
    samples = []
    start = time.time()
    while time.time() - start < duration:
        line = ser.readline().decode('utf-8').strip()
        try:
            samples.append(float(line))
        except ValueError:
            continue

    data = np.array(samples)
    snr = 10 * np.log10(np.mean(data**2) / np.var(data))
    print(f"Samples collected: {len(data)}")
    print(f"Mean amplitude:    {np.mean(np.abs(data)):.2f} ADC units")
    print(f"SNR estimate:      {snr:.1f} dB")
    print(f"Status:            {'✓ Good' if snr > 8 else '✗ Poor — check electrode contact'}")
    return data

data = check_signal_quality(ser)
```

---

## Part 3: Data Collection

### 3.1 Motor Imagery Experiment Script

```python
# src/experiment.py — runs the data collection paradigm

import pygame, serial, numpy as np, json, time
from pathlib import Path

def run_experiment(subject_id, n_trials=80, fs=250, cue_duration=4.0):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    font = pygame.font.Font(None, 72)
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

    Path(f'data/raw/subject_{subject_id:03d}').mkdir(parents=True, exist_ok=True)
    trials, labels = [], []

    for trial_n in range(n_trials):
        label = trial_n % 2  # 0 = LEFT, 1 = RIGHT (balanced)

        # READY screen
        screen.fill((30, 30, 30))
        screen.blit(font.render("Ready...", True, (200, 200, 200)), (300, 270))
        pygame.display.flip()
        time.sleep(2.0)

        # CUE
        cue_text = "← LEFT" if label == 0 else "RIGHT →"
        screen.fill((30, 30, 30))
        screen.blit(font.render(cue_text, True, (255, 255, 100)), (260, 270))
        pygame.display.flip()
        time.sleep(1.0)

        # MOTOR IMAGERY — collect EEG
        screen.fill((30, 30, 30))
        screen.blit(font.render("IMAGINE", True, (100, 255, 100)), (280, 270))
        pygame.display.flip()

        epoch_samples = []
        start = time.time()
        while time.time() - start < cue_duration:
            line = ser.readline().decode().strip()
            try:
                epoch_samples.append(float(line))
            except ValueError:
                pass

        trials.append(epoch_samples)
        labels.append(label)
        print(f"Trial {trial_n+1}/{n_trials} | {'LEFT' if label==0 else 'RIGHT'} | {len(epoch_samples)} samples")

        # REST
        screen.fill((30, 30, 30))
        pygame.display.flip()
        time.sleep(2.0)

    # Save
    np.savez(f'data/raw/subject_{subject_id:03d}/session_01.npz',
             data=np.array(trials, dtype=object), labels=np.array(labels))
    print(f"Subject {subject_id} data saved. {n_trials} trials total.")
    pygame.quit()

if __name__ == '__main__':
    run_experiment(subject_id=1)
```

### 3.2 Data Storage Format

```
data/raw/subject_001/
├── session_01.npz          # numpy arrays: data (80 × ~1000), labels (80,)
└── session_01_meta.json    # metadata (date, SNR, electrode notes)

session_01_meta.json:
{
  "subject_id": "001",
  "date": "2025-09-15",
  "fs": 250,
  "n_trials": 80,
  "electrode_placement": "C3,C4,Fpz",
  "signal_snr_db": 14.2,
  "notes": "Good signal quality, subject relaxed"
}
```

---

## Part 4: Preprocessing Pipeline

### 4.1 Complete Preprocessing Module

```python
# src/preprocessing.py

import numpy as np
from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler

class EEGPreprocessor:
    def __init__(self, fs=250, lowcut=8.0, highcut=30.0, target_fs=125):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.target_fs = target_fs

    def bandpass_filter(self, data):
        nyq = self.fs / 2
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = sp_signal.butter(4, [low, high], btype='band')
        return sp_signal.filtfilt(b, a, data)

    def notch_filter(self, data, freq=50.0):
        b, a = sp_signal.iirnotch(freq, Q=30, fs=self.fs)
        return sp_signal.filtfilt(b, a, data)

    def downsample(self, data):
        factor = self.fs // self.target_fs
        return data[::factor]

    def baseline_correct(self, epoch, baseline_samples=62):
        # baseline_samples = 0.5s at 125 Hz
        baseline_mean = np.mean(epoch[:baseline_samples])
        return epoch - baseline_mean

    def reject_artifact(self, epoch, threshold_uv=150):
        # Reject epochs with peak-to-peak > threshold
        ptp = np.ptp(epoch)
        return ptp < threshold_uv

    def preprocess_epoch(self, raw_epoch):
        x = self.notch_filter(raw_epoch)
        x = self.bandpass_filter(x)
        x = self.downsample(x)
        x = self.baseline_correct(x)
        return x

    def preprocess_all(self, raw_trials, labels):
        clean_epochs, clean_labels = [], []
        for i, (trial, label) in enumerate(zip(raw_trials, labels)):
            epoch = self.preprocess_epoch(trial)
            if self.reject_artifact(epoch):
                clean_epochs.append(epoch)
                clean_labels.append(label)
        kept = len(clean_epochs) / len(raw_trials)
        print(f"Trials retained: {len(clean_epochs)}/{len(raw_trials)} ({kept:.0%})")
        return np.array(clean_epochs), np.array(clean_labels)
```

---

## Part 5: Model Implementation

### 5.1 SVM Baseline

```python
# src/models.py — SVM with band power features

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import signal as sp_signal

def extract_band_power(epochs, fs=125):
    features = []
    for epoch in epochs:
        freqs, psd = sp_signal.welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
        alpha = np.mean(psd[(freqs >= 8)  & (freqs <= 12)])
        beta  = np.mean(psd[(freqs > 12) & (freqs <= 30)])
        entropy = -np.sum(psd * np.log(psd + 1e-10))
        features.append([alpha, beta, entropy, np.var(epoch), np.mean(np.abs(epoch))])
    return np.array(features)

def build_svm():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
    ])
```

### 5.2 1D-CNN Main Model

```python
# src/models.py — 1D-CNN

import tensorflow as tf

def build_1d_cnn(n_timepoints=500, n_classes=2):
    inputs = tf.keras.Input(shape=(n_timepoints, 1))

    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

### 5.3 LOSO Cross-Validation

```python
# src/evaluation.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def loso_evaluate(X, y, subject_ids, model_builder, model_type='cnn'):
    unique_subjects = np.unique(subject_ids)
    all_preds, all_true = [], []

    for sub in unique_subjects:
        test_mask  = subject_ids == sub
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        model = model_builder()

        if model_type == 'cnn':
            X_train_in = X_train[..., np.newaxis]
            X_test_in  = X_test[..., np.newaxis]
            y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
            model.fit(X_train_in, y_train_cat, epochs=50, batch_size=32,
                      verbose=0, validation_split=0.1,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
            preds = np.argmax(model.predict(X_test_in, verbose=0), axis=1)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        all_preds.extend(preds)
        all_true.extend(y_test)
        acc = np.mean(preds == y_test)
        print(f"  Subject {sub}: {acc:.1%}")

    print("\n" + "="*40)
    print(classification_report(all_true, all_preds, target_names=['LEFT', 'RIGHT']))
    return np.array(all_preds), np.array(all_true)
```

---

## Part 6: Flask REST API

```python
# api/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, pickle, tensorflow as tf, time

app = Flask(__name__)
CORS(app)

# Load models at startup
cnn_model = tf.keras.models.load_model('data/models/cnn_model.h5')
svm_model  = pickle.load(open('data/models/svm_model.pkl', 'rb'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['cnn', 'svm']})

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    data  = request.json
    epoch = np.array(data['epoch'], dtype=np.float32)  # (n_timepoints,)

    # CNN prediction
    epoch_input = epoch[np.newaxis, :, np.newaxis]  # (1, T, 1)
    probs = cnn_model.predict(epoch_input, verbose=0)[0]
    label = ['LEFT', 'RIGHT'][int(np.argmax(probs))]
    conf  = float(np.max(probs))

    latency_ms = (time.time() - start) * 1000
    return jsonify({
        'prediction':    label,
        'confidence':    round(conf, 3),
        'probabilities': {'LEFT': round(float(probs[0]), 3),
                          'RIGHT': round(float(probs[1]), 3)},
        'latency_ms':    round(latency_ms, 1)
    })

@app.route('/calibrate', methods=['POST'])
def calibrate():
    data   = request.json
    epochs = np.array(data['epochs'])
    labels = np.array(data['labels'])
    # Fine-tune last layer only
    for layer in cnn_model.layers[:-2]:
        layer.trainable = False
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(epochs[..., np.newaxis],
                  tf.keras.utils.to_categorical(labels, 2),
                  epochs=5, batch_size=8, verbose=0)
    return jsonify({'status': 'calibrated', 'n_samples': len(labels)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## Part 7: Real-Time Dashboard

```html
<!-- frontend/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BCI Assistive Control — Group 7</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; text-align: center; }
    h1 { color: #e94560; margin-top: 2rem; }
    #prediction { font-size: 5rem; margin: 2rem; color: #0f3460; background: #16213e;
                  padding: 1rem 3rem; border-radius: 12px; display: inline-block; }
    .LEFT  { color: #00d2ff !important; }
    .RIGHT { color: #f5a623 !important; }
    #confidence { font-size: 1.5rem; color: #aaa; }
    #status { font-size: 0.9rem; color: #666; margin-top: 1rem; }
    canvas { border: 1px solid #333; border-radius: 8px; margin: 1rem; }
  </style>
</head>
<body>
  <h1>🧠 BCI Assistive Control</h1>
  <div id="prediction">—</div>
  <div id="confidence">Confidence: —</div>
  <canvas id="eegCanvas" width="700" height="120"></canvas>
  <div id="status">Waiting for signal...</div>

  <script>
    const canvas = document.getElementById('eegCanvas');
    const ctx = canvas.getContext('2d');
    const buffer = [];

    async function sendEpoch(epoch) {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epoch })
      });
      const data = await res.json();
      const el = document.getElementById('prediction');
      el.textContent = data.prediction;
      el.className = data.prediction;
      document.getElementById('confidence').textContent =
        `Confidence: ${(data.confidence * 100).toFixed(1)}%  |  Latency: ${data.latency_ms} ms`;
    }

    function drawEEG(sample) {
      buffer.push(sample);
      if (buffer.length > 700) buffer.shift();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#00d2ff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      buffer.forEach((v, i) => {
        const x = i;
        const y = canvas.height / 2 - (v / 512) * 50;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Simulated streaming (replace with actual WebSocket or SSE from Flask)
    setInterval(() => {
      const sample = Math.sin(Date.now() / 100) * 200 + Math.random() * 50;
      drawEEG(sample);
    }, 4);
  </script>
</body>
</html>
```

---

## Part 8: Troubleshooting

### Common Issues & Fixes

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| No serial data | Wrong COM port | Run `ls /dev/tty*` (Linux) or Device Manager (Windows) |
| Flat signal | Loose electrode or no gel | Re-apply conductive gel, check clip connections |
| Very noisy signal (60 Hz hum) | No notch filter or bad grounding | Add 50 Hz notch filter, check ground electrode |
| Low accuracy (< 70%) | Poor signal quality or insufficient training data | Improve impedance, add subjects, tune frequency band |
| API 500 error | Model not loaded or wrong input shape | Check `data/models/` path, verify epoch dimensions |
| Real-time lag > 300 ms | CNN inference on CPU is slow | Switch to SVM for live demo, use CNN for offline results |
| Imbalanced classes | Unequal LEFT/RIGHT trials | Ensure 40 LEFT + 40 RIGHT per subject |

### Performance Optimization

```python
# Faster inference: pre-allocate arrays
epoch_buffer = np.zeros((500, 1), dtype=np.float32)  # reuse, don't reallocate

# Faster SVM (use for live demo, CNN for thesis accuracy)
from sklearn.svm import LinearSVC  # 5× faster than RBF kernel

# Multi-threading: separate collection and inference threads
import threading, queue
inference_queue = queue.Queue(maxsize=1)

def inference_worker():
    while True:
        epoch = inference_queue.get()
        pred  = model.predict(epoch)
        update_ui(pred)

threading.Thread(target=inference_worker, daemon=True).start()
```

---

## Part 9: Pre-Defence Checklist

### One Week Before
- [ ] All 480 trials collected and backed up (local + cloud)
- [ ] Preprocessing pipeline tested on all subjects
- [ ] SVM and CNN models trained and saved to `data/models/`
- [ ] LOSO results table finalized
- [ ] All thesis figures at 300 DPI
- [ ] GitHub repository public with README

### Day Before
- [ ] Flask API starts cleanly on presentation laptop
- [ ] Dashboard loads and connects to API
- [ ] Hardware tested (signal quality confirmed)
- [ ] Backup video of successful live demo recorded
- [ ] Presentation slides finalized and timed (30–45 min)

### 30 Minutes Before
- [ ] Kit charged and electrode gel fresh
- [ ] Laptop plugged into power
- [ ] Flask server running: `python api/app.py`
- [ ] Dashboard open in browser
- [ ] Projector/screen connected and mirroring
- [ ] Backup video loaded and ready

### Demo Script
```
1. "Let me show you the Upside Down Labs EEG kit" → hold up hardware
2. "Signal quality looks good" → show dashboard SNR reading
3. "I'm imagining squeezing my LEFT fist now..." → close eyes, 4 seconds
4. System predicts: LEFT (78% confidence) → audience sees it live
5. "Now RIGHT..." → system predicts RIGHT
6. "A paralyzed patient could use these commands to control a cursor,
   robotic arm, or communication device — for under ₹5,000 total cost"
```

---

## Latency Budget Summary

| Stage | Target |
|-------|--------|
| EEG epoch collection (4 sec) | 4,000 ms (required) |
| Preprocessing (filter + downsample) | 50–80 ms |
| CNN inference | 40–60 ms |
| API response + UI update | 10–20 ms |
| **Total end-to-end latency** | **< 200 ms** ✓ |

---

*Built with open hardware, open software, and real clinical impact. 🧠*
