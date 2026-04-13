# Non-Invasive BCI for Assistive Control — Project Plan
## Group No. 7 | Upside Down Labs + Deep Learning

---

## Executive Summary

**Goal:** Develop a complete, working non-invasive BCI system using EEG that decodes motor imagery intent and translates it into real-time assistive control commands for paralyzed patients.

| Aspect | Details |
|--------|---------|
| **Hardware** | Upside Down Labs Bio-Signal Acquisition Kit |
| **Backend** | Flask REST API + Python ML pipeline |
| **Frontend** | Real-time browser dashboard (Antigravity-compatible) |
| **ML Models** | SVM (baseline) + 1D-CNN (main) + LSTM (bonus) |
| **Timeline** | 8 weeks (1–2 months) |
| **Subjects** | 6 participants × 80 trials = 480 total |
| **Deliverable** | Working end-to-end BCI application + thesis document |

---

## Phase 1 — Week 1: Setup & Literature Review

### 1.1 Core Concepts to Master

**Neuroscience Fundamentals:**
- How EEG captures cortical electrical activity through scalp electrodes.
- The 10-20 electrode placement system and which positions correspond to motor cortex.
- Motor imagery neurophysiology: Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS).
- Alpha (8–12 Hz) and beta (12–30 Hz) band significance for motor imagery classification.

**BCI Paradigm Choice: Motor Imagery (MI)**

Motor imagery is the recommended paradigm for this project because:
- It is directly applicable to paralyzed patients (imagining limb movement is intuitive).
- Well-documented datasets and established benchmarks exist (PhysioNet, BCI Competition IV).
- The two-class task (LEFT vs. RIGHT hand) is achievable within the project timeline.
- Results are directly comparable to published literature.

**Recommended Papers (read in Week 1):**
1. Wolpaw et al. (2002) — BCI fundamentals (start here)
2. Lawhern et al. (2018) — EEGNet architecture
3. Roy et al. (2019) — Deep learning for EEG (Nature Reviews)
4. Ang et al. (2012) — Filter Bank CSP for feature extraction

### 1.2 Define Thesis Contribution

Choose one primary contribution angle:

| Angle | Description | Effort |
|-------|-------------|--------|
| Architecture comparison | Classical (SVM+CSP) vs 1D-CNN vs LSTM | Medium |
| Adaptive calibration | Model fine-tunes per user in real-time | High |
| Transfer learning | Pre-train on PhysioNet, fine-tune on kit data | High |
| Explainability | SHAP/Grad-CAM to show which brain regions matter | Medium |
| Real hardware novelty | Full pipeline on Upside Down Labs (no prior work) | Low–Medium |

**Recommended for this project:** Real hardware novelty + architecture comparison (achievable and impressive).

### 1.3 Development Environment Setup

```bash
# Core scientific stack
pip install numpy scipy pandas matplotlib seaborn

# EEG signal processing
pip install mne

# Machine learning
pip install scikit-learn xgboost

# Deep learning
pip install tensorflow  # or torch torchvision

# API backend
pip install flask flask-cors

# Upside Down Labs hardware interface
pip install pyserial  # for USB/serial communication
```

### 1.4 Repository Structure

```
bci-assistive-control/
├── data/
│   ├── raw/                    # Raw EEG recordings per subject
│   │   ├── subject_001/
│   │   │   ├── session_01_data.json
│   │   │   └── session_01.npz
│   │   └── subject_002/ ...
│   ├── preprocessed/
│   │   └── all_clean_epochs.npz
│   └── models/
│       ├── svm_model.pkl
│       ├── cnn_model.h5
│       └── results.csv
├── src/
│   ├── acquisition.py          # Upside Down Labs kit interface
│   ├── preprocessing.py        # Filtering, ICA, epoching
│   ├── feature_extraction.py   # Band power, CSP features
│   ├── models.py               # SVM, CNN, LSTM definitions
│   ├── evaluation.py           # LOSO CV, metrics, plots
│   └── utils.py                # Helpers, config
├── api/
│   ├── app.py                  # Flask REST API
│   └── routes.py
├── frontend/
│   └── dashboard.html          # Real-time UI
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── results/                    # Figures, confusion matrices, tables
├── README.md
├── requirements.txt
└── thesis_draft.md
```

---

## Phase 2 — Week 1–2: Hardware Setup & Data Acquisition

### 2.1 Upside Down Labs Kit Setup

**Electrode Placement (10-20 System):**

```
Primary Motor Cortex: C3 (Left) and C4 (Right)
Reference/Ground:     Fpz (Forehead) or Earlobe

Motor Imagery Pattern:
  LEFT hand imagery  → C3 shows ERD
  RIGHT hand imagery → C4 shows ERD
```

**Daily Calibration Checklist:**
- Check electrode-skin contact (gel application if needed).
- Verify signal quality: SNR > 10 dB, no flat channels.
- Record 30 seconds of resting baseline before each session.
- Log any environmental noise sources (WiFi routers, fluorescent lights).

**Signal Quality Check (Python):**
```python
import serial
import numpy as np

# Connect to Upside Down Labs kit
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

def collect_baseline(duration_sec=30, fs=250):
    samples = []
    for _ in range(duration_sec * fs):
        line = ser.readline().decode('utf-8').strip()
        samples.append([float(x) for x in line.split(',')])
    return np.array(samples)

baseline = collect_baseline()
snr = 10 * np.log10(np.mean(baseline**2) / np.var(baseline))
print(f"Signal SNR: {snr:.1f} dB  {'✓ Good' if snr > 10 else '✗ Poor'}")
```

### 2.2 Subject Preparation Protocol

**Before Each Session:**
- Obtain written informed consent.
- Advise subject to avoid caffeine 2 hours before session.
- Schedule in the morning (more stable alpha rhythms).
- Seat subject in a comfortable, upright chair facing the monitor.

**During Session:**
- Apply conductive gel to electrode sites if needed.
- Check signal quality before beginning trials.
- Run 5 practice trials (not recorded) so subject understands the task.
- Allow 2-minute rest breaks every 10 minutes.
- Keep total recording session under 30 minutes.

**Trial Timing Sequence:**
```
t = 0s  →  [RELAX]         "Ready" — subject rests, no imagery
t = 2s  →  [CUE]           Arrow LEFT or RIGHT displayed for 1 second
t = 3s  →  [MOTOR IMAGERY] "Imagine squeezing LEFT/RIGHT fist" — 4 seconds
t = 7s  →  [REST]          Blank screen — 2 seconds recovery
t = 9s  →  Next trial begins
```

**Session Structure:**
- Block 1: 20 trials (10 LEFT + 10 RIGHT) — 3 minutes
- Rest: 2 minutes
- Block 2: 20 trials — 3 minutes
- Rest: 2 minutes
- Block 3: 20 trials — 3 minutes
- Block 4: 20 trials — 3 minutes (optional, if subject is comfortable)

**Target: 80 trials per subject × 6 subjects = 480 total trials**

---

## Phase 3 — Week 2–3: Preprocessing Pipeline

### 3.1 Pipeline Steps (in order)

```python
import mne
import numpy as np
from mne.preprocessing import ICA

def preprocess_subject(raw_data, fs=250):
    # Step 1: Band-pass filter (8–30 Hz for motor imagery)
    raw_data.filter(l_freq=8.0, h_freq=30.0, method='fir')

    # Step 2: Notch filter (50 Hz power line noise — India)
    raw_data.notch_filter(freqs=50.0)

    # Step 3: Common Average Reference (reduce common-mode noise)
    raw_data.set_eeg_reference('average', projection=True)
    raw_data.apply_proj()

    # Step 4: ICA artifact removal (eye blinks, muscle noise)
    ica = ICA(n_components=15, random_state=42)
    ica.fit(raw_data)
    # Auto-detect eye blink components
    eog_indices, _ = ica.find_bads_eog(raw_data)
    ica.exclude = eog_indices
    ica.apply(raw_data)

    # Step 5: Epoch extraction (0 to 4 seconds post-cue)
    events, event_id = mne.events_from_annotations(raw_data)
    epochs = mne.Epochs(raw_data, events, event_id,
                        tmin=0.0, tmax=4.0,
                        baseline=(0, 0.5),
                        reject=dict(eeg=150e-6),  # reject high-variance trials
                        preload=True)

    # Step 6: Downsample to 125 Hz (speeds up training)
    epochs.resample(125)

    return epochs
```

### 3.2 Quality Metrics to Document

| Metric | Acceptable Range |
|--------|-----------------|
| Trials retained after rejection | > 75% of total |
| Channels with flat signal | 0 |
| ICA components removed | 1–3 per subject |
| Epoch peak-to-peak amplitude | < 150 µV |

---

## Phase 4 — Week 3–4: Feature Extraction

### 4.1 Traditional Features (SVM Baseline)

**Band Power Features:**
```python
from scipy import signal as sp_signal

def extract_band_power(epoch, fs=125):
    freqs, psd = sp_signal.welch(epoch, fs=fs, nperseg=256)
    alpha = np.mean(psd[(freqs >= 8) & (freqs <= 12)], axis=-1)
    beta  = np.mean(psd[(freqs > 12) & (freqs <= 30)], axis=-1)
    return np.concatenate([alpha, beta])  # shape: (2 * n_channels,)
```

**Common Spatial Patterns (CSP):**
```python
from mne.decoding import CSP

csp = CSP(n_components=4, reg=None, log=True)
X_csp = csp.fit_transform(X_train, y_train)
# Reduces n_channels → 4 most discriminative spatial filters
```

**Total feature vector: ~64 features → reduced to 16–20 via PCA**

### 4.2 Deep Learning (1D-CNN — Direct from Raw Signal)

Skip hand-crafted features. Feed preprocessed epochs directly to the CNN.

```python
import tensorflow as tf

def build_1d_cnn(n_channels=7, n_timepoints=500, n_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_channels, n_timepoints)),
        tf.keras.layers.Reshape((n_timepoints, n_channels)),

        # Temporal Conv Block 1
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # Temporal Conv Block 2
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

---

## Phase 5 — Week 4–5: Model Training & Evaluation

### 5.1 SVM Baseline

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

# LOSO cross-validation
logo = LeaveOneGroupOut()
scores = []
for train_idx, test_idx in logo.split(X, y, groups=subject_ids):
    svm_pipeline.fit(X[train_idx], y[train_idx])
    scores.append(svm_pipeline.score(X[test_idx], y[test_idx]))

print(f"SVM LOSO Accuracy: {np.mean(scores):.1%} ± {np.std(scores):.1%}")
```

### 5.2 1D-CNN Training

```python
# Data augmentation
def augment_epoch(epoch, noise_std=0.05):
    shift = np.random.randint(-10, 10)
    noisy = epoch + np.random.normal(0, noise_std, epoch.shape)
    return np.roll(noisy, shift, axis=-1)

# LOSO training loop
for subject_out in range(n_subjects):
    X_train = X[subject_ids != subject_out]
    y_train = y[subject_ids != subject_out]
    X_test  = X[subject_ids == subject_out]
    y_test  = y[subject_ids == subject_out]

    model = build_1d_cnn()
    model.fit(X_train, tf.keras.utils.to_categorical(y_train),
              epochs=50, batch_size=32,
              validation_split=0.15,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

    acc = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test))[1]
    print(f"Subject {subject_out+1} accuracy: {acc:.1%}")
```

### 5.3 Evaluation Metrics to Report

- Accuracy (primary, per subject + mean ± std)
- Precision, Recall, F1-score per class
- Confusion matrix (heatmap)
- ROC-AUC curve
- Training/validation loss curves (for overfitting check)
- Processing latency (ms per prediction)

---

## Phase 6 — Week 6: Real-Time Integration

### 6.1 Flask API

```python
# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)
CORS(app)

cnn_model = tf.keras.models.load_model('data/models/cnn_model.h5')
svm_model  = pickle.load(open('data/models/svm_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    epoch = np.array(data['epoch'])  # shape: (n_channels, n_timepoints)
    epoch_input = epoch[np.newaxis, ...]  # add batch dim

    probs = cnn_model.predict(epoch_input)[0]
    label = ['LEFT', 'RIGHT'][np.argmax(probs)]
    confidence = float(np.max(probs))

    return jsonify({'prediction': label, 'confidence': confidence,
                    'probabilities': probs.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 6.2 Real-Time Acquisition Loop

```python
# src/acquisition.py
import serial, threading, queue
import numpy as np

BUFFER_SIZE = 500  # 4 seconds at 125 Hz
data_queue  = queue.Queue()

def stream_eeg(port='/dev/ttyUSB0', fs=125):
    ser = serial.Serial(port, 115200, timeout=1)
    buffer = []
    while True:
        line = ser.readline().decode().strip()
        sample = [float(x) for x in line.split(',')]
        buffer.append(sample)
        if len(buffer) >= BUFFER_SIZE:
            data_queue.put(np.array(buffer).T)  # (channels, time)
            buffer = buffer[BUFFER_SIZE // 2:]   # 50% overlap

# Start streaming in background thread
thread = threading.Thread(target=stream_eeg, daemon=True)
thread.start()
```

---

## Phase 7 — Week 7: Adaptive Calibration

### 7.1 Per-User Fine-Tuning

```python
def calibrate_for_user(base_model, user_epochs, user_labels,
                       n_epochs=10, lr=0.0001):
    # Freeze all layers except the final classifier head
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Use a lower learning rate to avoid catastrophic forgetting
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy', metrics=['accuracy'])

    base_model.fit(user_epochs,
                   tf.keras.utils.to_categorical(user_labels),
                   epochs=n_epochs, batch_size=8, verbose=0)
    return base_model
```

### 7.2 Confidence Thresholding

```python
def safe_predict(model, epoch, threshold=0.65):
    probs = model.predict(epoch[np.newaxis, ...])[0]
    confidence = np.max(probs)
    if confidence < threshold:
        return 'NEUTRAL', confidence  # hold, don't act
    return ['LEFT', 'RIGHT'][np.argmax(probs)], confidence
```

---

## Phase 8 — Week 8: Thesis Writing & Demo Preparation

### 8.1 Thesis Structure

```
1. Introduction (3–4 pages)
   - Problem: paralysis and the BCI accessibility gap
   - Proposed solution and its novelty
   - Contributions summary

2. Background & Literature Review (6–8 pages)
   - EEG neurophysiology
   - Motor imagery paradigm and ERD/ERS
   - Classical BCI approaches (CSP + SVM)
   - Deep learning for EEG (CNNs, LSTMs, EEGNet)
   - Affordable neurotech landscape

3. Methodology (8–10 pages)
   - Hardware setup (Upside Down Labs kit)
   - Data collection protocol
   - Preprocessing pipeline
   - Feature extraction
   - Model architectures
   - Adaptive calibration
   - Evaluation protocol (LOSO CV)

4. Results (6–8 pages)
   - Accuracy tables (per subject + mean)
   - Confusion matrices
   - Comparison: SVM vs 1D-CNN vs LSTM
   - Latency measurements
   - Statistical significance tests

5. Discussion (4–5 pages)
   - Analysis of results
   - Comparison with published benchmarks
   - Clinical implications
   - Limitations and mitigation strategies
   - Future work

6. Conclusion (1–2 pages)

7. References (30+ papers)

8. Appendices
   - Code listings
   - Data collection forms
   - Ethics consent template
   - Full results tables
```

### 8.2 Key Figures to Include

- [ ] System architecture diagram (end-to-end pipeline)
- [ ] EEG electrode placement diagram
- [ ] Raw vs. filtered EEG signal comparison
- [ ] ICA component visualization
- [ ] Trial timing paradigm figure
- [ ] Model architecture diagram (1D-CNN layers)
- [ ] Training/validation loss and accuracy curves
- [ ] Confusion matrix heatmap (per model)
- [ ] Accuracy comparison bar chart (SVM vs CNN vs LSTM)
- [ ] LOSO per-subject accuracy scatter plot
- [ ] Real-time UI screenshot
- [ ] Topographic brain activity map during LEFT vs RIGHT imagery

---

## Weekly Milestones Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** | Hardware setup, literature review, environment | Working kit, environment, 5 key papers read |
| **2** | Data collection (You + 2 volunteers) | 240 trials, raw EEG saved |
| **3** | Data collection (3 more subjects) + preprocessing | 480 trials, clean epochs |
| **4** | Feature extraction + SVM baseline | SVM LOSO accuracy ≥ 78% |
| **5** | 1D-CNN training + evaluation | CNN LOSO accuracy ≥ 85% |
| **6** | Real-time system + Flask API + UI | Working demo end-to-end |
| **7** | Adaptive calibration + LSTM bonus | Per-user fine-tuning live |
| **8** | Thesis writing + demo prep + defence | Complete thesis + live demo |

---

## Contingency Plans

| Scenario | Fallback |
|----------|----------|
| Can't collect 6 subjects | Use 3–4 subjects + augment with PhysioNet data (transfer learning) |
| Hardware unreliable | Simulate EEG using PhysioNet data streamed in real-time |
| Accuracy stuck below 70% | Try 10–28 Hz band, increase subjects, switch to ensemble models |
| Running out of time (4 weeks left) | Skip LSTM, prioritize SVM + CNN + thesis writing |
| Demo fails live | Pre-recorded video backup of successful prediction session |

---

## Critical Success Factors

**Do:**
- Collect real data with Upside Down Labs kit — this is the novel contribution.
- Use LOSO cross-validation — it is the gold standard for BCI papers.
- Compare at least two models (SVM and CNN) to show engineering depth.
- Document every preprocessing decision with justification.
- Have a backup video for the live demo.

**Avoid:**
- Jumping to fancy architectures before establishing a working baseline.
- Using standard k-fold cross-validation (not accepted in BCI literature).
- Claiming clinical validation without actual paralyzed patient testing.
- Leaving thesis writing to the final week.

---

*Good luck! Build incrementally, test constantly, and document as you go.* 🧠⚡
