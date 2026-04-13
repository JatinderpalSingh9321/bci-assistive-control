<p align="center">
  <h1 align="center">🧠 Non-Invasive BCI for Assistive Control</h1>
  <p align="center">
    <strong>Affordable Brain-Computer Interface using Upside Down Labs + Deep Learning</strong>
  </p>
  <p align="center">
    Group No. 7 &nbsp;·&nbsp; 8th Semester Major Project &nbsp;·&nbsp; 2026
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Hardware-Upside%20Down%20Labs-blueviolet?style=flat-square" alt="Hardware">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Software Setup](#-software-setup)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Data Collection Protocol](#-data-collection-protocol)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [API Reference](#-api-reference)
- [Real-Time Dashboard](#-real-time-dashboard)
- [Expected Results](#-expected-results)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)
- [License](#-license)

---

## 🔬 Overview

This project implements a **complete, end-to-end, non-invasive Brain-Computer Interface (BCI)** that enables paralyzed patients to generate assistive control commands using only their thoughts.

The system captures EEG signals during **motor imagery** (imagining hand movements), classifies them in real-time using a **1D Convolutional Neural Network**, and outputs control commands through a **live browser dashboard** — all powered by the affordable, open-hardware **Upside Down Labs BioAmp EXG Pill** kit.

### Key Highlights

| Feature | Details |
|---------|---------|
| 🎯 **Task** | Binary motor imagery classification (LEFT vs. RIGHT hand) |
| 🧬 **Hardware** | Upside Down Labs BioAmp EXG Pill (~₹3,000–₹5,000) |
| 🤖 **Main Model** | 1D-CNN (TensorFlow/Keras) |
| 📊 **Baseline** | SVM with hand-crafted spectral features |
| 🎓 **Evaluation** | Leave-One-Subject-Out (LOSO) cross-validation |
| 📈 **Expected Accuracy** | 81–87% |
| ⚡ **Inference Latency** | < 200 ms end-to-end |
| 👥 **Dataset** | 6 subjects × 80 trials = 480 trials |
| 🌐 **Interface** | Flask REST API + real-time browser dashboard |

---

## 🎯 Problem Statement

Millions of paralyzed individuals worldwide lack access to assistive technology:

- **Invasive BCIs** (e.g., Neuralink): Effective but require surgery, cost $50,000+, available only in specialized centres.
- **Commercial non-invasive BCIs**: Cost $1,000–$20,000, proprietary software, inaccessible in low-resource settings.
- **No technology at all**: The reality for most patients in developing countries.

**This project bridges that gap** by combining an affordable open-hardware EEG kit with state-of-the-art deep learning, creating a deployable BCI system for under ₹5,000 — practical for district hospitals, rehabilitation centres, and rural healthcare settings.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: SIGNAL ACQUISITION                  │
│                                                                 │
│  ┌──────────────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │  EEG Electrodes  │───▶│  BioAmp   │───▶│  Arduino Uno     │  │
│  │  (10-20 System)  │    │  EXG Pill │    │  (250 Hz ADC)    │  │
│  └──────────────────┘    └───────────┘    └────────┬─────────┘  │
│                                                     │ USB Serial│
├─────────────────────────────────────────────────────┼───────────┤
│                    LAYER 2: PROCESSING              │           │
│                                                     ▼           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Python Pipeline                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │   │
│  │  │ Notch   │─▶│Bandpass │─▶│  ICA    │─▶│ Baseline  │  │   │
│  │  │ 50 Hz   │  │ 8-30 Hz │  │ Cleanup │  │ Correct   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────┬─────┘  │   │
│  │                                                │        │   │
│  │                            ┌────────────────────▼──────┐ │   │
│  │                            │   1D-CNN / SVM Model      │ │   │
│  │                            │   Prediction + Confidence │ │   │
│  │                            └────────────────────┬──────┘ │   │
│  └─────────────────────────────────────────────────┼────────┘   │
├─────────────────────────────────────────────────────┼───────────┤
│                    LAYER 3: INTERFACE               │           │
│                                                     ▼           │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Flask REST API  │───▶│  Real-Time Browser Dashboard     │   │
│  │  localhost:5000   │    │  • Live EEG waveform             │   │
│  │  /predict         │    │  • LEFT / RIGHT prediction       │   │
│  │  /calibrate       │    │  • Confidence bar                │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hardware Requirements

### Components Checklist

| Component | Purpose | Est. Cost |
|-----------|---------|-----------|
| BioAmp EXG Pill | Biosignal amplifier (µV-range) | Included in kit |
| Gel electrodes (×3) | Scalp EEG contact | Included in kit |
| Reference electrode (clip) | Earlobe reference | Included in kit |
| Ground electrode (clip) | Earlobe / Fpz ground | Included in kit |
| Snap-to-lead wires | Electrode to amplifier | Included in kit |
| Arduino Uno / Nano | ADC + USB serial streaming | ₹500–₹800 |
| USB-A to USB-B cable | Arduino ↔ Laptop | Included |
| Electrode gel (Ten20) | Improve electrode contact | ₹200–₹500 |
| **Total** | | **~₹3,000–₹5,000** |

### Electrode Placement (Motor Imagery)

Follow the **10-20 international system**, focusing on motor cortex channels:

```
IN+ (Signal): C3 ( Left Motor Cortex)
IN- (Signal): C4 ( Right Motor Cortex)
Ground:      Fpz (Forehead midline) or Earlobe
```

---

## 💻 Software Setup

### Prerequisites

- Python 3.9+
- Arduino IDE (for firmware upload)
- Modern web browser (Chrome recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bci-assistive-control.git
cd bci-assistive-control

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import serial, mne, tensorflow, flask; print('✓ All dependencies OK')"
```

### `requirements.txt`

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
mne>=1.4
scikit-learn>=1.2
tensorflow>=2.12
flask>=2.3
flask-cors>=4.0
pyserial>=3.5
pygame>=2.5
```

---

## 📁 Project Structure

```
bci-assistive-control/
│
├── data/
│   ├── raw/                         # Raw EEG recordings per subject
│   │   ├── subject_001/
│   │   │   ├── session_01.npz       # numpy arrays: data (80×~1000), labels (80)
│   │   │   └── session_01_meta.json # metadata: date, SNR, electrode notes
│   │   └── subject_006/
│   ├── preprocessed/
│   │   └── all_clean_epochs.npz     # Cleaned, ready for ML
│   └── models/
│       ├── svm_model.pkl            # Trained SVM pipeline
│       ├── cnn_model.h5             # Trained 1D-CNN
│       └── results.csv              # Per-subject LOSO results
│
├── src/
│   ├── acquisition.py               # Serial streaming from Upside Down Labs kit
│   ├── experiment.py                # Motor imagery paradigm (Pygame)
│   ├── preprocessing.py             # Filtering, epoching, artifact rejection
│   ├── feature_extraction.py        # Band power, CSP features for SVM
│   ├── models.py                    # SVM, 1D-CNN, LSTM definitions
│   ├── evaluation.py               # LOSO cross-validation, metrics, plots
│   ├── train.py                     # Training orchestration
│   └── utils.py                     # Config and helpers
│
├── api/
│   └── app.py                       # Flask REST API
│
├── frontend/
│   └── dashboard.html               # Real-time browser dashboard
│
├── firmware/
│   └── eeg_stream.ino               # Arduino sketch (250 Hz streaming)
│
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── results/                         # Figures: confusion matrices, ROC, etc.
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🚀 Quick Start

### Step 1: Flash Arduino Firmware

Open `firmware/eeg_stream.ino` in Arduino IDE and upload to your Arduino Uno.

### Step 2: Verify Hardware Connection

```bash
python -c "
import serial, serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p.device, p.description)
"
```

### Step 3: Check Signal Quality

```bash
python src/acquisition.py --check-quality
```

### Step 4: Collect Data

```bash
python src/experiment.py --subject 1
```

### Step 5: Preprocess

```bash
python src/preprocessing.py --all-subjects
```

### Step 6: Train Models

```bash
# SVM baseline
python src/train.py --model svm

# 1D-CNN (main model)
python src/train.py --model cnn
```

### Step 7: Launch the System

```bash
# Terminal 1: Start API server
python api/app.py

# Terminal 2: Open dashboard
# Navigate to frontend/dashboard.html in your browser
```

---

## 📡 Data Collection Protocol

### Subject Requirements
- **6 participants** (diverse demographics)
- **80 trials per subject** (40 LEFT + 40 RIGHT, balanced)
- **480 total trials**

### Trial Timing

```
0s  ─── [RELAX]           "Ready" screen, subject rests
2s  ─── [CUE]             Arrow LEFT or RIGHT displayed (1 sec)
3s  ─── [MOTOR IMAGERY]   Imagine hand squeeze (4 sec) ← EEG RECORDED
7s  ─── [REST]            Blank screen (2 sec recovery)
9s  ─── [NEXT TRIAL]
```

### Session Protocol
1. Written informed consent
2. Electrode application + conductive gel
3. 30-second resting baseline recording
4. 5 practice trials (not recorded)
5. 4 blocks × 20 trials (2-min breaks between blocks)
6. Total duration: ~30 minutes per subject

---

## 🔄 Preprocessing Pipeline

Applied in strict order:

| Step | Operation | Parameters | Purpose |
|------|-----------|-----------|---------|
| 1 | Notch filter | 50 Hz, Q=30 | Remove power line noise |
| 2 | Band-pass filter | 8–30 Hz, 4th-order Butterworth | Isolate motor imagery bands |
| 3 | ICA removal | ~15 components | Remove eye blinks, muscle artifacts |
| 4 | Epoch extraction | 0–4 sec post-cue | Time-lock to imagery period |
| 5 | Baseline correction | Subtract mean(0–0.5s) | Remove DC offset |
| 6 | Artifact rejection | PTP > 150 µV → discard | Remove noisy trials |
| 7 | Downsample | 250 → 125 Hz | Reduce dimensionality |

**Quality target:** Retain > 75% of trials after rejection.

---

## 🤖 Model Architecture

### SVM Baseline

```
Input: 5 hand-crafted features per epoch
  → Alpha band power (8–12 Hz)
  → Beta band power (12–30 Hz)
  → Spectral entropy
  → Signal variance
  → Mean absolute value
  ↓
StandardScaler → SVM (RBF kernel, C=1.0)
  ↓
Output: LEFT / RIGHT
```

### 1D-CNN (Primary Model)

```
Input: (500 timepoints, 1 channel)
  ↓
Conv1D(64, k=5, ReLU) → BatchNorm → Dropout(0.3)
  ↓
Conv1D(128, k=5, ReLU) → BatchNorm → MaxPool(2) → Dropout(0.3)
  ↓
Conv1D(64, k=3, ReLU) → GlobalAveragePooling1D
  ↓
Dense(128, ReLU) → Dropout(0.3)
  ↓
Dense(2, Softmax) → [LEFT | RIGHT]

Optimizer:  Adam (lr=0.001)
Loss:       Categorical cross-entropy
Evaluation: LOSO cross-validation
```

---

## 🌐 API Reference

**Base URL:** `http://localhost:5000`

### `GET /health`

Check API and model status.

```json
// Response
{ "status": "ok", "models": ["cnn", "svm"] }
```

### `POST /predict`

Classify an EEG epoch.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"epoch": [0.12, -0.34, 0.56, ...]}'
```

```json
// Response
{
  "prediction": "LEFT",
  "confidence": 0.82,
  "probabilities": { "LEFT": 0.82, "RIGHT": 0.18 },
  "latency_ms": 48.3
}
```

### `POST /calibrate`

Fine-tune the model for a specific user.

```bash
curl -X POST http://localhost:5000/calibrate \
  -H "Content-Type: application/json" \
  -d '{"epochs": [[...], [...]], "labels": [0, 1, 0, ...]}'
```

```json
// Response
{ "status": "calibrated", "n_samples": 30 }
```

---

## 📺 Real-Time Dashboard

The browser dashboard (`frontend/dashboard.html`) provides:

- **Live EEG Waveform** — Canvas-rendered real-time signal trace
- **Prediction Output** — Large, color-coded LEFT / RIGHT display
- **Confidence Score** — Percentage + inference latency
- **Visual Styling** — Dark theme, glowing accents (cyan for LEFT, orange for RIGHT)

To launch:
```bash
python api/app.py                    # Start the API
# Open frontend/dashboard.html in Chrome
```

---

## 📈 Expected Results

### Model Comparison (LOSO Cross-Validation)

| Model | Accuracy | Latency | Recommended Use |
|-------|----------|---------|-----------------|
| SVM (RBF) | ~81% ± 2.5% | 25 ms | Live demo (fastest) |
| 1D-CNN | ~85% ± 3.1% | 50 ms | Thesis main result |
| LSTM | ~86% ± 2.8% | 100 ms | Bonus experiment |

### Per-Subject Breakdown (Expected)

| Subject | Accuracy | Notes |
|---------|----------|-------|
| 1 | ~82% | Good signal quality |
| 2 | ~79% | Noisier, more blinks |
| 3 | ~85% | Strong motor imagery |
| 4 | ~81% | Average |
| 5 | ~83% | Consistent |
| 6 | ~80% | Average |
| **Mean** | **~81.7% ± 2.2%** | |

### System Latency

| Stage | Time |
|-------|------|
| EEG epoch (4 sec) | 4,000 ms (required) |
| Preprocessing | 50–100 ms |
| CNN inference | 40–60 ms |
| UI update | 10–20 ms |
| **Total latency** | **< 200 ms ✓** |

---

## 🔧 Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| No serial data | Wrong COM port | Check Device Manager (Windows) or `ls /dev/tty*` (Linux) |
| Flat signal | Loose electrode, no gel | Re-apply conductive gel, check clips |
| 60/50 Hz noise | No notch filter / bad ground | Enable 50 Hz notch, check ground electrode |
| Low accuracy (< 70%) | Poor signal or few trials | Improve impedance, add subjects, tune band |
| API 500 error | Model not loaded / wrong shape | Verify `data/models/` path, check epoch dims |
| Real-time lag > 300 ms | CNN on CPU too slow | Use SVM for live demo, CNN for offline |
| Imbalanced classes | Unequal LEFT/RIGHT | Ensure 40 LEFT + 40 RIGHT per subject |

---

## 📚 References

1. Wolpaw, J. R., et al. (2002). Brain-computer interfaces for communication and control. *Clinical Neurophysiology*, 113(6), 767–791.
2. Padfield, N., et al. (2019). EEG-based brain-computer interfaces using motor-imagery. *Sensors*, 19(6), 1423.
3. Roy, Y., et al. (2019). Deep learning-based EEG analysis: a systematic review. *Journal of Neural Engineering*, 16(5).
4. Ang, K. K., et al. (2012). Filter Bank Common Spatial Pattern. *Frontiers in Neuroscience*, 6, 39.
5. Lawhern, V. J., et al. (2018). EEGNet: A compact CNN for EEG-based BCIs. *Journal of Neural Engineering*, 15(5).
6. Upside Down Labs. (2023). BioAmp EXG Pill — Open Hardware Biosignal Acquisition. https://upsidedownlabs.tech

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with open hardware, open software, and real clinical impact. 🧠⚡</strong>
</p>
<p align="center">
  <em>Group 7 — 8th Semester Major Project, 2026</em>
</p>
