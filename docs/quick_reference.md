# BCI Assistive Control — Quick Reference
## Group No. 7 | Upside Down Labs + Deep Learning

---

## Project Snapshot

| Aspect | Details |
|--------|---------|
| **Title** | Non-Invasive BCI for Assistive Control using Affordable Neuro-Tech |
| **Group** | No. 7 |
| **Duration** | 8 weeks (1–2 months) |
| **Hardware** | Upside Down Labs BioAmp EXG Pill |
| **Task** | Motor imagery: LEFT vs RIGHT hand |
| **Subjects** | 6 people × 80 trials = 480 total |
| **Models** | SVM (baseline) + 1D-CNN (main) + LSTM (bonus) |
| **Expected Accuracy** | 81–87% (LOSO cross-validation) |
| **Backend** | Flask REST API |
| **Frontend** | Real-time browser dashboard |
| **Deliverable** | Working BCI app + thesis document |

---

## Quick Start (Week 1)

```bash
# 1. Clone repository
git clone https://github.com/your-repo/bci-assistive-control
cd bci-assistive-control

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify Upside Down Labs kit connection
python -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
print('Kit connected:', ser.isOpen())
line = ser.readline().decode().strip()
print('Sample reading:', line)
"

# 4. Run signal quality check
python src/acquisition.py --check-quality

# 5. Start data collection (subject 001)
python src/experiment.py --subject 1

# 6. Preprocess data
python src/preprocessing.py --all-subjects

# 7. Train models
python src/train.py --model cnn

# 8. Start Flask API
python api/app.py

# 9. Open dashboard in browser
open frontend/dashboard.html
```

---

## Weekly Milestones

| Week | Task | Deliverable |
|------|------|-------------|
| **1** | Hardware setup + literature review | Working kit, signal quality confirmed |
| **2** | Data collection — You + 2 volunteers | 240 trials, raw EEG saved |
| **3** | Data collection — 3 more subjects | 480 trials total |
| **4** | Preprocessing pipeline | Clean epochs, quality metrics |
| **5** | Feature extraction + SVM | SVM accuracy ≥ 78% |
| **6** | 1D-CNN training | CNN accuracy ≥ 85% |
| **7** | Real-time system + Flask + dashboard | Working end-to-end demo |
| **8** | Thesis writing + defence prep | Complete thesis + live demo |

---

## Key Electrode Locations

```
For MOTOR IMAGERY, focus on:

  Prefrontal:  Fp1 ←——→ Fp2    (planning signals)
  Frontal:     F3  ←——→ F4     (left/right motor planning)
  Central:     C3  ←——→ Cz ←——→ C4   (primary motor cortex)
  Parietal:    P3  ←——→ P4     (sensory integration)

Motor Imagery Signature (ERD):
  LEFT hand  → F3, C3, P3 desynchronize (alpha-beta power drops)
  RIGHT hand → F4, C4, P4 desynchronize
  Cz:          Active for both hands
```

---

## Preprocessing Pipeline

```python
# Apply in this exact order:

1. Notch filter: 50 Hz          (remove Indian power line noise)
2. Band-pass filter: 8–30 Hz    (isolate alpha + beta motor rhythms)
3. Common Average Reference      (reduce common-mode electrical noise)
4. ICA artifact removal          (remove eye blinks, muscle noise)
5. Epoch extraction: 0–4s post-cue
6. Baseline correction: subtract mean of 0–0.5s pre-imagery
7. Artifact rejection: drop epochs with peak-to-peak > 150 µV
8. Downsample: 250 Hz → 125 Hz  (speeds up training, no info lost)

Code:
  preprocessor = EEGPreprocessor(fs=250, lowcut=8, highcut=30)
  clean_epochs, clean_labels = preprocessor.preprocess_all(raw_trials, labels)
```

---

## Feature Extraction

```python
# For SVM baseline — total ~5 features per epoch:
  1. Alpha band power  (8–12 Hz)
  2. Beta band power   (12–30 Hz)
  3. Spectral entropy  (signal complexity)
  4. Signal variance   (amplitude measure)
  5. Mean absolute value

# For 1D-CNN — no manual features:
  Feed preprocessed epoch directly: shape (n_timepoints, 1)
  CNN learns discriminative features automatically

Expected feature importance:
  ✓ Beta ERD:             Most important (explains ~45% variance)
  ✓ Motor channels (C3, Cz, C4):  High importance
  ✓ Frontal channels (F3, F4):    Medium importance
  ✗ Prefrontal (Fp1, Fp2):        Lower importance (more artefact-prone)
```

---

## Model Comparison

| Model | Accuracy | Latency | Complexity | Recommended Use |
|-------|----------|---------|-----------|----------------|
| **SVM (RBF)** | ~81% | 25 ms | Low | Live demo (faster) |
| **1D-CNN** | ~85% | 50 ms | Medium | Thesis main model |
| **LSTM** | ~86% | 100 ms | High | Bonus experiment |

**Strategy:** Train both SVM + 1D-CNN. Report CNN in thesis. Use SVM for live demo (lower latency).

---

## Expected Results

```
SVM (baseline):    78 ± 2.5%
1D-CNN (main):     85 ± 3.1%
LSTM (bonus):      86 ± 2.8%
Published papers:  80–95%   ← Your results are competitive!

Per-Subject Breakdown (expected):
  Subject 1: 82%   (good signal)
  Subject 2: 79%   (noisier, more blinks)
  Subject 3: 85%   (strong motor imagery)
  Subject 4: 81%
  Subject 5: 83%
  Subject 6: 80%
  Mean:      81.7% ± 2.2%
```

---

## API Reference

**Base URL:** `http://localhost:5000`

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/health` | GET | — | `{status, models}` |
| `/predict` | POST | `{epoch: [float]}` | `{prediction, confidence, probabilities, latency_ms}` |
| `/calibrate` | POST | `{epochs: [[]], labels: []}` | `{status, n_samples}` |

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"epoch": [0.12, -0.34, 0.56, ...]}'

# Response:
# {
#   "prediction": "LEFT",
#   "confidence": 0.82,
#   "probabilities": {"LEFT": 0.82, "RIGHT": 0.18},
#   "latency_ms": 48.3
# }
```

---

## Data Structure

```
data/
├── raw/
│   ├── subject_001/
│   │   ├── session_01.npz          (data: 80×500, labels: 80)
│   │   └── session_01_meta.json    (date, SNR, notes)
│   ├── subject_002/ ...
│   └── subject_006/
├── preprocessed/
│   └── all_clean_epochs.npz        (ready for ML training)
└── models/
    ├── svm_model.pkl               (trained SVM)
    ├── cnn_model.h5                (trained 1D-CNN)
    └── results.csv                 (LOSO accuracy per subject)
```

---

## Common Pitfalls & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Low accuracy (< 70%) | Bad signal or too few trials | Check electrode contact, collect more data |
| High variance epochs | Eye blink / jaw clench artefacts | Apply ICA, reject flagged epochs |
| No serial data from kit | Wrong port or baud rate | Check `ls /dev/tty*`, use 115200 baud |
| API lag > 300 ms | CNN on CPU too slow | Switch to SVM for demo, use CNN offline |
| Poor signal quality | Dry electrodes | Apply Ten20 conductive gel |
| Classes imbalanced | Random trial order | Use strict alternating LEFT/RIGHT |
| Overfitting | Too few subjects | Add L2 regularization, increase dropout |

---

## Live Demo Script

```
[Fit Upside Down Labs electrode headband]

"The kit is now reading my brainwaves. Let me show you motor imagery."

[Close eyes]
"I'm imagining squeezing my LEFT fist..."
[Wait 4 seconds]
✓ Dashboard shows: LEFT HAND — 82% confidence

"Now RIGHT hand..."
[Imagine right fist]
✓ Dashboard shows: RIGHT HAND — 79% confidence

"A paralyzed patient could use these two commands to navigate a
computer cursor, control a robotic arm, or operate a powered wheelchair —
using only their thoughts, with hardware costing under ₹5,000."
```

---

## Pre-Defence Checklist

```
ONE WEEK BEFORE:
  ☐ All 480 trials collected and backed up
  ☐ All models trained and saved
  ☐ LOSO results table complete
  ☐ All figures at 300 DPI
  ☐ GitHub repo public with README
  ☐ Thesis proofread ×2

30 MIN BEFORE:
  ☐ Electrode gel fresh, kit charged
  ☐ Flask server running (python api/app.py)
  ☐ Dashboard loaded in browser
  ☐ Projector connected and mirroring
  ☐ Backup video queued (in case live demo fails)
  ☐ Practice demo 3× in the room
  ☐ Breathe. You built a working BCI. 🧠
```

---

## Key Papers to Cite

1. **Wolpaw et al. (2002)** — BCI fundamentals *(cite in Introduction)*
2. **Lawhern et al. (2018)** — EEGNet compact CNN *(cite in Model Architecture)*
3. **Roy et al. (2019)** — Deep learning for EEG, Nature Reviews *(cite in Related Work)*
4. **Ang et al. (2012)** — Filter Bank CSP feature extraction *(cite in Feature Extraction)*
5. **Padfield et al. (2019)** — Motor imagery BCI survey *(cite in Background)*

---

## EEG Quick Stats

```
Motor Imagery Frequency Bands:
  Alpha:  8–12 Hz   ← ERD during motor imagery
  Beta:  12–30 Hz   ← ERD during imagery, ERS after

Typical EEG Amplitudes:
  Raw signal:          10–100 µV
  After filtering:      5–50 µV
  Rejection threshold: > 150 µV peak-to-peak

System Latency Budget (Real-Time):
  EEG epoch (4 sec):     4,000 ms (unavoidable)
  Preprocessing:         50–100 ms
  CNN inference:         40–60 ms
  UI update:             10–20 ms
  Total perceived lag:   < 200 ms ✓ (clinically acceptable)

Subject Variability (Normal in BCI):
  Best performers:   90–95% accuracy
  Average:           80–85% accuracy
  Weakest:           70–75% accuracy
  → Inter-subject variability is expected and discussed in thesis
```

---

## Bonus Ideas (If Time Permits)

```
1. CROSS-DATASET GENERALIZATION
   Fine-tune on PhysioNet data before training on kit data
   Shows the model can leverage large public datasets

2. ONLINE LEARNING
   Model incrementally updates as the patient uses it
   Mimics real clinical deployment scenario

3. 4-CLASS CLASSIFICATION
   Add "both feet" + "tongue" imageries for richer vocabulary
   More impressive, enables multi-directional control

4. ATTENTION VISUALIZATION
   Grad-CAM or SHAP to highlight important time windows
   Answers "what brain signal does the model focus on?"

5. IOT DEVICE CONTROL
   Connect Flask predictions to an Arduino via MQTT
   Drive a servo motor or LED to show physical control
```

---

## Thesis Grade Targets

| Grade | Requirements |
|-------|-------------|
| **A / A+** | Working hardware system, ≥ 85% accuracy, complete pipeline, real-time demo works, publication-ready results |
| **A / A−** | Hardware mostly working, ≥ 80% accuracy, solid implementation, demo works most of the time |
| **B+ / A−** | Hardware tested, ≥ 75% accuracy (or strong explanation if lower), complete code, thorough writeup |

---

**YOU HAVE A COMPLETE ROADMAP. NOW BUILD IT. 🚀🧠**

*Start with Week 1 checklist. Work systematically. Test at every step.*
