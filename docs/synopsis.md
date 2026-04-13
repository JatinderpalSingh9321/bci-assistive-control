# Project Synopsis
## Non-Invasive Brain-Computer Interface (BCI) for Assistive Control using Affordable Neuro-Tech

**Group No.:** 7  
**Duration:** 8 Weeks (1–2 Months)  
**Hardware:** Upside Down Labs Bio-Signal Acquisition Kit  
**Domain:** Neurotechnology · Human-Computer Interaction · Deep Learning

---

## 1. Introduction

Motor paralysis resulting from spinal cord injuries, ALS, stroke, or other neurological disorders robs individuals of the ability to interact with the world around them. For millions of such patients globally, even basic acts of communication or device control require external assistance. Brain-Computer Interfaces (BCIs) offer a transformative solution — they establish a direct communication channel between the brain and external systems, bypassing the motor pathways that are damaged or non-functional.

However, existing BCI solutions are caught in a paradox: those that work reliably (invasive implants like Neuralink) require surgical procedures that carry significant risks, while accessible consumer devices often lack the signal fidelity required for clinically meaningful control. This project directly addresses that gap by building a complete, end-to-end, non-invasive BCI system using the **Upside Down Labs** open-hardware bio-signal acquisition kit — an affordable, research-grade EEG platform — paired with state-of-the-art deep learning to achieve real-time assistive control for paralyzed users.

---

## 2. Problem Statement

Paralyzed individuals currently face one of three unsatisfactory options:

- **Invasive implants**: Effective but risky, expensive (~$50,000+), and available only in specialized centres.
- **Commercial non-invasive BCIs**: Expensive ($1,000–$20,000), proprietary, and inaccessible in low-resource settings.
- **No assistive technology at all**: The reality for the majority of patients in developing countries.

The result is a massive **accessibility gap** — life-changing technology exists but is out of reach for those who need it most. This project proposes to close that gap using affordable open hardware (Upside Down Labs, ~₹3,000–₹5,000) and open-source software (Python, TensorFlow/PyTorch), making a deployable BCI system a practical reality for resource-constrained healthcare settings.

---

## 3. Objectives

1. Design and implement a complete non-invasive EEG acquisition pipeline using the Upside Down Labs bio-signal kit.
2. Build a robust preprocessing pipeline to clean and segment raw EEG signals for motor imagery tasks.
3. Develop and evaluate a 1D-CNN deep learning model for binary motor imagery classification (Left vs. Right hand).
4. Implement a real-time interface that maps classified brain signals to assistive control commands.
5. Incorporate adaptive calibration to account for inter-subject variability in brain signal patterns.
6. Demonstrate the complete system end-to-end with a working real-time demo suitable for clinical illustration.

---

## 4. Background & Motivation

### 4.1 What is EEG Motor Imagery?

Electroencephalography (EEG) measures the electrical activity of the brain via electrodes placed on the scalp. When a person imagines moving a limb — even without physically moving — the motor cortex generates characteristic patterns of neural oscillations. Specifically:

- **Event-Related Desynchronization (ERD):** A decrease in alpha (8–12 Hz) and beta (12–30 Hz) power over the contralateral motor cortex during imagined movement.
- **Event-Related Synchronization (ERS):** A rebound increase in beta power after the imagined movement ends.

These signatures are spatially distinct: imagining left-hand movement produces ERD on the contralateral right motor cortex (C4), while right-hand imagery produces ERD on the contralateral left motor cortex (C3). This spatial and spectral separability is the neurophysiological foundation of motor imagery BCI.

### 4.2 Why Upside Down Labs?

The Upside Down Labs bio-signal acquisition kits are open-hardware platforms designed for affordable biosignal research. Key advantages:

- **Cost:** Approximately ₹3,000–₹5,000 vs. ₹15,00,000+ for clinical EEG systems.
- **Open source:** Full hardware schematics and software libraries are publicly available.
- **Community-backed:** Active developer and researcher community ensures continued support.
- **Hackable:** Designed to interface with Python, Arduino, and Raspberry Pi, enabling custom ML pipelines.

### 4.3 Why Deep Learning for EEG Classification?

Traditional machine learning approaches (LDA, SVM with hand-crafted features like CSP band power) require extensive domain knowledge for feature engineering and often fail to generalize across subjects. Deep learning, specifically **1D Convolutional Neural Networks (1D-CNNs)**, learns discriminative temporal features directly from raw or minimally processed EEG signals. Key benefits:

- Automatic feature extraction without domain-specific engineering.
- Captures hierarchical temporal patterns across multiple time scales.
- Outperforms SVM/LDA on benchmark BCI datasets (Padfield et al., 2019; Roy et al., 2019).
- Supports end-to-end training from signal to command.

---

## 5. System Architecture

The proposed system is organized into three tightly coupled layers:

### Layer 1: Signal Acquisition & Preprocessing

**Hardware:** Upside Down Labs EEG kit (Single-channel: electrodes placed at C3, C4, and Fpz following the 10-20 international system).

**Software Pipeline:**
1. Raw EEG streaming via serial/USB interface to Python.
2. **Band-pass filtering (0.5–40 Hz):** Isolates physiologically relevant brainwave frequencies (delta, alpha, beta), removing DC drift and high-frequency muscle noise.
3. **Notch filter (50 Hz):** Removes Indian power line interference.
4. **Independent Component Analysis (ICA):** Decomposes mixed signals into statistically independent components, allowing removal of eye-blink and muscle artefacts without discarding entire epochs.
5. **Epoch segmentation:** Time-locked windows (0–4 seconds post-cue) extracted for each motor imagery trial.
6. **Baseline correction:** Signal normalized to a 0.5-second pre-cue baseline period.

### Layer 2: Feature Learning with Deep Learning

**Model Architecture — 1D-CNN:**

```
Input: EEG Epoch (channels × time_samples)
  ↓
Temporal Convolution Block 1 (64 filters, kernel=5, ReLU)
  ↓ BatchNorm → Dropout (0.3)
  ↓
Temporal Convolution Block 2 (128 filters, kernel=5, ReLU)
  ↓ BatchNorm → MaxPool(2) → Dropout (0.3)
  ↓
Flatten → Dense (128, ReLU) → Dropout (0.3)
  ↓
Output: Dense (2, Softmax) → [LEFT | RIGHT]
```

**Training Protocol:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical cross-entropy
- Evaluation: Leave-One-Subject-Out (LOSO) cross-validation
- Data augmentation: Time-shift, Gaussian noise injection
- Expected accuracy: 81–87%

**Baseline Comparison:**

| Model | Accuracy | Latency | Use Case |
|-------|----------|---------|----------|
| SVM (RBF kernel) | ~81% | 25 ms | Baseline / Real-time demo |
| 1D-CNN | ~85% | 50 ms | Main thesis model |
| LSTM | ~86% | 100 ms | Bonus / Sequence context |

### Layer 3: Real-Time Interface & Control

**Backend:** Flask REST API receives raw EEG epochs from the acquisition script, runs preprocessing and inference, and returns a classified command with confidence score.

**Frontend:** Lightweight browser-based dashboard (Antigravity-compatible) displays:
- Live EEG waveform visualization.
- Real-time prediction output (LEFT / RIGHT).
- Confidence bar per class.
- Control mapping overlay (cursor, wheelchair, robotic arm).

**IoT Extension:** Classified commands can be forwarded over MQTT to microcontroller-based systems (e.g., robotic arm, wheelchair motor controller), enabling physical device control.

**System Latency Budget:**

| Stage | Time |
|-------|------|
| Data collection (4 sec epoch) | 4,000 ms (required) |
| Preprocessing | 50–100 ms |
| Feature extraction / inference | 30–60 ms |
| API response + UI render | 10–20 ms |
| **Total perceived latency** | **~100–200 ms** |

---

## 6. Adaptive Calibration

A key challenge in BCI systems is **inter-subject variability** — the same mental task produces significantly different EEG patterns across individuals due to differences in brain anatomy, skull thickness, mental imagery strategy, and attention levels. To address this:

- **Session-level fine-tuning:** After collecting 20–30 calibration trials from a new user, the pre-trained CNN is fine-tuned on that individual's data using transfer learning.
- **Online adaptation:** A rolling buffer of recent predictions is used to update model weights incrementally during use, improving accuracy over time.
- **Confidence thresholding:** Predictions below a set confidence threshold (e.g., 65%) are withheld and a "neutral" command is issued, reducing false positives during noisy periods.

---

## 7. Data Collection Protocol

**Subjects:** 6 participants (diverse demographics for generalizability)  
**Trials per subject:** 80 trials (40 LEFT + 40 RIGHT)  
**Total dataset:** 480 trials

**Trial Timing Sequence:**
```
0s → [RELAX]          "Ready" screen — subject rests
2s → [CUE]            Arrow pointing LEFT or RIGHT appears
3s → [MOTOR IMAGERY]  Subject imagines hand movement for 4 seconds
7s → [REST]           Blank screen — 2 seconds recovery
9s → [NEXT TRIAL]     Loop repeats
```

**Session Structure per Subject:**
- 5 practice trials (familiarization)
- 3 blocks × 20 trials each (2-minute breaks between blocks)
- Total session time: ~30 minutes

**Ethics & Consent:**
- Written informed consent obtained from all subjects.
- Data anonymized and stored securely.
- No personally identifiable information retained.

---

## 8. VETS Evaluation

### Viability
The Upside Down Labs kit brings the hardware cost below ₹5,000, making this solution deployable in district hospitals, rehabilitation centres, and rural healthcare settings across India. The entire software stack — Python, MNE, TensorFlow, Flask — is open source with no licensing costs. The system can run on a standard laptop (no GPU required for inference), further lowering the barrier to deployment.

### Engineering Depth
This project demands mastery across multiple disciplines simultaneously:
- **Neuroscience:** Understanding EEG neurophysiology, motor cortex topology, and artifact sources.
- **Signal Processing:** Band-pass filtering, ICA, epoch extraction, spectral analysis.
- **Deep Learning:** 1D-CNN design, regularization, LOSO evaluation, transfer learning.
- **Systems Engineering:** Real-time data pipelines, REST APIs, IoT device integration, latency optimization.
- **Human Factors:** Subject preparation protocols, adaptive calibration, usability for disabled users.

The intersection of these fields at a functional, working-system level constitutes substantial engineering depth well beyond typical final-year projects.

### Trend Alignment
The global BCI market is projected to reach USD 6.2 billion by 2030, driven by:
- Advances in non-invasive neurotechnology (Neurosity, OpenBCI, Meta's CTRL-Labs acquisition).
- Growing demand for assistive technology in ageing and disabled populations.
- Government initiatives in India (NHRC, Ayushman Bharat) emphasizing affordable healthcare technology.
- Active academic interest in EEGNet, EEG-Conformers, and foundation models for biosignals.

This project is positioned at the leading edge of the affordable neurotech movement.

### Social & Sustainability Impact
- **Immediate impact:** Restores a degree of autonomous control and communication for paralyzed individuals who currently have none.
- **Inclusivity:** Low cost and open-source design enable deployment in communities that cannot afford proprietary systems.
- **Sustainability:** Edge inference (lightweight CNN on CPU) consumes minimal energy. No cloud dependency for real-time prediction.
- **Scalability:** The software pipeline is modular and hardware-agnostic — it can be extended to other Upside Down Labs modules (EMG, ECG) or higher-channel EEG kits as the project scales.

---

## 9. Expected Results

| Metric | Target |
|--------|--------|
| SVM baseline accuracy | ≥ 78% |
| 1D-CNN accuracy (LOSO CV) | ≥ 85% |
| Real-time latency | ≤ 200 ms |
| Subject coverage | 6 participants |
| Total trials | 480 |
| False positive rate | ≤ 15% |

---

## 10. Limitations & Future Work

**Current Limitations:**
- Binary classification only (LEFT vs. RIGHT) — limits command vocabulary.
- Small subject pool (6 subjects) — may limit generalizability claims.
- Upside Down Labs kit has fewer channels (vs. 22-channel clinical EEG) — some spatial resolution is sacrificed.
- EEG motor imagery requires training and concentration from the user — not plug-and-play.

**Future Extensions:**
1. **4-class classification:** Add "both feet" and "tongue" imagery for richer control vocabularies.
2. **P300 integration:** Combine motor imagery with P300 speller for communication applications.
3. **On-device inference:** Deploy model on Raspberry Pi for fully portable, standalone operation.
4. **Clinical trial:** Validate with actual paralyzed patients in a rehabilitation setting.
5. **Foundation model pre-training:** Leverage large EEG datasets (PhysioNet, MOABB) for pre-training before subject-specific fine-tuning.

---

## 11. References

1. Wolpaw, J. R., et al. (2002). Brain-computer interfaces for communication and control. *Clinical Neurophysiology*, 113(6), 767–791.
2. Padfield, N., et al. (2019). EEG-based brain-computer interfaces using motor-imagery: Techniques and challenges. *Sensors*, 19(6), 1423.
3. Roy, Y., et al. (2019). Deep learning-based electroencephalography analysis: a systematic review. *Journal of Neural Engineering*, 16(5).
4. Ang, K. K., et al. (2012). Filter Bank Common Spatial Pattern algorithm on BCI competition IV datasets 2a and 2b. *Frontiers in Neuroscience*, 6, 39.
5. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5).
6. Upside Down Labs. (2023). BioAmp EXG Pill — Open Hardware Biosignal Acquisition. Retrieved from https://upsidedownlabs.tech

---

*Submitted as part of the final-year major project. All code will be made available on GitHub under the MIT licence upon project completion.*
