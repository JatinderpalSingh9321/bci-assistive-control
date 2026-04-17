# BCI Assistive Control — Project Manual
### Group No. 7 | 8th Semester Major Project
### A Brain-Computer Interface for Hands-Free Mouse Control

---

## What Is This Project?

This project lets a person **control a computer mouse using only their brain and eye movements** — no hands needed.

We place small electrodes (sensors) on a person's head. These sensors pick up tiny electrical signals produced by the brain. A computer reads these signals, figures out what the person is *thinking* about doing, and moves the mouse cursor accordingly.

**Who is this for?**
People with paralysis, ALS, or other conditions that prevent them from using their hands. This system gives them a way to interact with a computer independently.

**How does it work (in simple words)?**

1. You wear electrodes on your head
2. You *imagine* moving your left hand → the cursor moves **left**
3. You *imagine* moving your right hand → the cursor moves **right**
4. You *imagine* curling your tongue → the cursor moves **up**
5. You *imagine* wiggling your toes → the cursor moves **down**
6. You **blink** both eyes → the mouse **clicks**
7. You **wink** one eye → the mouse **right-clicks** or **double-clicks**

No actual hand/eye movement is needed for the cursor — just *thinking* about it.

---

## Hardware Used

| Component | What It Does |
|-----------|-------------|
| **BioAmp EXG Pill** (by Upside Down Labs) | A tiny amplifier board that boosts the weak brain signals so the Arduino can read them |
| **Arduino R4 Minima** | Reads the amplified signal 250 times per second and sends it to the computer |
| **Gel Electrodes** (3 pieces) | Stick-on sensors placed on the scalp and earlobes to pick up brain/eye signals |
| **USB Cable** | Connects Arduino to the computer |

**Total cost:** Under ₹5,000

---

## The Journey: Problems We Faced and How We Solved Them

### Problem 1: "100% Accuracy in Testing, 50% in Real Life"

This was our biggest challenge. Here's what happened:

**What we expected:**
We trained a machine learning model (SVM — Support Vector Machine) on brain data. In testing, it scored **100% accuracy** — it could perfectly tell LEFT from RIGHT.

**What actually happened:**
When we tried it in real-time with actual electrodes on a person's head, the accuracy dropped to around **50%** — which is the same as random guessing (flipping a coin between LEFT and RIGHT).

**Why did this happen?**

The 100% accuracy was on **simulated data** — computer-generated fake brain signals. These fake signals had clear, perfect patterns:
- LEFT thinking = strong signal on one side
- RIGHT thinking = strong signal on the other side

Real brain signals are nothing like this. They are:
- **Very weak** — only 10-50 microvolts (a AA battery is 1,500,000 microvolts)
- **Very noisy** — muscle movement, power line interference (50 Hz hum), and eye blinks all create noise that is 10-100× stronger than the actual brain signal
- **Different for every person** — your brain patterns are not the same as mine

**The lesson:**
A model trained on clean, fake data will always score high on that same type of data. But it fails on real, messy data because it has never seen real-world noise. This is a well-known issue in BCI research — even professional systems struggle with this gap.

---

### Problem 2: Wrong Electrode Setup (Bipolar vs Referential)

This was the root cause of our 50% real-world accuracy.

**What is "Bipolar" recording?**

Our BioAmp EXG Pill has two input pins: IN+ and IN-. It outputs the *difference* between them.

In our **old setup (bipolar)**, we placed:
- IN+ on **C3** (left side of head, over left motor cortex)
- IN- on **C4** (right side of head, over right motor cortex)

The pill then outputs: **C3 minus C4**

**Why is this bad?**

When you imagine moving your LEFT hand:
- C3 shows a certain pattern (let's say value = 80)
- C4 shows a slightly different pattern (let's say value = 75)
- Output = 80 - 75 = **5** (tiny difference)

When you imagine moving your RIGHT hand:
- C3 shows value = 75
- C4 shows value = 80
- Output = 75 - 80 = **-5** (tiny difference, opposite sign)

The problem: **the useful signal (the difference) is tiny**, and it gets buried in noise. Both sides of the brain produce similar signals during motor imagery, so subtracting them cancels out most of the information.

**What is "Referential" recording? (Our fix)**

In the **new setup (referential)**, we place:
- IN+ on **C3** (left motor cortex — where the action is)
- IN- on the **earlobe** (electrically quiet — no brain signal)

The pill now outputs: **C3 minus earlobe = just C3**

This preserves the **full brain signal** from C3 without cancellation. The earlobe has essentially zero brain activity, so it acts as a clean reference point.

**The difference in simple terms:**

| Setup | What You Measure | Signal Quality | Analogy |
|-------|-----------------|---------------|---------|
| **Bipolar** (C3 - C4) | Difference between two brain areas | Very weak, signals cancel | Weighing two similar apples to find which is heavier — hard to tell |
| **Referential** (C3 - earlobe) | Full signal from one brain area | Strong, clear | Weighing one apple against nothing — easy to measure |

---

### Problem 3: Hair Blocking Electrode Contact

EEG electrodes need good skin contact to pick up signals. But C3 and C4 are on the scalp where there is hair. Hair acts as an insulator and blocks the signal.

**Our solution: Two-step approach**

Instead of trying to record from 4 scalp locations simultaneously (which would need 4 amplifiers and lots of hair-parting), we split data collection into two sessions:

**Test 1 — Brain signals (for cursor movement)**
- Place ONE electrode at C3 (part the hair, use conductive gel)
- Record LEFT/RIGHT/UP/DOWN motor imagery

**Test 2 — Eye signals (for clicking)**
- Move the SAME electrode to the forehead (Fp1 — no hair!)
- Record BLINK/WINK_LEFT/WINK_RIGHT

The forehead has no hair, so electrode contact is easy and signals are huge (200+ microvolts vs 10-50 for brain signals). Blinks and winks create massive electrical artifacts that are very easy to detect.

---

### Problem 4: Only 2 Directions (LEFT/RIGHT) → Need 4

Originally, the system could only detect LEFT and RIGHT motor imagery. For full mouse control, we need UP and DOWN too.

**Our solution: Different mental tasks**

Motor imagery isn't limited to hands. Different body parts activate different brain regions:

| Direction | Mental Task | What Happens in the Brain |
|-----------|------------|--------------------------|
| LEFT | Imagine squeezing left fist | Less suppression at C3 (ipsilateral) = higher mu rhythm |
| RIGHT | Imagine squeezing right fist | Strong suppression at C3 (contralateral) = lower mu rhythm |
| UP | Imagine curling tongue upward | Burst of high-beta activity (25-30 Hz) |
| DOWN | Imagine wiggling toes | Increase in theta rhythm (4-8 Hz) |

Each task creates a different frequency pattern at C3, and our feature extraction captures these differences using:
- **Band power** — how much energy is in each frequency range
- **ERD/ERS** — whether the mu rhythm gets suppressed or enhanced
- **Hjorth parameters** — mathematical descriptions of signal shape
- **Spectral entropy** — how "complex" the signal is

---

## The Machine Learning Pipeline

### How the Model Works

1. **Data Collection** — The person sits in front of a screen that shows arrows (←, →, ↑, ↓). They imagine the corresponding movement for 4 seconds while we record the EEG signal.

2. **Feature Extraction** — We don't feed raw EEG into the model. Instead, we calculate 42 meaningful numbers (features) from each 4-second recording:
   - Power in 7 frequency bands (delta, theta, alpha, beta, gamma)
   - Ratios between bands (e.g., theta/alpha ratio helps detect DOWN vs LEFT)
   - Signal statistics (variance, skewness, kurtosis)
   - Time-frequency patterns

3. **Classification** — We use an **SVM (Support Vector Machine)** classifier that learns to map these 42 features to one of 4 classes.

4. **Cross-Validation** — We test the model by splitting data into 5 parts, training on 4 parts and testing on 1, rotating through all combinations. This gives a realistic accuracy estimate.

### Current Accuracy Results

| Data Type | Accuracy | Why |
|-----------|----------|-----|
| **Simulated data** | 100% | Clean, perfect signals with clear patterns |
| **Old real data (bipolar)** | ~50% | Wrong electrode setup, signals cancelled out |
| **New real data (referential)** | To be collected | Expected to be significantly better |

**Why simulated = 100% but real = 50%:**

Think of it like training someone to recognize dogs vs cats using perfect textbook photos. They'll score 100% on the textbook test. But show them a blurry, dark photo taken from a weird angle? They might fail. The model needs to be trained on *real, messy data* to work in the real world.

**Expected improvement with referential setup:**
Based on BCI literature, switching from bipolar to referential recording should increase the Signal-to-Noise Ratio from approximately -10 dB to +5-8 dB. This means the brain signal will be **10-100× more visible** above the noise floor. Published studies show 70-85% accuracy for 2-class motor imagery with similar hardware.

---

## File Structure

```
d:\8th sem\bio\
│
├── firmware/
│   └── eeg_stream.ino           ← Arduino code (upload to board)
│
├── src/
│   ├── test_motor_imagery.py    ← TEST 1: Collect LEFT/RIGHT/UP/DOWN (C3)
│   ├── test_blink_wink.py       ← TEST 2: Collect BLINK/WINK (Fp1 forehead)
│   ├── train_referential.py     ← Train the SVM model
│   ├── mouse_control_ref.py     ← Real-time mouse control
│   ├── acquisition.py           ← Serial data reading
│   ├── utils.py                 ← Constants and helpers
│   └── eye_tracker.py           ← Webcam-based eye tracking (backup)
│
├── data/
│   ├── raw/                     ← Recorded EEG data (.npz files)
│   └── models/                  ← Trained model files (.pkl)
│
├── results/                     ← Accuracy reports and plots
├── frontend/                    ← Web dashboard (Flask)
└── requirements.txt             ← Python dependencies
```

---

## How to Run (Step by Step)

### Prerequisites
```bash
pip install numpy scipy scikit-learn matplotlib pygame pyserial pyautogui
```

### Step 1: Upload Arduino Firmware
1. Open `firmware/eeg_stream.ino` in the Arduino IDE
2. Select board: Arduino UNO R4 Minima
3. Select port: COM7 (or whichever port your board is on)
4. Click Upload

### Step 2: Collect Motor Imagery Data (Test 1)
**Electrode placement:** IN+ → C3 (scalp), IN- → left earlobe, GND → right earlobe
```bash
python -m src.test_motor_imagery --subject 1 --port COM7
```
Or try with fake data first:
```bash
python -m src.test_motor_imagery --subject 1 --simulate
```

### Step 3: Collect Blink/Wink Data (Test 2)
**Move IN+ only:** IN+ → Fp1 (forehead, above left eye)
```bash
python -m src.test_blink_wink --subject 1 --port COM7
```

### Step 4: Train the Model
```bash
python -m src.train_referential --subject 1
```

### Step 5: Control the Mouse!
**Put electrode back on C3 for live control:**
```bash
python -m src.mouse_control_ref --port COM7 --speed 30
```

**Safety:** Move your actual mouse to any screen corner to immediately stop the system.

---

## Glossary (Simple Definitions)

| Term | Meaning |
|------|---------|
| **BCI** | Brain-Computer Interface — a system that reads brain signals to control a device |
| **EEG** | Electroencephalography — recording electrical activity of the brain using scalp electrodes |
| **EOG** | Electrooculography — recording electrical signals from eye movements |
| **Motor Imagery** | Imagining a movement without actually doing it — this activates the same brain areas as real movement |
| **ERD** | Event-Related Desynchronization — when the brain's mu rhythm (10 Hz) gets suppressed during motor imagery |
| **C3/C4** | Standard electrode positions over the left/right motor cortex |
| **Fp1/Fp2** | Standard electrode positions on the left/right forehead |
| **Mu Rhythm** | A 8-12 Hz brainwave associated with the motor cortex — it decreases when you imagine movement |
| **SVM** | Support Vector Machine — a machine learning algorithm that finds the best boundary between different classes |
| **Bipolar** | Recording the difference between two active brain sites (our old, problematic setup) |
| **Referential** | Recording one brain site against a neutral reference like an earlobe (our improved setup) |
| **SNR** | Signal-to-Noise Ratio — how strong the useful signal is compared to noise. Higher = better |
| **BioAmp EXG Pill** | A small, affordable bio-signal amplifier made by Upside Down Labs (Indian company) |

---

## Summary of the Thesis Story

1. **We built a complete BCI pipeline** — from raw electrode signals to real-time mouse cursor control
2. **We achieved 100% accuracy on simulated data** — proving the ML pipeline (preprocessing → feature extraction → SVM classification) works correctly
3. **We discovered that real-world accuracy was only 50%** — and through systematic analysis, identified that the **bipolar electrode setup was the root cause**
4. **We designed an improved referential protocol** — which should eliminate signal cancellation and dramatically improve real-world accuracy
5. **We extended the system from 2 classes to 7** — adding UP, DOWN, BLINK, WINK_LEFT, WINK_RIGHT for full mouse control
6. **We solved the hair problem** — by splitting data collection into scalp (C3 for movement) and forehead (Fp1 for clicks)

This project demonstrates the full research cycle: build → test → fail → analyze → improve → rebuild.

---

*Last updated: April 2025 | Group 7 | Upside Down Labs BioAmp EXG Pill + Arduino R4 Minima*
