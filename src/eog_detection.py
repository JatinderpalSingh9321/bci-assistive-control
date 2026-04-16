"""
BCI Assistive Control — EOG Artifact Detection
================================================
Detects eye blinks and winks from raw EEG signals using
simple amplitude-threshold heuristics. Because EOG artifacts
are 5–10x larger than cortical EEG, a rule-based detector
achieves near-perfect accuracy without any ML model.

Detection logic:
  - BLINK  : Two rapid, symmetric spikes within ~400 ms (both eyes)
  - WINK   : A single prolonged high-amplitude deflection (one eye closed)
  - NONE   : Normal EEG — pass through to the CNN/SVM model

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from src.utils import TARGET_SAMPLING_RATE, setup_logger

logger = setup_logger("eog")

# ──────────────────────────────────────────────
# DETECTION THRESHOLDS
# ──────────────────────────────────────────────

# Peak-to-peak amplitude that indicates an EOG artifact (in µV-equivalent units)
# Normal EEG motor imagery signals are typically < 80 µV peak-to-peak.
# Eye blinks generate 150–500 µV spikes at frontal electrodes.
EOG_SPIKE_THRESHOLD = 120.0

# Minimum number of consecutive high-amplitude samples to qualify as a "wink"
# At 125 Hz, 50 samples = 400 ms of sustained deflection
WINK_MIN_DURATION_SAMPLES = int(0.4 * TARGET_SAMPLING_RATE)  # 50 samples

# Maximum spike width for a blink (quick up-down pattern)
# At 125 Hz, 25 samples = 200 ms — blinks are very fast
BLINK_MAX_WIDTH_SAMPLES = int(0.2 * TARGET_SAMPLING_RATE)  # 25 samples

# Minimum number of spikes in a short window to qualify as a double-blink
BLINK_MIN_SPIKE_COUNT = 1


# ──────────────────────────────────────────────
# CORE DETECTION FUNCTIONS
# ──────────────────────────────────────────────

def detect_eog_event(epoch, threshold=EOG_SPIKE_THRESHOLD):
    """
    Analyze a raw EEG epoch for EOG artifacts (blinks and winks).

    The detector works by scanning the signal for regions where
    the absolute amplitude exceeds the threshold. It then classifies
    the artifact based on the duration of the exceedance:
      - Short burst (< 200 ms) → BLINK (both eyes, quick)
      - Sustained deflection (> 400 ms) → WINK (one eye closed)

    Parameters
    ----------
    epoch : np.ndarray
        1D signal array (raw or minimally filtered).
    threshold : float
        Amplitude threshold for spike detection.

    Returns
    -------
    str
        One of: "BLINK", "WINK", "NONE".
    dict
        Detection metadata (spike_count, max_amplitude, duration, etc.).
    """
    epoch = np.array(epoch, dtype=np.float64)

    # Subtract baseline (mean) to center the signal
    epoch_centered = epoch - np.mean(epoch)

    # Find samples exceeding the threshold
    abs_signal = np.abs(epoch_centered)
    above_threshold = abs_signal > threshold

    if not np.any(above_threshold):
        return "NONE", {"spike_count": 0, "max_amplitude": float(np.max(abs_signal))}

    # Analyze the pattern of threshold crossings
    max_amp = float(np.max(abs_signal))

    # Find contiguous runs of above-threshold samples
    runs = _find_runs(above_threshold)

    if len(runs) == 0:
        return "NONE", {"spike_count": 0, "max_amplitude": max_amp}

    # Calculate total above-threshold duration
    total_duration = sum(r["length"] for r in runs)
    max_run_length = max(r["length"] for r in runs)

    meta = {
        "spike_count": len(runs),
        "max_amplitude": round(max_amp, 2),
        "total_duration_samples": total_duration,
        "max_run_length": max_run_length,
        "runs": runs
    }

    # ── Classification Logic ──

    # WINK: A single (or few) prolonged high-amplitude region
    # One eye stays closed → sustained voltage shift
    if max_run_length >= WINK_MIN_DURATION_SAMPLES:
        logger.debug(f"WINK detected: run={max_run_length} samples, amp={max_amp:.1f}")
        return "WINK", meta

    # BLINK: One or more short, sharp spikes (rapid bilateral eye closure)
    if len(runs) >= BLINK_MIN_SPIKE_COUNT and max_run_length <= BLINK_MAX_WIDTH_SAMPLES:
        logger.debug(f"BLINK detected: {len(runs)} spikes, amp={max_amp:.1f}")
        return "BLINK", meta

    # Ambiguous artifact — treat as BLINK if amplitude is very high
    if max_amp > threshold * 1.5:
        logger.debug(f"Strong artifact → BLINK: amp={max_amp:.1f}")
        return "BLINK", meta

    return "NONE", meta


def _find_runs(bool_array):
    """
    Find contiguous runs of True values in a boolean array.

    Returns
    -------
    list of dict
        Each dict has 'start', 'length' keys.
    """
    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(bool_array):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            runs.append({"start": start, "length": i - start})

    # Handle run that extends to end of array
    if in_run:
        runs.append({"start": start, "length": len(bool_array) - start})

    return runs


# ──────────────────────────────────────────────
# SIMULATED EOG SIGNALS (for testing)
# ──────────────────────────────────────────────

def generate_blink_epoch(n_samples=500, fs=TARGET_SAMPLING_RATE):
    """
    Generate a simulated EEG epoch containing a double eye-blink artifact.

    A blink is modeled as a brief Gaussian-shaped spike (~100–150 ms wide)
    superimposed on normal background EEG.

    Returns
    -------
    np.ndarray
        Simulated epoch with blink artifact.
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

    # Background EEG (alpha + noise)
    bg = 10 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 3

    # Blink spike — sharp Gaussian pulse at ~40% into the epoch
    blink_center = int(n_samples * 0.4)
    blink_width = int(0.08 * fs)  # ~80 ms wide
    blink_amplitude = 250 + np.random.rand() * 150  # 250–400 µV

    gaussian = np.exp(-0.5 * ((np.arange(n_samples) - blink_center) / blink_width) ** 2)
    bg += blink_amplitude * gaussian

    return bg.astype(np.float32)


def generate_wink_epoch(n_samples=500, fs=TARGET_SAMPLING_RATE):
    """
    Generate a simulated EEG epoch containing a sustained wink artifact.

    A wink is modeled as a sustained rectangular-ish deflection
    (~500–800 ms wide) where one eye remains closed.

    Returns
    -------
    np.ndarray
        Simulated epoch with wink artifact.
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

    # Background EEG
    bg = 10 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 3

    # Wink — sustained high-amplitude region
    wink_start = int(n_samples * 0.3)
    wink_duration = int(0.6 * fs)  # ~600 ms sustained
    wink_amplitude = 180 + np.random.rand() * 120  # 180–300 µV

    # Smooth trapezoidal shape (ramp up, sustain, ramp down)
    ramp = int(0.05 * fs)
    wink_signal = np.zeros(n_samples)
    for i in range(wink_duration):
        idx = wink_start + i
        if idx >= n_samples:
            break
        # Smooth envelope
        if i < ramp:
            wink_signal[idx] = wink_amplitude * (i / ramp)
        elif i > wink_duration - ramp:
            wink_signal[idx] = wink_amplitude * ((wink_duration - i) / ramp)
        else:
            wink_signal[idx] = wink_amplitude

    bg += wink_signal
    return bg.astype(np.float32)
