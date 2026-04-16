"""
BCI Assistive Control — Configuration & Utility Functions
=========================================================
Central configuration constants and shared helper functions
used across all modules in the BCI pipeline.

Group No. 7 | 8th Semester Major Project
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────

# Project root directory (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PREPROCESSED_DIR   = DATA_DIR / "preprocessed"
MODELS_DIR         = DATA_DIR / "models"
RESULTS_DIR        = PROJECT_ROOT / "results"

# Ensure directories exist
for d in [RAW_DATA_DIR, PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# HARDWARE CONFIGURATION
# ──────────────────────────────────────────────

# Serial port for Upside Down Labs BioAmp EXG Pill + Arduino
# Adjust for your system: COM3 (Windows), /dev/ttyUSB0 (Linux), /dev/cu.usbmodem* (Mac)
SERIAL_PORT  = "COM7"
BAUD_RATE    = 115200
SERIAL_TIMEOUT = 1  # seconds

# Sampling parameters
SAMPLING_RATE       = 250    # Hz — set by Arduino firmware
TARGET_SAMPLING_RATE = 125   # Hz — after downsampling
DOWNSAMPLE_FACTOR    = SAMPLING_RATE // TARGET_SAMPLING_RATE  # = 2

# ──────────────────────────────────────────────
# EEG SIGNAL PARAMETERS
# ──────────────────────────────────────────────

# Frequency bands of interest
FREQ_LOW     = 8.0    # Hz — lower bound (alpha band start)
FREQ_HIGH    = 30.0   # Hz — upper bound (beta band end)
NOTCH_FREQ   = 50.0   # Hz — Indian power line frequency

# Band definitions for feature extraction
ALPHA_BAND   = (8, 12)    # Hz — Event-Related Desynchronization
BETA_BAND    = (12, 30)   # Hz — Motor imagery modulation

# ──────────────────────────────────────────────
# EXPERIMENT PARAMETERS
# ──────────────────────────────────────────────

N_SUBJECTS        = 6      # Total participants
TRIALS_PER_SUBJECT = 80    # 40 LEFT + 40 RIGHT (balanced)
TOTAL_TRIALS       = N_SUBJECTS * TRIALS_PER_SUBJECT  # 480

# Trial timing (seconds)
RELAX_DURATION     = 2.0   # "Ready" screen
CUE_DURATION       = 1.0   # Arrow display
IMAGERY_DURATION   = 4.0   # Motor imagery period (EEG recorded here)
REST_DURATION      = 2.0   # Inter-trial rest
TRIAL_DURATION     = RELAX_DURATION + CUE_DURATION + IMAGERY_DURATION + REST_DURATION  # = 9s

# Epoch parameters
EPOCH_SAMPLES      = int(IMAGERY_DURATION * TARGET_SAMPLING_RATE)  # 500 at 125 Hz
BASELINE_SAMPLES   = int(0.5 * TARGET_SAMPLING_RATE)               # 62 at 125 Hz

# ──────────────────────────────────────────────
# PREPROCESSING PARAMETERS
# ──────────────────────────────────────────────

FILTER_ORDER          = 4       # Butterworth filter order
NOTCH_Q               = 30      # Notch filter quality factor
ARTIFACT_THRESHOLD_UV = 150     # Peak-to-peak rejection threshold (µV)
MIN_TRIAL_RETENTION   = 0.75    # Warn if retention < 75%

# ──────────────────────────────────────────────
# MODEL PARAMETERS
# ──────────────────────────────────────────────

N_CLASSES = 2  # LEFT = 0, RIGHT = 1
CLASS_NAMES = ["LEFT", "RIGHT"]

# 1D-CNN hyperparameters
CNN_LEARNING_RATE = 0.001
CNN_EPOCHS        = 100
CNN_BATCH_SIZE    = 16
CNN_DROPOUT       = 0.3
CNN_PATIENCE      = 20   # Early stopping patience

# SVM hyperparameters
SVM_KERNEL = "rbf"
SVM_C      = 1.0
SVM_GAMMA  = "scale"

# Data augmentation
AUGMENT_NOISE_STD  = 0.08   # Gaussian noise standard deviation
AUGMENT_SHIFT_MAX  = 20     # Max time-shift (samples)

# ──────────────────────────────────────────────
# API CONFIGURATION
# ──────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = 5000
CONFIDENCE_THRESHOLD = 0.65  # Below this → NEUTRAL prediction

# ──────────────────────────────────────────────
# MODEL FILE PATHS
# ──────────────────────────────────────────────

CNN_MODEL_PATH = MODELS_DIR / "cnn_model.h5"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
RESULTS_CSV    = MODELS_DIR / "results.csv"

# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────

def setup_logger(name, level=logging.INFO):
    """Create a logger with a consistent format for all modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def subject_dir(subject_id):
    """Return the raw data directory for a given subject ID."""
    path = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_session_metadata(subject_id, session_id=1, **extra_fields):
    """
    Save a JSON metadata file for a recording session.

    Parameters
    ----------
    subject_id : int
        Subject number (1-indexed).
    session_id : int
        Session number (default 1).
    **extra_fields : dict
        Additional metadata (e.g., signal_snr_db, notes).
    """
    meta = {
        "subject_id": f"{subject_id:03d}",
        "session_id": f"{session_id:02d}",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sampling_rate_hz": SAMPLING_RATE,
        "target_sampling_rate_hz": TARGET_SAMPLING_RATE,
        "trials_per_subject": TRIALS_PER_SUBJECT,
        "imagery_duration_sec": IMAGERY_DURATION,
        "freq_band_hz": [FREQ_LOW, FREQ_HIGH],
        "electrode_placement": "C3,C4,Fpz (Single Channel Bipolar)",
    }
    meta.update(extra_fields)

    filepath = subject_dir(subject_id) / f"session_{session_id:02d}_meta.json"
    with open(filepath, "w") as f:
        json.dump(meta, f, indent=2)

    return filepath


def get_all_subjects():
    """Return a sorted list of subject IDs found in the raw data directory."""
    subjects = []
    if RAW_DATA_DIR.exists():
        for d in sorted(RAW_DATA_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("subject_"):
                try:
                    sid = int(d.name.split("_")[1])
                    subjects.append(sid)
                except ValueError:
                    pass
    return subjects


def print_banner():
    """Print a project banner to the console."""
    banner = """
    ==========================================================
              BCI Assistive Control - Group 7            
         Non-Invasive Motor Imagery Classification           
         Upside Down Labs  *  1D-CNN  *  Flask API           
    ==========================================================
    """
    print(banner)


if __name__ == "__main__":
    print_banner()
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Raw data dir:       {RAW_DATA_DIR}")
    print(f"Preprocessed dir:   {PREPROCESSED_DIR}")
    print(f"Models dir:         {MODELS_DIR}")
    print(f"Results dir:        {RESULTS_DIR}")
    print(f"Sampling rate:      {SAMPLING_RATE} Hz → {TARGET_SAMPLING_RATE} Hz")
    print(f"Frequency band:     {FREQ_LOW}–{FREQ_HIGH} Hz")
    print(f"Subjects × Trials:  {N_SUBJECTS} × {TRIALS_PER_SUBJECT} = {TOTAL_TRIALS}")
    print(f"Epoch samples:      {EPOCH_SAMPLES} (at {TARGET_SAMPLING_RATE} Hz)")
    print(f"Serial port:        {SERIAL_PORT} @ {BAUD_RATE} baud")
