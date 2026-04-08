"""
BCI Assistive Control — EEG Preprocessing Pipeline
====================================================
Implements the complete preprocessing pipeline for motor imagery
EEG signals: notch filtering, bandpass filtering, downsampling,
baseline correction, and artifact rejection.

Pipeline order:
  1. Notch filter (50 Hz)
  2. Band-pass filter (8–30 Hz, 4th-order Butterworth)
  3. Downsample (250 Hz → 125 Hz)
  4. Baseline correction (subtract mean of first 0.5s)
  5. Artifact rejection (peak-to-peak > 150 µV → discard)

Group No. 7 | 8th Semester Major Project
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler

from src.utils import (
    SAMPLING_RATE, TARGET_SAMPLING_RATE, DOWNSAMPLE_FACTOR,
    FREQ_LOW, FREQ_HIGH, NOTCH_FREQ, NOTCH_Q, FILTER_ORDER,
    BASELINE_SAMPLES, ARTIFACT_THRESHOLD_UV, MIN_TRIAL_RETENTION,
    RAW_DATA_DIR, PREPROCESSED_DIR, EPOCH_SAMPLES,
    get_all_subjects, setup_logger
)

logger = setup_logger("preprocessing")


class EEGPreprocessor:
    """
    Complete EEG preprocessing pipeline for motor imagery BCI.

    Processes raw EEG epochs through filtering, downsampling,
    baseline correction, and artifact rejection to produce
    clean data ready for feature extraction or deep learning.

    Parameters
    ----------
    fs : int
        Original sampling rate in Hz (default: 250).
    lowcut : float
        Band-pass lower frequency in Hz (default: 8.0).
    highcut : float
        Band-pass upper frequency in Hz (default: 30.0).
    target_fs : int
        Downsampled sampling rate in Hz (default: 125).
    notch_freq : float
        Power line frequency to remove in Hz (default: 50.0).
    """

    def __init__(self, fs=SAMPLING_RATE, lowcut=FREQ_LOW, highcut=FREQ_HIGH,
                 target_fs=TARGET_SAMPLING_RATE, notch_freq=NOTCH_FREQ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.target_fs = target_fs
        self.notch_freq = notch_freq
        self.downsample_factor = fs // target_fs

        # Pre-compute filter coefficients (avoids recomputing per epoch)
        self._notch_b, self._notch_a = sp_signal.iirnotch(
            notch_freq, Q=NOTCH_Q, fs=fs
        )

        nyq = fs / 2.0
        low = lowcut / nyq
        high = highcut / nyq
        self._bp_b, self._bp_a = sp_signal.butter(
            FILTER_ORDER, [low, high], btype="band"
        )

        logger.info(f"Preprocessor initialized: {fs} Hz → {target_fs} Hz, "
                    f"band {lowcut}–{highcut} Hz, notch {notch_freq} Hz")

    # ──────────────────────────────────────
    # INDIVIDUAL PROCESSING STEPS
    # ──────────────────────────────────────

    def notch_filter(self, data):
        """
        Apply notch filter to remove power line interference.

        Parameters
        ----------
        data : np.ndarray
            1D signal array.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        return sp_signal.filtfilt(self._notch_b, self._notch_a, data)

    def bandpass_filter(self, data):
        """
        Apply band-pass filter to isolate motor imagery frequency bands.

        Parameters
        ----------
        data : np.ndarray
            1D signal array.

        Returns
        -------
        np.ndarray
            Filtered signal (8–30 Hz by default).
        """
        return sp_signal.filtfilt(self._bp_b, self._bp_a, data)

    def downsample(self, data):
        """
        Downsample signal by the configured factor.

        Parameters
        ----------
        data : np.ndarray
            1D signal array at original sampling rate.

        Returns
        -------
        np.ndarray
            Downsampled signal.
        """
        return data[:: self.downsample_factor]

    def baseline_correct(self, epoch, baseline_samples=BASELINE_SAMPLES):
        """
        Subtract the mean of the initial baseline period.

        Parameters
        ----------
        epoch : np.ndarray
            1D epoch array (at target sampling rate).
        baseline_samples : int
            Number of samples in the baseline window (default: 62 = 0.5s at 125 Hz).

        Returns
        -------
        np.ndarray
            Baseline-corrected epoch.
        """
        if len(epoch) < baseline_samples:
            baseline_samples = max(1, len(epoch) // 4)
        baseline_mean = np.mean(epoch[:baseline_samples])
        return epoch - baseline_mean

    def reject_artifact(self, epoch, threshold_uv=ARTIFACT_THRESHOLD_UV):
        """
        Check if an epoch exceeds the peak-to-peak artifact threshold.

        Parameters
        ----------
        epoch : np.ndarray
            1D epoch array.
        threshold_uv : float
            Maximum allowed peak-to-peak amplitude.

        Returns
        -------
        bool
            True if the epoch is clean (below threshold), False if artifact.
        """
        ptp = np.ptp(epoch)
        return ptp < threshold_uv

    # ──────────────────────────────────────
    # FULL PIPELINE
    # ──────────────────────────────────────

    def preprocess_epoch(self, raw_epoch):
        """
        Apply the full preprocessing pipeline to a single raw epoch.

        Steps: notch → bandpass → downsample → baseline correct.

        Parameters
        ----------
        raw_epoch : np.ndarray
            1D raw signal array at original sampling rate.

        Returns
        -------
        np.ndarray
            Preprocessed epoch at target sampling rate.
        """
        x = np.array(raw_epoch, dtype=np.float64)

        # Minimum length check for filtering
        min_length = max(3 * FILTER_ORDER + 1, 15)
        if len(x) < min_length:
            logger.warning(f"Epoch too short ({len(x)} samples), padding.")
            x = np.pad(x, (0, min_length - len(x)), mode="edge")

        x = self.notch_filter(x)
        x = self.bandpass_filter(x)
        x = self.downsample(x)
        x = self.baseline_correct(x)
        return x

    def preprocess_all(self, raw_trials, labels, target_length=EPOCH_SAMPLES):
        """
        Preprocess a batch of raw trials with artifact rejection.

        Parameters
        ----------
        raw_trials : list or np.ndarray
            List of raw EEG epochs (possibly variable length).
        labels : np.ndarray
            Class labels for each trial.
        target_length : int
            Desired length of each epoch after preprocessing.
            Epochs are padded or truncated to this length.

        Returns
        -------
        clean_epochs : np.ndarray
            Shape (n_clean, target_length) — preprocessed, artifact-free epochs.
        clean_labels : np.ndarray
            Shape (n_clean,) — corresponding labels.
        """
        clean_epochs = []
        clean_labels = []
        rejected = 0

        for i, (trial, label) in enumerate(zip(raw_trials, labels)):
            try:
                epoch = self.preprocess_epoch(trial)

                # Pad or truncate to target length
                if len(epoch) < target_length:
                    epoch = np.pad(epoch, (0, target_length - len(epoch)), mode="edge")
                elif len(epoch) > target_length:
                    epoch = epoch[:target_length]

                # Artifact rejection
                if self.reject_artifact(epoch):
                    clean_epochs.append(epoch)
                    clean_labels.append(label)
                else:
                    rejected += 1
                    logger.debug(f"  Trial {i}: rejected (PTP = {np.ptp(epoch):.1f})")

            except Exception as e:
                rejected += 1
                logger.warning(f"  Trial {i}: error during preprocessing: {e}")

        total = len(raw_trials)
        kept = len(clean_epochs)
        retention = kept / max(total, 1)

        logger.info(f"Preprocessing complete: {kept}/{total} trials retained "
                    f"({retention:.0%}), {rejected} rejected")

        if retention < MIN_TRIAL_RETENTION:
            logger.warning(f"⚠ Retention rate ({retention:.0%}) below threshold "
                          f"({MIN_TRIAL_RETENTION:.0%}). Check signal quality.")

        return np.array(clean_epochs, dtype=np.float32), np.array(clean_labels, dtype=np.int32)


# ──────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────

def preprocess_subject(subject_id, session_id=1, preprocessor=None):
    """
    Load and preprocess raw data for a single subject.

    Parameters
    ----------
    subject_id : int
        Subject number.
    session_id : int
        Session number.
    preprocessor : EEGPreprocessor or None
        Preprocessor instance (created with defaults if None).

    Returns
    -------
    clean_epochs : np.ndarray
    clean_labels : np.ndarray
    """
    if preprocessor is None:
        preprocessor = EEGPreprocessor()

    data_path = RAW_DATA_DIR / f"subject_{subject_id:03d}" / f"session_{session_id:02d}.npz"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None, None

    logger.info(f"Loading subject {subject_id:03d}, session {session_id:02d}...")
    loaded = np.load(data_path, allow_pickle=True)
    raw_trials = loaded["data"]
    labels = loaded["labels"]

    logger.info(f"  Raw: {len(raw_trials)} trials")
    clean_epochs, clean_labels = preprocessor.preprocess_all(raw_trials, labels)
    return clean_epochs, clean_labels


def preprocess_all_subjects():
    """
    Preprocess all subjects found in the raw data directory
    and save a combined clean dataset.

    Saves to: data/preprocessed/all_clean_epochs.npz
    """
    preprocessor = EEGPreprocessor()
    subjects = get_all_subjects()

    if not subjects:
        logger.warning("No subject data found in raw/ directory.")
        logger.info("Use --simulate flag with acquisition.py to generate test data first.")
        return

    all_epochs = []
    all_labels = []
    all_subject_ids = []

    for sid in subjects:
        clean_epochs, clean_labels = preprocess_subject(sid, preprocessor=preprocessor)
        if clean_epochs is not None and len(clean_epochs) > 0:
            all_epochs.append(clean_epochs)
            all_labels.append(clean_labels)
            all_subject_ids.append(np.full(len(clean_labels), sid, dtype=np.int32))

    if not all_epochs:
        logger.error("No valid data after preprocessing.")
        return

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)

    out_path = PREPROCESSED_DIR / "all_clean_epochs.npz"
    np.savez(out_path, X=X, y=y, subject_ids=subject_ids)
    logger.info(f"✓ Saved preprocessed dataset: {out_path}")
    logger.info(f"  Shape: X={X.shape}, y={y.shape}, subjects={np.unique(subject_ids)}")

    return X, y, subject_ids


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BCI EEG Preprocessing Pipeline")
    parser.add_argument("--subject", type=int, help="Preprocess a single subject ID")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Preprocess all subjects and create combined dataset")
    args = parser.parse_args()

    if args.all_subjects:
        preprocess_all_subjects()
    elif args.subject:
        clean, labels = preprocess_subject(args.subject)
        if clean is not None:
            logger.info(f"Result: {clean.shape[0]} clean epochs, shape {clean.shape}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
