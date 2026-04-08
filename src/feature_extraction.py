"""
BCI Assistive Control — Feature Extraction
============================================
Extracts hand-crafted features from preprocessed EEG epochs
for use with classical ML models (SVM). Features include:
  - Alpha band power (8–12 Hz)
  - Beta band power (12–30 Hz)
  - Spectral entropy
  - Signal variance
  - Mean absolute value

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from scipy import signal as sp_signal

from src.utils import (
    TARGET_SAMPLING_RATE, ALPHA_BAND, BETA_BAND,
    setup_logger
)

logger = setup_logger("features")


def extract_band_power(epoch, fs=TARGET_SAMPLING_RATE):
    """
    Extract spectral power features from a single EEG epoch.

    Parameters
    ----------
    epoch : np.ndarray
        1D preprocessed signal at target sampling rate.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Feature vector: [alpha_power, beta_power, entropy, variance, mav].
    """
    # Compute power spectral density (Welch's method)
    nperseg = min(256, len(epoch))
    freqs, psd = sp_signal.welch(epoch, fs=fs, nperseg=nperseg)

    # Band power: average PSD within each band
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    beta_mask  = (freqs > BETA_BAND[0]) & (freqs <= BETA_BAND[1])

    alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0.0
    beta_power  = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0.0

    # Spectral entropy (measure of signal complexity)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # Time-domain features
    variance = np.var(epoch)
    mean_abs = np.mean(np.abs(epoch))

    return np.array([alpha_power, beta_power, entropy, variance, mean_abs])


def extract_features_batch(epochs, fs=TARGET_SAMPLING_RATE):
    """
    Extract features from a batch of epochs.

    Parameters
    ----------
    epochs : np.ndarray
        Shape (n_epochs, n_timepoints).
    fs : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Shape (n_epochs, n_features) — feature matrix.
    """
    features = []
    for i, epoch in enumerate(epochs):
        feat = extract_band_power(epoch, fs)
        features.append(feat)

    features = np.array(features)
    logger.info(f"Extracted features: {features.shape[0]} epochs × "
                f"{features.shape[1]} features")
    return features


def extract_extended_features(epoch, fs=TARGET_SAMPLING_RATE):
    """
    Extract an extended feature set for potentially higher accuracy.

    Adds: alpha/beta ratio, peak frequency, zero-crossing rate,
    signal skewness, and kurtosis.

    Parameters
    ----------
    epoch : np.ndarray
        1D preprocessed signal.
    fs : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Extended feature vector (10 features).
    """
    # Base features
    base = extract_band_power(epoch, fs)
    alpha_power, beta_power = base[0], base[1]

    # Alpha/Beta ratio (ERD indicator)
    ab_ratio = alpha_power / (beta_power + 1e-10)

    # Peak frequency
    nperseg = min(256, len(epoch))
    freqs, psd = sp_signal.welch(epoch, fs=fs, nperseg=nperseg)
    peak_freq = freqs[np.argmax(psd)]

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(epoch)) != 0)
    zcr = zero_crossings / len(epoch)

    # Statistical moments
    skewness = float(np.mean(((epoch - np.mean(epoch)) / (np.std(epoch) + 1e-10)) ** 3))
    kurtosis = float(np.mean(((epoch - np.mean(epoch)) / (np.std(epoch) + 1e-10)) ** 4))

    return np.concatenate([base, [ab_ratio, peak_freq, zcr, skewness, kurtosis]])


FEATURE_NAMES = [
    "alpha_power", "beta_power", "spectral_entropy",
    "variance", "mean_abs_value"
]

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "alpha_beta_ratio", "peak_frequency", "zero_crossing_rate",
    "skewness", "kurtosis"
]


if __name__ == "__main__":
    # Quick test with random data
    test_epoch = np.random.randn(500)
    feats = extract_band_power(test_epoch)
    print("Features:", dict(zip(FEATURE_NAMES, feats)))

    ext_feats = extract_extended_features(test_epoch)
    print("Extended:", dict(zip(EXTENDED_FEATURE_NAMES, ext_feats)))
