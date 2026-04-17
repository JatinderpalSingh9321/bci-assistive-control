"""
BCI Assistive Control — 4-Channel Training Pipeline
=====================================================
Trains two models from 4-channel data collected by experiment_4ch.py:

  Model 1: Motor Imagery Classifier (C3/C4 channels)
    - 4 classes: LEFT, RIGHT, UP, DOWN
    - Features: band power, asymmetry ratios, CSP-like features

  Model 2: EOG Classifier (Fp1/Fp2 channels)
    - 3 classes: BLINK, WINK_LEFT, WINK_RIGHT
    - Features: peak amplitude, asymmetry, waveform shape

Usage:
  python -m src.train_4ch --subject 1 --session 1
  python -m src.train_4ch --simulate
  python -m src.train_4ch --subject 1 --session 1 --block mi

Group No. 7 | 8th Semester Major Project
"""

import argparse
import json
import time
import pickle
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from src.utils import (
    SAMPLING_RATE, RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    setup_logger
)

logger = setup_logger("train_4ch")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

MI_CLASS_NAMES = ["LEFT", "RIGHT", "UP", "DOWN"]
EOG_CLASS_NAMES = ["BLINK", "WINK_LEFT", "WINK_RIGHT"]

FREQ_BANDS = {
    "delta":    (0.5, 4),
    "theta":    (4, 8),
    "alpha_mu": (8, 12),
    "beta_low": (12, 20),
    "beta_high": (20, 30),
    "gamma":    (30, 45),
}


# ──────────────────────────────────────────────
# SIGNAL PROCESSING
# ──────────────────────────────────────────────

def bandpass_filter(data, low, high, fs=SAMPLING_RATE, order=4):
    """Apply bandpass Butterworth filter."""
    nyq = fs / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low_n, high_n], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0)


def notch_filter(data, freq=50.0, fs=SAMPLING_RATE, Q=30):
    """Apply notch filter for power line noise."""
    b, a = scipy_signal.iirnotch(freq, Q, fs)
    return scipy_signal.filtfilt(b, a, data, axis=0)


def compute_band_power(signal_1d, fs=SAMPLING_RATE):
    """Compute power in each frequency band using Welch's method."""
    freqs, psd = scipy_signal.welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            powers[band_name] = np.mean(psd[mask])
        else:
            powers[band_name] = 0.0
    return powers


# ──────────────────────────────────────────────
# FEATURE EXTRACTION: MOTOR IMAGERY (C3/C4)
# ──────────────────────────────────────────────

def extract_mi_features(epoch_4ch, fs=SAMPLING_RATE):
    """
    Extract motor imagery features from C3/C4 channels.

    Parameters
    ----------
    epoch_4ch : np.ndarray
        Shape (n_samples, 4) — [C3, C4, Fp1, Fp2].

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    if epoch_4ch.ndim == 1 or epoch_4ch.shape[1] < 2:
        # Single channel fallback
        c3 = epoch_4ch[:, 0] if epoch_4ch.ndim > 1 else epoch_4ch
        c4 = c3.copy()
    else:
        c3 = epoch_4ch[:, 0]
        c4 = epoch_4ch[:, 1]

    # Filter to motor-relevant range
    c3_filtered = bandpass_filter(c3, 0.5, 45, fs)
    c4_filtered = bandpass_filter(c4, 0.5, 45, fs)
    c3_filtered = notch_filter(c3_filtered, 50, fs)
    c4_filtered = notch_filter(c4_filtered, 50, fs)

    features = []

    # 1. Band power for each channel (6 bands × 2 channels = 12)
    for ch_data, ch_name in [(c3_filtered, "C3"), (c4_filtered, "C4")]:
        powers = compute_band_power(ch_data, fs)
        for band_name in FREQ_BANDS:
            features.append(powers[band_name])

    # 2. Asymmetry ratios per band (C3-C4)/(C3+C4) — key for L/R (6 features)
    c3_powers = compute_band_power(c3_filtered, fs)
    c4_powers = compute_band_power(c4_filtered, fs)
    for band_name in FREQ_BANDS:
        p3 = c3_powers[band_name]
        p4 = c4_powers[band_name]
        ratio = (p3 - p4) / (p3 + p4 + 1e-10)
        features.append(ratio)

    # 3. Log band power ratio (C3/C4) per band (6 features)
    for band_name in FREQ_BANDS:
        p3 = c3_powers[band_name]
        p4 = c4_powers[band_name]
        features.append(np.log1p(p3) - np.log1p(p4))

    # 4. Temporal statistics per channel (4 × 2 = 8)
    for ch_data in [c3_filtered, c4_filtered]:
        features.append(np.var(ch_data))
        features.append(kurtosis(ch_data))
        features.append(skew(ch_data))
        features.append(np.sqrt(np.mean(ch_data ** 2)))  # RMS

    # 5. Cross-channel features (3 features)
    # Correlation between C3 and C4
    if len(c3_filtered) > 1:
        corr = np.corrcoef(c3_filtered, c4_filtered)[0, 1]
        features.append(corr if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Power difference (total)
    features.append(np.var(c3_filtered) - np.var(c4_filtered))

    # Phase coherence (alpha band)
    c3_alpha = bandpass_filter(c3_filtered, 8, 12, fs)
    c4_alpha = bandpass_filter(c4_filtered, 8, 12, fs)
    if len(c3_alpha) > 1:
        phase_diff = np.angle(scipy_signal.hilbert(c3_alpha)) - \
                     np.angle(scipy_signal.hilbert(c4_alpha))
        coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
        features.append(coherence)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# FEATURE EXTRACTION: EOG (Fp1/Fp2)
# ──────────────────────────────────────────────

def extract_eog_features(epoch_4ch, fs=SAMPLING_RATE):
    """
    Extract eye-movement features from Fp1/Fp2 channels.

    Parameters
    ----------
    epoch_4ch : np.ndarray
        Shape (n_samples, 4) — [C3, C4, Fp1, Fp2].

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    if epoch_4ch.shape[1] < 4:
        fp1 = epoch_4ch[:, min(2, epoch_4ch.shape[1] - 1)]
        fp2 = fp1.copy()
    else:
        fp1 = epoch_4ch[:, 2]
        fp2 = epoch_4ch[:, 3]

    # Light filtering (keep DC and low-freq for EOG)
    fp1_f = notch_filter(fp1, 50, fs)
    fp2_f = notch_filter(fp2, 50, fs)

    features = []

    # 1. Peak amplitude per channel (2)
    features.append(np.max(np.abs(fp1_f)))
    features.append(np.max(np.abs(fp2_f)))

    # 2. Mean amplitude (2)
    features.append(np.mean(np.abs(fp1_f)))
    features.append(np.mean(np.abs(fp2_f)))

    # 3. Variance (2)
    features.append(np.var(fp1_f))
    features.append(np.var(fp2_f))

    # 4. Asymmetry: (Fp1 - Fp2) / (Fp1 + Fp2) for peak (1)
    pk1 = np.max(np.abs(fp1_f))
    pk2 = np.max(np.abs(fp2_f))
    features.append((pk1 - pk2) / (pk1 + pk2 + 1e-10))

    # 5. Asymmetry: variance ratio (1)
    v1 = np.var(fp1_f)
    v2 = np.var(fp2_f)
    features.append((v1 - v2) / (v1 + v2 + 1e-10))

    # 6. Number of peaks (blink = multiple peaks) (2)
    for ch in [fp1_f, fp2_f]:
        threshold = np.mean(np.abs(ch)) + 2 * np.std(ch)
        peaks, _ = scipy_signal.find_peaks(np.abs(ch), height=threshold,
                                            distance=int(0.3 * fs))
        features.append(len(peaks))

    # 7. Peak width (average) (2)
    for ch in [fp1_f, fp2_f]:
        threshold = np.mean(np.abs(ch)) + 2 * np.std(ch)
        peaks, props = scipy_signal.find_peaks(np.abs(ch), height=threshold,
                                                distance=int(0.3 * fs))
        if len(peaks) > 0:
            widths = scipy_signal.peak_widths(np.abs(ch), peaks)[0]
            features.append(np.mean(widths))
        else:
            features.append(0.0)

    # 8. Kurtosis and skewness per channel (4)
    features.append(kurtosis(fp1_f))
    features.append(kurtosis(fp2_f))
    features.append(skew(fp1_f))
    features.append(skew(fp2_f))

    # 9. Energy ratio: first half vs second half (2)
    mid = len(fp1_f) // 2
    for ch in [fp1_f, fp2_f]:
        e1 = np.sum(ch[:mid] ** 2)
        e2 = np.sum(ch[mid:] ** 2)
        features.append(e1 / (e2 + 1e-10))

    # 10. Cross-correlation peak between Fp1 and Fp2 (1)
    if len(fp1_f) > 10:
        cross_corr = np.correlate(fp1_f - np.mean(fp1_f),
                                   fp2_f - np.mean(fp2_f), mode='full')
        cross_corr /= (np.std(fp1_f) * np.std(fp2_f) * len(fp1_f) + 1e-10)
        features.append(np.max(cross_corr))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_4ch_data(subject_id, session_id=1, block="all"):
    """Load 4-channel data collected by experiment_4ch.py."""
    data_dir = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    data_path = data_dir / f"session_{session_id:02d}_4ch_{block}.npz"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None, None

    loaded = np.load(data_path, allow_pickle=True)
    data = loaded["data"]
    labels = loaded["labels"]

    logger.info(f"Loaded {len(labels)} trials from {data_path}")
    return data, labels


def generate_simulated_training_data(mi_trials=25, eog_trials=20):
    """Generate simulated 4-channel data for pipeline testing."""
    from src.experiment_4ch import generate_simulated_4ch

    all_data = []
    all_labels = []
    n_samples = int(4.0 * SAMPLING_RATE)  # 4 seconds

    # Motor imagery: 4 classes
    for class_id in range(4):
        for _ in range(mi_trials):
            epoch = generate_simulated_4ch(class_id, n_samples)
            # Add subject variability
            epoch += np.random.randn() * 2
            epoch *= (0.8 + 0.4 * np.random.rand())
            all_data.append(epoch)
            all_labels.append(class_id)

    # EOG: 3 classes (labels 4, 5, 6 → remap to 0, 1, 2 for EOG model)
    for class_id in range(4, 7):
        for _ in range(eog_trials):
            epoch = generate_simulated_4ch(class_id, n_samples)
            epoch += np.random.randn() * 1
            all_data.append(epoch)
            all_labels.append(class_id)

    return np.array(all_data, dtype=object), np.array(all_labels, dtype=np.int32)


# ──────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────

def train_mi_model(data, labels, do_grid_search=False):
    """
    Train Motor Imagery SVM (4-class: LEFT/RIGHT/UP/DOWN).

    Parameters
    ----------
    data : array of np.ndarray
        Each element is (n_samples, 4) — 4-channel epoch.
    labels : np.ndarray
        Class labels (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN).

    Returns
    -------
    dict
        Model pipeline, accuracy, classification report.
    """
    logger.info("=" * 60)
    logger.info("  TRAINING MOTOR IMAGERY MODEL (4-class)")
    logger.info("=" * 60)

    # Filter for MI classes only (labels 0-3)
    mi_mask = labels < 4
    mi_data = data[mi_mask]
    mi_labels = labels[mi_mask]

    logger.info(f"  MI trials: {len(mi_labels)}")
    for cid in range(4):
        count = np.sum(mi_labels == cid)
        logger.info(f"    {MI_CLASS_NAMES[cid]:6s}: {count} trials")

    # Extract features
    logger.info("  Extracting MI features...")
    X = []
    valid_idx = []
    for i, epoch in enumerate(mi_data):
        try:
            epoch_arr = np.array(epoch, dtype=np.float32)
            if epoch_arr.ndim == 1:
                epoch_arr = epoch_arr.reshape(-1, 1)
            if epoch_arr.shape[0] < 50:
                logger.warning(f"    Skipping trial {i}: too short ({epoch_arr.shape[0]} samples)")
                continue
            features = extract_mi_features(epoch_arr)
            if np.all(np.isfinite(features)):
                X.append(features)
                valid_idx.append(i)
        except Exception as e:
            logger.warning(f"    Failed trial {i}: {e}")

    X = np.array(X, dtype=np.float32)
    y = mi_labels[valid_idx]
    logger.info(f"  Feature matrix: {X.shape}")

    # Train SVM
    if do_grid_search:
        logger.info("  Running GridSearchCV...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42)),
        ])
        param_grid = {
            "svm__C": [0.1, 1, 10, 50],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
            "svm__kernel": ["rbf", "linear"],
        }
        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy",
                           n_jobs=-1, verbose=0)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {grid.best_score_:.4f}")
    else:
        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10, gamma="scale", kernel="rbf",
                       probability=True, random_state=42)),
        ])
        best_model.fit(X, y)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

    # Full training accuracy
    y_pred = best_model.predict(X)
    train_acc = accuracy_score(y, y_pred)

    logger.info(f"  Training accuracy: {train_acc:.4f}")
    logger.info(f"  CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=MI_CLASS_NAMES)}")

    # Save model
    model_path = MODELS_DIR / "mi_4class_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "class_names": MI_CLASS_NAMES,
            "n_features": X.shape[1],
            "train_accuracy": train_acc,
            "cv_accuracy": float(np.mean(cv_scores)),
        }, f)
    logger.info(f"  ✓ MI model saved: {model_path}")

    return {
        "model": best_model,
        "accuracy": train_acc,
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "report": classification_report(y, y_pred, target_names=MI_CLASS_NAMES, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def train_eog_model(data, labels, do_grid_search=False):
    """
    Train EOG Classifier (3-class: BLINK/WINK_LEFT/WINK_RIGHT).

    Parameters
    ----------
    data : array of np.ndarray
        Each element is (n_samples, 4) — 4-channel epoch.
    labels : np.ndarray
        Class labels (4=BLINK, 5=WINK_LEFT, 6=WINK_RIGHT).

    Returns
    -------
    dict
        Model pipeline, accuracy, classification report.
    """
    logger.info("=" * 60)
    logger.info("  TRAINING EOG MODEL (3-class)")
    logger.info("=" * 60)

    # Filter for EOG classes only (labels 4-6)
    eog_mask = labels >= 4
    eog_data = data[eog_mask]
    eog_labels = labels[eog_mask] - 4  # Remap to 0, 1, 2

    logger.info(f"  EOG trials: {len(eog_labels)}")
    for cid in range(3):
        count = np.sum(eog_labels == cid)
        logger.info(f"    {EOG_CLASS_NAMES[cid]:12s}: {count} trials")

    # Extract features
    logger.info("  Extracting EOG features...")
    X = []
    valid_idx = []
    for i, epoch in enumerate(eog_data):
        try:
            epoch_arr = np.array(epoch, dtype=np.float32)
            if epoch_arr.ndim == 1:
                epoch_arr = epoch_arr.reshape(-1, 1)
            if epoch_arr.shape[0] < 50:
                continue
            features = extract_eog_features(epoch_arr)
            if np.all(np.isfinite(features)):
                X.append(features)
                valid_idx.append(i)
        except Exception as e:
            logger.warning(f"    Failed trial {i}: {e}")

    X = np.array(X, dtype=np.float32)
    y = eog_labels[valid_idx]
    logger.info(f"  Feature matrix: {X.shape}")

    # Train SVM
    if do_grid_search:
        logger.info("  Running GridSearchCV...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42)),
        ])
        param_grid = {
            "svm__C": [0.1, 1, 10, 50],
            "svm__gamma": ["scale", "auto", 0.01],
            "svm__kernel": ["rbf", "linear"],
        }
        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy",
                           n_jobs=-1, verbose=0)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {grid.best_score_:.4f}")
    else:
        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10, gamma="scale", kernel="rbf",
                       probability=True, random_state=42)),
        ])
        best_model.fit(X, y)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

    y_pred = best_model.predict(X)
    train_acc = accuracy_score(y, y_pred)

    logger.info(f"  Training accuracy: {train_acc:.4f}")
    logger.info(f"  CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=EOG_CLASS_NAMES)}")

    # Save model
    model_path = MODELS_DIR / "eog_3class_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "class_names": EOG_CLASS_NAMES,
            "n_features": X.shape[1],
            "train_accuracy": train_acc,
            "cv_accuracy": float(np.mean(cv_scores)),
        }, f)
    logger.info(f"  ✓ EOG model saved: {model_path}")

    return {
        "model": best_model,
        "accuracy": train_acc,
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "report": classification_report(y, y_pred, target_names=EOG_CLASS_NAMES, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BCI 4-Channel Model Training")
    parser.add_argument("--subject", type=int, help="Subject ID")
    parser.add_argument("--session", type=int, default=1, help="Session number")
    parser.add_argument("--simulate", action="store_true",
                        help="Train on simulated data")
    parser.add_argument("--block", type=str, default="all",
                        choices=["mi", "eog", "all"],
                        help="Which model to train")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run GridSearchCV for hyperparameter tuning")
    parser.add_argument("--mi-trials", type=int, default=40,
                        help="Simulated MI trials per class")
    parser.add_argument("--eog-trials", type=int, default=30,
                        help="Simulated EOG trials per class")
    args = parser.parse_args()

    from src.utils import print_banner
    print_banner()
    print("  >> 4-CHANNEL MODEL TRAINING\n")

    # Load data
    if args.simulate:
        logger.info("Generating simulated 4-channel data...")
        data, labels = generate_simulated_training_data(
            mi_trials=args.mi_trials,
            eog_trials=args.eog_trials
        )
    else:
        if args.subject is None:
            logger.error("--subject is required when not using --simulate")
            return
        data, labels = load_4ch_data(args.subject, args.session, args.block)
        if data is None:
            return

    results = {}

    # Train MI model
    if args.block in ("mi", "all"):
        mi_results = train_mi_model(data, labels, do_grid_search=args.grid_search)
        results["mi"] = mi_results

    # Train EOG model
    if args.block in ("eog", "all"):
        eog_results = train_eog_model(data, labels, do_grid_search=args.grid_search)
        results["eog"] = eog_results

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    if "mi" in results:
        print(f"  Motor Imagery (4-class): {results['mi']['accuracy']:.1%} train, "
              f"{results['mi']['cv_accuracy']:.1%} ± {results['mi']['cv_std']:.1%} CV")
    if "eog" in results:
        print(f"  EOG Detection (3-class): {results['eog']['accuracy']:.1%} train, "
              f"{results['eog']['cv_accuracy']:.1%} ± {results['eog']['cv_std']:.1%} CV")
    print(f"\n  Models saved to: {MODELS_DIR}")
    print("=" * 60)

    # Save results JSON
    results_path = RESULTS_DIR / "4ch_training_results.json"
    serializable = {}
    for key in results:
        serializable[key] = {k: v for k, v in results[key].items() if k != "model"}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
