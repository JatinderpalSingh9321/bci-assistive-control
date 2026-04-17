"""
BCI Assistive Control — Blink/Wink Training Pipeline (1-Channel Fp1)
=====================================================================
Trains an EOG classifier from data collected by test_blink_wink.py.

  3 classes: BLINK, WINK_LEFT, WINK_RIGHT
  Input:     1-channel Fp1 referential (forehead to earlobe)

Usage:
  python -m src.train_blink_wink --subject 1
  python -m src.train_blink_wink --subject 1 --grid-search

Group No. 7 | 8th Semester Major Project
"""

import argparse
import json
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

logger = setup_logger("train_bw")

CLASS_NAMES = ["BLINK", "WINK_LEFT", "WINK_RIGHT"]


# ──────────────────────────────────────────────
# FEATURE EXTRACTION (1-channel Fp1)
# ──────────────────────────────────────────────

def extract_bw_features(epoch_1d, fs=SAMPLING_RATE):
    """
    Extract blink/wink features from single-channel Fp1 signal.

    Parameters
    ----------
    epoch_1d : np.ndarray
        1D array of Fp1 values.

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    sig = epoch_1d.astype(np.float64)

    # Notch filter (50 Hz power line)
    b, a = scipy_signal.iirnotch(50.0, 30, fs)
    sig = scipy_signal.filtfilt(b, a, sig)

    # Remove DC offset (baseline correction)
    sig = sig - np.mean(sig)

    features = []

    # ── Amplitude features ──
    abs_sig = np.abs(sig)

    # 1. Peak amplitude (1)
    features.append(np.max(abs_sig))

    # 2. Mean amplitude (1)
    features.append(np.mean(abs_sig))

    # 3. RMS (1)
    features.append(np.sqrt(np.mean(sig ** 2)))

    # 4. Variance (1)
    features.append(np.var(sig))

    # 5. Standard deviation (1)
    features.append(np.std(sig))

    # ── Peak analysis ──
    threshold = np.mean(abs_sig) + 2 * np.std(sig)

    # 6. Number of peaks (1)
    peaks, props = scipy_signal.find_peaks(
        abs_sig, height=threshold, distance=int(0.2 * fs)
    )
    features.append(len(peaks))

    # 7. Average peak height (1)
    if len(peaks) > 0:
        features.append(np.mean(props["peak_heights"]))
    else:
        features.append(0.0)

    # 8. Max peak height (1)
    if len(peaks) > 0:
        features.append(np.max(props["peak_heights"]))
    else:
        features.append(0.0)

    # 9. Average peak width (1)
    if len(peaks) > 0:
        widths = scipy_signal.peak_widths(abs_sig, peaks)[0]
        features.append(np.mean(widths))
    else:
        features.append(0.0)

    # 10. Peak width std (1)
    if len(peaks) > 0:
        widths = scipy_signal.peak_widths(abs_sig, peaks)[0]
        features.append(np.std(widths))
    else:
        features.append(0.0)

    # ── Statistical features ──
    # 11. Kurtosis (1) — blinks are peaky
    features.append(kurtosis(sig))

    # 12. Skewness (1) — asymmetry of signal
    features.append(skew(sig))

    # ── Energy features ──
    # 13-14. Energy: first half vs second half (2)
    mid = len(sig) // 2
    e1 = np.sum(sig[:mid] ** 2)
    e2 = np.sum(sig[mid:] ** 2)
    features.append(e1 / (e1 + e2 + 1e-10))  # First-half energy ratio
    features.append(e1 / (e2 + 1e-10))        # Energy ratio

    # 15-16. Energy: thirds (2) — where in the epoch does the action happen?
    third = len(sig) // 3
    e_t1 = np.sum(sig[:third] ** 2)
    e_t2 = np.sum(sig[third:2*third] ** 2)
    e_t3 = np.sum(sig[2*third:] ** 2)
    total_e = e_t1 + e_t2 + e_t3 + 1e-10
    features.append(e_t1 / total_e)
    features.append(e_t2 / total_e)

    # ── Frequency features ──
    # 17-20. Band power: DC-2Hz, 2-8Hz, 8-15Hz, 15-30Hz (4)
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    bands = [(0.5, 2), (2, 8), (8, 15), (15, 30)]
    for fmin, fmax in bands:
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            features.append(np.mean(psd[mask]))
        else:
            features.append(0.0)

    # 21. Dominant frequency (1)
    if len(psd) > 0:
        features.append(freqs[np.argmax(psd)])
    else:
        features.append(0.0)

    # ── Waveform shape features ──
    # 22. Zero crossing rate (1)
    zcr = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
    features.append(zcr)

    # 23. Signal range (peak-to-peak) (1)
    features.append(np.ptp(sig))

    # 24. Positive peak / negative peak ratio (1) — wink direction indicator
    pos_peak = np.max(sig) if np.max(sig) > 0 else 1e-10
    neg_peak = np.abs(np.min(sig)) if np.min(sig) < 0 else 1e-10
    features.append(pos_peak / (neg_peak + 1e-10))

    # 25. Hjorth mobility (1)
    d1 = np.diff(sig)
    mobility = np.std(d1) / (np.std(sig) + 1e-10)
    features.append(mobility)

    # 26. Hjorth complexity (1)
    d2 = np.diff(d1)
    mobility_d1 = np.std(d2) / (np.std(d1) + 1e-10)
    features.append(mobility_d1 / (mobility + 1e-10))

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_blink_wink_data(subject_id, session_id=1):
    """Load blink/wink data collected by test_blink_wink.py."""
    data_dir = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    data_path = data_dir / f"session_{session_id:02d}_blink_wink.npz"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run first: python -m src.test_blink_wink --subject 1 --port COM7")
        return None, None

    loaded = np.load(data_path, allow_pickle=True)
    data = loaded["data"]
    labels = loaded["labels"]

    logger.info(f"✓ Loaded {len(labels)} trials from {data_path}")
    for cid in range(3):
        count = np.sum(labels == cid)
        logger.info(f"    {CLASS_NAMES[cid]:12s}: {count} trials")

    return data, labels


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train(data, labels, do_grid_search=False):
    """Train SVM on blink/wink features."""

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING BLINK/WINK MODEL (3-class)")
    logger.info("=" * 60)

    # Extract features
    logger.info("  Extracting features...")
    X = []
    valid_idx = []

    for i, epoch in enumerate(data):
        try:
            arr = np.array(epoch, dtype=np.float32).flatten()
            if len(arr) < 50:
                logger.warning(f"  Skipping trial {i}: too short ({len(arr)} samples)")
                continue

            feats = extract_bw_features(arr)

            if np.all(np.isfinite(feats)):
                X.append(feats)
                valid_idx.append(i)
            else:
                logger.warning(f"  Skipping trial {i}: non-finite features")
        except Exception as e:
            logger.warning(f"  Failed trial {i}: {e}")

    X = np.array(X, dtype=np.float32)
    y = labels[valid_idx]

    logger.info(f"  Feature matrix: {X.shape} ({X.shape[1]} features × {X.shape[0]} trials)")
    logger.info(f"  Valid trials: {len(y)} / {len(labels)}")

    # Train SVM
    if do_grid_search:
        logger.info("  Running GridSearchCV (this may take a minute)...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42)),
        ])
        param_grid = {
            "svm__C": [0.1, 1, 10, 50, 100],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
            "svm__kernel": ["rbf", "linear"],
        }
        n_splits = min(5, np.min(np.bincount(y)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="accuracy", n_jobs=-1, verbose=0
        )
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

    # Cross-validation
    n_splits = min(5, np.min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

    # Full training accuracy
    y_pred = best_model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Results
    logger.info(f"\n  Training Accuracy: {train_acc:.1%}")
    logger.info(f"  CV Accuracy:       {cv_mean:.1%} ± {cv_std:.1%}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(y, y_pred)
    logger.info("  Confusion Matrix:")
    logger.info(f"             {'  '.join(f'{n:>10s}' for n in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        logger.info(f"  {CLASS_NAMES[i]:12s} {'  '.join(f'{v:>10d}' for v in row)}")

    # Save model (compatible with mouse_control_eog.py)
    model_path = MODELS_DIR / "eog_3class_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "class_names": CLASS_NAMES,
            "n_features": X.shape[1],
            "train_accuracy": train_acc,
            "cv_accuracy": float(cv_mean),
            "cv_std": float(cv_std),
            "feature_extractor": "extract_bw_features",
            "source": "train_blink_wink.py",
            "data_type": "1ch_fp1_referential",
        }, f)
    logger.info(f"\n  ✓ Model saved: {model_path}")

    # Save results
    results = {
        "train_accuracy": float(train_acc),
        "cv_accuracy": float(cv_mean),
        "cv_std": float(cv_std),
        "cv_scores": [float(s) for s in cv_scores],
        "n_trials": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {CLASS_NAMES[c]: int(np.sum(y == c)) for c in range(3)},
        "report": classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    results_path = RESULTS_DIR / "blink_wink_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✓ Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Training Accuracy: {train_acc:.1%}")
    print(f"  CV Accuracy:       {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"  Model saved:       {model_path}")
    print()
    print("  Next steps:")
    print("    python -m src.mouse_control_eog --port COM7")
    print("=" * 60)

    return results


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Blink/Wink EOG Model (1-Channel Fp1)"
    )
    parser.add_argument("--subject", type=int, required=True,
                        help="Subject ID")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number (default: 1)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run GridSearchCV for best hyperparameters")
    args = parser.parse_args()

    data, labels = load_blink_wink_data(args.subject, args.session)
    if data is None:
        return

    train(data, labels, do_grid_search=args.grid_search)


if __name__ == "__main__":
    main()
