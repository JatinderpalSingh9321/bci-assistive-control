"""
BCI Assistive Control — Training Pipeline (1-Channel Referential)
==================================================================
Trains a 4-class motor imagery SVM from SINGLE referential C3 channel.

Uses the SAME data format as experiment_referential.py:
  - 1D EEG signal per trial (C3 referenced to earlobe)
  - Labels: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN

Feature Strategy for 4-class from 1 channel:
  LEFT vs RIGHT  → Mu/Beta power differences (ERD strength at C3)
  UP   (tongue)  → High-beta burst (25-30 Hz increase)
  DOWN (feet)    → Theta increase (4-8 Hz), different from mu pattern

Usage:
  python -m src.train_referential --simulate
  python -m src.train_referential --subject 1 --session 1
  python -m src.train_referential --simulate --grid-search

Group No. 7 | 8th Semester Major Project
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis, skew, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from src.utils import SAMPLING_RATE, RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, setup_logger

logger = setup_logger("train_ref")

MI_NAMES = ["LEFT", "RIGHT", "UP", "DOWN"]

BANDS = {
    "delta":     (0.5, 4),
    "theta":     (4, 8),
    "low_alpha": (8, 10),
    "high_alpha": (10, 12),
    "low_beta":  (12, 20),
    "high_beta": (20, 30),
    "gamma":     (30, 45),
}


# ──────────────────────────────────────────────
# FEATURE EXTRACTION (single C3 channel)
# ──────────────────────────────────────────────

def bandpass(data, lo, hi, fs=SAMPLING_RATE, order=4):
    """Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = sig.butter(order, [max(lo/nyq, 0.001), min(hi/nyq, 0.999)], btype='band')
    return sig.filtfilt(b, a, data)


def extract_features(epoch_1d, fs=SAMPLING_RATE):
    """
    Extract comprehensive features from a single C3 referential epoch.

    Returns ~50 features designed to discriminate 4 MI classes.
    """
    # Preprocess
    x = epoch_1d.copy().astype(np.float64)
    # Remove DC
    x -= np.mean(x)
    # Notch filter 50 Hz
    b, a = sig.iirnotch(50.0, 30.0, fs)
    x = sig.filtfilt(b, a, x)
    # Bandpass 0.5-45 Hz
    x = bandpass(x, 0.5, 45, fs)

    features = []

    # ─── 1. Band Power (7 bands) ───
    freqs, psd = sig.welch(x, fs=fs, nperseg=min(256, len(x)))
    total_power = np.sum(psd) + 1e-10

    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        bp = np.mean(psd[mask]) if np.any(mask) else 0.0
        features.append(bp)                           # Absolute power
        features.append(bp / total_power)              # Relative power

    # ─── 2. Band Power Ratios (key discriminators) ───
    powers = {}
    for bname, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        powers[bname] = np.mean(psd[mask]) if np.any(mask) else 1e-10

    # Theta/Alpha ratio (UP/DOWN vs LEFT/RIGHT)
    features.append(powers["theta"] / (powers["low_alpha"] + powers["high_alpha"] + 1e-10))
    # Beta/Alpha ratio (RIGHT ERD indicator)
    features.append((powers["low_beta"] + powers["high_beta"]) /
                     (powers["low_alpha"] + powers["high_alpha"] + 1e-10))
    # High-beta / Low-beta (UP tongue indicator)
    features.append(powers["high_beta"] / (powers["low_beta"] + 1e-10))
    # Theta/Beta ratio (DOWN feet indicator)
    features.append(powers["theta"] / (powers["low_beta"] + 1e-10))
    # Alpha peak frequency
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    if np.any(alpha_mask) and np.sum(psd[alpha_mask]) > 0:
        features.append(np.average(freqs[alpha_mask], weights=psd[alpha_mask]))
    else:
        features.append(10.0)

    # ─── 3. Temporal Statistics ───
    features.append(np.var(x))
    features.append(np.std(x))
    features.append(kurtosis(x))
    features.append(skew(x))
    features.append(np.sqrt(np.mean(x**2)))                 # RMS
    features.append(np.max(np.abs(x)))                       # Peak amplitude
    features.append(np.sum(np.abs(np.diff(x))))              # Line length
    features.append(np.sum(np.diff(np.sign(x)) != 0))       # Zero crossings

    # ─── 4. Hjorth Parameters ───
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x) + 1e-10
    var_dx = np.var(dx) + 1e-10
    var_ddx = np.var(ddx) + 1e-10

    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-10)
    features.extend([activity, mobility, complexity])

    # ─── 5. Sub-band Energies (sliding windows) ───
    # Split epoch into 4 quarters and compute mu-band energy in each
    quarter = len(x) // 4
    for q in range(4):
        segment = x[q*quarter : (q+1)*quarter]
        if len(segment) > 20:
            mu_seg = bandpass(segment, 8, 12, fs)
            features.append(np.var(mu_seg))
        else:
            features.append(0.0)

    # ─── 6. ERD/ERS Approximation ───
    # Compare first half (baseline) to second half (imagery) in mu band
    mid = len(x) // 2
    if mid > 20:
        mu_first = bandpass(x[:mid], 8, 12, fs)
        mu_second = bandpass(x[mid:], 8, 12, fs)
        erd_mu = (np.var(mu_second) - np.var(mu_first)) / (np.var(mu_first) + 1e-10)
        features.append(erd_mu)

        beta_first = bandpass(x[:mid], 12, 30, fs)
        beta_second = bandpass(x[mid:], 12, 30, fs)
        erd_beta = (np.var(beta_second) - np.var(beta_first)) / (np.var(beta_first) + 1e-10)
        features.append(erd_beta)

        theta_first = bandpass(x[:mid], 4, 8, fs)
        theta_second = bandpass(x[mid:], 4, 8, fs)
        erd_theta = (np.var(theta_second) - np.var(theta_first)) / (np.var(theta_first) + 1e-10)
        features.append(erd_theta)
    else:
        features.extend([0.0, 0.0, 0.0])

    # ─── 7. Spectral Entropy ───
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    features.append(entropy(psd_norm))

    # ─── 8. Wavelet Energy (if enough samples) ───
    try:
        from scipy.signal import cwt, morlet2
        widths = np.arange(1, 31)
        cwtm = cwt(x[:min(500, len(x))], morlet2, widths)
        for w_idx in [4, 9, 14, 24]:  # ~theta, alpha, beta, gamma
            if w_idx < len(cwtm):
                features.append(np.mean(np.abs(cwtm[w_idx])**2))
            else:
                features.append(0.0)
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(subject_id, session_id=1, block="all"):
    """Load referential data from experiment_referential.py."""
    path = RAW_DATA_DIR / f"subject_{subject_id:03d}" / f"session_{session_id:02d}_ref_{block}.npz"
    if not path.exists():
        logger.error(f"Not found: {path}")
        return None, None
    loaded = np.load(path, allow_pickle=True)
    return loaded["data"], loaded["labels"]


def generate_simulated(n_per_class=40):
    """Generate simulated referential C3 data for all 4 MI classes."""
    from src.experiment_referential import simulate_referential_epoch

    all_data = []
    all_labels = []
    n_samples = int(4.0 * SAMPLING_RATE)

    for cid in range(4):
        for _ in range(n_per_class):
            epoch = simulate_referential_epoch(cid, n_samples)
            epoch += np.random.randn() * 1.5
            epoch *= (0.85 + 0.3 * np.random.rand())
            all_data.append(epoch)
            all_labels.append(cid)

    return np.array(all_data, dtype=object), np.array(all_labels, dtype=np.int32)


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train(data, labels, do_grid_search=False):
    """
    Train 4-class MI SVM from single referential C3 channel.
    """
    logger.info("=" * 60)
    logger.info("  TRAINING 4-CLASS MI MODEL (Referential C3)")
    logger.info("=" * 60)

    # Filter MI classes only (0-3)
    mi_mask = labels < 4
    data = data[mi_mask]
    labels = labels[mi_mask]

    for cid in range(4):
        logger.info(f"  {MI_NAMES[cid]:6s}: {np.sum(labels == cid)} trials")

    # Extract features
    logger.info("  Extracting features...")
    X, valid = [], []
    for i, epoch in enumerate(data):
        try:
            arr = np.array(epoch, dtype=np.float32)
            if len(arr) < 100:
                continue
            feats = extract_features(arr)
            if np.all(np.isfinite(feats)):
                X.append(feats)
                valid.append(i)
        except Exception as e:
            logger.warning(f"  Trial {i} failed: {e}")

    X = np.array(X, dtype=np.float32)
    y = labels[valid]
    logger.info(f"  Features: {X.shape} ({X.shape[1]} per trial)")

    # Pipeline: Scale → SelectKBest → PCA → SVM
    if do_grid_search:
        logger.info("  Running GridSearchCV...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k="all")),
            ("svm", SVC(probability=True, random_state=42)),
        ])
        grid_params = {
            "select__k": [20, 30, "all"],
            "svm__C": [0.1, 1, 10, 50, 100],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
            "svm__kernel": ["rbf", "linear"],
        }
        n_splits = min(5, min(np.bincount(y)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, grid_params, cv=cv, scoring="accuracy",
                           n_jobs=-1, verbose=0)
        grid.fit(X, y)
        model = grid.best_estimator_
        logger.info(f"  Best: {grid.best_params_}")
        logger.info(f"  Best CV: {grid.best_score_:.4f}")
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(30, X.shape[1]))),
            ("svm", SVC(C=10, gamma="scale", kernel="rbf",
                       probability=True, random_state=42)),
        ])
        model.fit(X, y)

    # Evaluate
    n_splits = min(5, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)

    logger.info(f"\n  Training accuracy: {train_acc:.4f}")
    logger.info(f"  CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=MI_NAMES)}")

    # Save
    model_path = MODELS_DIR / "mi_referential_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "class_names": MI_NAMES,
            "n_features": X.shape[1],
            "train_accuracy": float(train_acc),
            "cv_accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "montage": "referential (C3 to earlobe)",
        }, f)
    logger.info(f"  ✓ Saved: {model_path}")

    return {
        "model": model,
        "accuracy": train_acc,
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, target_names=MI_NAMES, output_dict=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 4-class MI from Referential C3")
    parser.add_argument("--subject", type=int, help="Subject ID")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--trials", type=int, default=40,
                        help="Simulated trials per class")
    args = parser.parse_args()

    from src.utils import print_banner
    print_banner()
    print("  >> REFERENTIAL C3 MODEL TRAINING (4-class)\n")

    if args.simulate:
        logger.info("Generating simulated referential data...")
        data, labels = generate_simulated(n_per_class=args.trials)
    else:
        if not args.subject:
            logger.error("Need --subject or --simulate")
            return
        data, labels = load_data(args.subject, args.session)
        if data is None:
            return

    results = train(data, labels, do_grid_search=args.grid_search)

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  4-class MI (C3 referential): "
          f"{results['accuracy']:.1%} train, "
          f"{results['cv_accuracy']:.1%} ± {results['cv_std']:.1%} CV")
    print(f"  Model: {MODELS_DIR / 'mi_referential_model.pkl'}")
    print("=" * 60)

    # Save JSON
    out = {k: v for k, v in results.items() if k != "model"}
    with open(RESULTS_DIR / "referential_training_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
