"""
BCI Assistive Control -- Improved Training Pipeline
=====================================================
Full improved pipeline starting from RAW data:
  1. Improved preprocessing (0.5-50 Hz, no ICA, CAR)
  2. Better feature extraction (5 types, ~22 features/channel)
  3. Improved SVM training (scaling, selection, PCA, grid search)
  4. Optional data augmentation (mixup, jitter, time-warp)

Usage:
  python -m src.train_improved --mode simple         # Quick training
  python -m src.train_improved --mode grid_search    # Full grid search
  python -m src.train_improved --mode augmented      # With data augmentation
  python -m src.train_improved --mode compare        # Compare old vs new
  python -m src.train_improved --mode simple --simulate  # Use simulated data

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import pickle
from pathlib import Path

import numpy as np

from src.utils import (
    PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR,
    SVM_MODEL_PATH, SAMPLING_RATE, TARGET_SAMPLING_RATE,
    N_CLASSES, CLASS_NAMES, EPOCH_SAMPLES,
    DOWNSAMPLE_FACTOR, get_all_subjects,
    setup_logger, print_banner
)
from src.improved_preprocessing import ImprovedEEGPreprocessor
from src.better_feature_extraction import BetterFeatureExtractor
from src.improved_svm_trainer import ImprovedSVMTrainer, SyntheticEEGGenerator

logger = setup_logger("train_improved")


# ------------------------------------------------------------------ #
#  DATA LOADING                                                        #
# ------------------------------------------------------------------ #

def load_raw_data():
    """
    Load RAW (unprocessed) data from data/raw/ for all subjects.
    Returns variable-length trial lists, labels, and subject IDs.
    """
    subjects = get_all_subjects()
    if not subjects:
        logger.warning("No raw subject data found.")
        return None, None, None

    all_trials = []
    all_labels = []
    all_sids = []

    for sid in subjects:
        data_path = RAW_DATA_DIR / f"subject_{sid:03d}" / "session_01.npz"
        if not data_path.exists():
            continue
        loaded = np.load(data_path, allow_pickle=True)
        raw_trials = loaded["data"]  # object array of variable-length 1-D lists
        labels = loaded["labels"]

        for trial, lbl in zip(raw_trials, labels):
            all_trials.append(np.array(trial, dtype=np.float64))
            all_labels.append(int(lbl))
            all_sids.append(sid)

    if not all_trials:
        return None, None, None

    labels = np.array(all_labels, dtype=np.int32)
    sids = np.array(all_sids, dtype=np.int32)
    logger.info(
        f"Loaded RAW data: {len(all_trials)} trials, "
        f"{len(np.unique(sids))} subjects, "
        f"trial lengths {min(len(t) for t in all_trials)}-"
        f"{max(len(t) for t in all_trials)} samples"
    )
    return all_trials, labels, sids


def load_preprocessed_data():
    """Load already-preprocessed data (for the OLD pipeline comparison)."""
    data_path = PREPROCESSED_DIR / "all_clean_epochs.npz"
    if not data_path.exists():
        data_path = PREPROCESSED_DIR / "simulated_dataset.npz"
    if not data_path.exists():
        return None, None, None

    loaded = np.load(data_path)
    X = loaded["X"]
    y = loaded["y"]
    subject_ids = loaded["subject_ids"]
    logger.info(f"Loaded preprocessed data: {X.shape[0]} trials, shape {X.shape}")
    return X, y, subject_ids


def load_simulated_data():
    """Generate simulated data."""
    logger.info("Generating simulated dataset...")
    from src.acquisition import generate_simulated_dataset
    X, y, subject_ids = generate_simulated_dataset()
    return X, y, subject_ids


# ------------------------------------------------------------------ #
#  IMPROVED PREPROCESSING  (from raw)                                  #
# ------------------------------------------------------------------ #

def preprocess_raw_trials(raw_trials, target_length=EPOCH_SAMPLES):
    """
    Apply the IMPROVED preprocessing pipeline to raw variable-length trials.

    Steps per trial:
      1. Robust outlier removal (5-sigma)
      2. Notch filter 50 Hz
      3. Wide bandpass 0.5-50 Hz  (key change!)
      4. Downsample 250 -> 125 Hz
      5. Baseline correction
      6. Pad / truncate to target_length

    Returns
    -------
    epochs : np.ndarray  (n_trials, target_length)
    """
    from scipy import signal as sp_signal

    preprocessor = ImprovedEEGPreprocessor(fs=SAMPLING_RATE, verbose=False)

    epochs = []
    for i, trial in enumerate(raw_trials):
        x = trial.copy().astype(np.float64)

        # Minimum-length guard for filtering
        min_len = 15
        if len(x) < min_len:
            x = np.pad(x, (0, min_len - len(x)), mode="edge")

        # 1. Outlier removal (> 5 sigma)
        mean, std = np.mean(x), np.std(x)
        mask = np.abs(x - mean) > 5 * std
        if np.any(mask):
            idx = np.arange(len(x))
            good = ~mask
            if np.any(good):
                x[mask] = np.interp(idx[mask], idx[good], x[good])

        # 2. Notch filter 50 Hz
        b, a = sp_signal.iirnotch(50, 30, fs=SAMPLING_RATE)
        if len(x) > 3 * max(len(b), len(a)):
            x = sp_signal.filtfilt(b, a, x)

        # 3. WIDE bandpass: 0.5 -- 50 Hz  (was 8-30 Hz!)
        nyq = SAMPLING_RATE / 2.0
        hi = min(50.0, nyq - 1)
        sos = sp_signal.butter(4, [0.5, hi], "bandpass",
                               fs=SAMPLING_RATE, output="sos")
        x = sp_signal.sosfiltfilt(sos, x)

        # 4. Downsample (250 -> 125 Hz)
        x = x[::DOWNSAMPLE_FACTOR]

        # 5. Baseline correction (subtract first 0.5 s mean)
        bl = max(1, int(0.5 * TARGET_SAMPLING_RATE))
        x = x - np.mean(x[:bl])

        # 6. Pad or truncate
        if len(x) < target_length:
            x = np.pad(x, (0, target_length - len(x)), mode="edge")
        else:
            x = x[:target_length]

        epochs.append(x.astype(np.float32))

        if (i + 1) % 100 == 0:
            logger.info(f"  Preprocessed {i+1}/{len(raw_trials)} trials")

    return np.array(epochs, dtype=np.float32)


# ------------------------------------------------------------------ #
#  MAIN PIPELINE                                                       #
# ------------------------------------------------------------------ #

def run_improved_pipeline(X, y, subject_ids, mode="simple",
                          augment=False, cv_folds=5):
    """
    Run the full improved BCI pipeline on already-epoch'd data.

    Parameters
    ----------
    X : np.ndarray   (n_trials, n_timepoints)
    y : np.ndarray   (n_trials,)
    subject_ids : np.ndarray   (n_trials,)
    mode : str       "simple" | "grid_search"
    augment : bool
    cv_folds : int
    """
    print("\n" + "=" * 60)
    print("  IMPROVED BCI TRAINING PIPELINE")
    print("=" * 60)
    start_time = time.time()

    # -------- STEP 2: Better Feature Extraction --------
    print("\n" + "-" * 60)
    print("  STEP 2: Better Feature Extraction")
    print("-" * 60)

    extractor = BetterFeatureExtractor(fs=TARGET_SAMPLING_RATE)
    features = extractor.extract_batch(X)

    print(f"  [ok] Extracted: {features.shape[1]} features per epoch")
    print(f"  [ok] Feature types: band power (6) + temporal (9) + "
          f"frequency (4) + nonlinear (3)")
    print(f"  [ok] Total feature matrix: {features.shape}")

    # -------- STEP 3 (Optional): Data Augmentation --------
    y_work = y.copy()
    sids_work = subject_ids.copy()

    if augment:
        print("\n" + "-" * 60)
        print("  STEP 3: Data Augmentation")
        print("-" * 60)

        n_before = len(features)

        # Mixup on raw epochs
        X_aug, y_aug = SyntheticEEGGenerator.mixup_augmentation(
            X, y, n_new_samples=min(500, len(X))
        )
        features_aug = extractor.extract_batch(X_aug)

        # Jitter on features
        features, y_work = SyntheticEEGGenerator.jitter_augmentation(
            features_aug, y_aug, n_copies=2, noise_factor=0.03
        )

        n_rep = len(features) // len(subject_ids) + 1
        sids_work = np.tile(subject_ids, n_rep)[:len(features)]

        print(f"  [ok] Augmented: {n_before} -> {len(features)} samples")

    # -------- STEP 4: LOSO with Improved SVM --------
    print("\n" + "-" * 60)
    print("  STEP 4: Improved SVM Training (LOSO)")
    print("-" * 60)

    unique_subs = np.unique(sids_work)
    per_sub_acc = {}
    all_preds, all_true, all_probs = [], [], []

    print(f"  Subjects: {len(unique_subs)}")
    print(f"  Mode: {mode}")
    print()

    for sub in unique_subs:
        te = sids_work == sub
        tr = ~te
        X_tr, y_tr = features[tr], y_work[tr]
        X_te, y_te = features[te], y_work[te]

        if len(X_tr) < 10 or len(X_te) < 2:
            logger.warning(f"  Subject {int(sub)}: skipped (too few samples)")
            continue

        trainer = ImprovedSVMTrainer(verbose=False)
        if mode == "grid_search":
            trainer.train_with_grid_search(X_tr, y_tr, cv_folds=cv_folds)
        else:
            trainer.train_simple(X_tr, y_tr)

        preds, probs = trainer.predict(X_te)
        acc = float(np.mean(preds == y_te))
        per_sub_acc[int(sub)] = round(acc, 4)

        all_preds.extend(preds)
        all_true.extend(y_te)
        all_probs.extend(probs)

        info = ""
        if trainer.best_params_:
            info = f" (C={trainer.best_params_.get('C')}, " \
                   f"kernel={trainer.best_params_.get('kernel')})"
        print(f"  Subject {int(sub):3d}: {acc:.1%} "
              f"({int(np.sum(preds == y_te))}/{len(y_te)}){info}")

    # -------- RESULTS --------
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs) if all_probs else None

    overall = float(np.mean(all_preds == all_true))
    vals = list(per_sub_acc.values())
    mean_acc = float(np.mean(vals)) if vals else 0
    std_acc = float(np.std(vals)) if vals else 0
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  IMPROVED SVM RESULTS")
    print("=" * 60)
    print(f"  Overall accuracy:    {overall:.1%}")
    print(f"  Mean +/- Std:        {mean_acc:.1%} +/- {std_acc:.1%}")
    print(f"  Per-subject:         {per_sub_acc}")
    print(f"  Training time:       {elapsed:.1f}s")
    print(f"  Mode:                {mode}")
    print(f"  Augmented:           {augment}")
    print("=" * 60)

    # Train final model on ALL data
    print("\nTraining final model on ALL data...")
    final = ImprovedSVMTrainer(verbose=True)
    if mode == "grid_search":
        final.train_with_grid_search(features, y_work, cv_folds=cv_folds)
    else:
        final.train_simple(features, y_work)
    final.save(MODELS_DIR / "improved_svm_model.pkl")

    results = {
        "model_type": "improved_svm",
        "per_subject_accuracy": per_sub_acc,
        "overall_accuracy": round(overall, 4),
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "predictions": all_preds,
        "true_labels": all_true,
        "probabilities": all_probs,
        "training_time_sec": round(elapsed, 1),
        "mode": mode,
        "augmented": augment,
    }
    return results


# ------------------------------------------------------------------ #
#  OLD vs NEW COMPARISON                                               #
# ------------------------------------------------------------------ #

def compare_old_vs_new():
    """
    End-to-end comparison:
      OLD  = already-preprocessed (8-30 Hz) + 5 features + default SVM
      NEW  = raw -> improved preproc (0.5-50 Hz) + 22 features + tuned SVM
    """
    from src.feature_extraction import extract_features_batch
    from src.models import build_svm
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 60)
    print("  OLD vs NEW SVM COMPARISON")
    print("=" * 60)

    # ---------- OLD PIPELINE (uses pre-processed data) ----------
    print("\n--- OLD Pipeline (8-30 Hz + 5 features + default SVM) ---")
    X_pre, y_pre, sids_pre = load_preprocessed_data()
    if X_pre is None:
        print("  [!] No preprocessed data found; skipping old pipeline.")
        old_acc = None
    else:
        start = time.time()
        unique_subs = np.unique(sids_pre)
        old_preds, old_true = [], []

        for sub in unique_subs:
            te = sids_pre == sub
            tr = ~te
            feat_tr = extract_features_batch(X_pre[tr])
            feat_te = extract_features_batch(X_pre[te])

            model = build_svm()
            model.fit(feat_tr, y_pre[tr])
            preds = model.predict(feat_te)
            old_preds.extend(preds)
            old_true.extend(y_pre[te])

        old_acc = accuracy_score(old_true, old_preds)
        print(f"  Old accuracy: {old_acc:.1%} ({time.time()-start:.1f}s)")

    # ---------- NEW PIPELINE (starts from raw) ----------
    print("\n--- NEW Pipeline (RAW -> 0.5-50 Hz + 22 features + tuned SVM) ---")

    raw_trials, y_raw, sids_raw = load_raw_data()
    if raw_trials is None:
        print("  [!] No raw data found; cannot run new pipeline.")
        return

    print("  Step 1: Re-preprocessing from raw (0.5-50 Hz, no ICA)...")
    X_new = preprocess_raw_trials(raw_trials)
    print(f"  [ok] Re-preprocessed: {X_new.shape}")

    # Run improved pipeline with grid search for best results
    new_results = run_improved_pipeline(
        X_new, y_raw, sids_raw, mode="grid_search", cv_folds=3
    )
    new_acc = new_results["overall_accuracy"]

    # ---------- COMPARISON ----------
    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    if old_acc is not None:
        improvement = (new_acc - old_acc) * 100
        print(f"  Old SVM accuracy:      {old_acc:.1%}")
        print(f"  New SVM accuracy:      {new_acc:.1%}")
        print(f"  Improvement:           {improvement:+.1f} percentage points")
        rel = improvement / max(old_acc * 100, 1)
        print(f"  Relative improvement:  {rel:+.0f}%")
    else:
        print(f"  New SVM accuracy:      {new_acc:.1%}")
    print("=" * 60)

    return new_results


# ------------------------------------------------------------------ #
#  VISUALIZATION                                                       #
# ------------------------------------------------------------------ #

def plot_improvement_results(results):
    """Generate plots for improved results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix as cm_func

        # Confusion matrix
        cm = cm_func(results["true_labels"], results["predictions"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, square=True, linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Improved SVM -- Confusion Matrix\n"
                     f"Accuracy: {results['overall_accuracy']:.1%}", fontsize=14)
        plt.tight_layout()
        save_path = RESULTS_DIR / "confusion_matrix_improved_svm.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")

        # Per-subject accuracy
        subjects = list(results["per_subject_accuracy"].keys())
        accuracies = list(results["per_subject_accuracy"].values())

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#00d2ff" if a >= 0.8 else "#f5a623" if a >= 0.7 else "#e94560"
                  for a in accuracies]
        bars = ax.bar(range(len(subjects)), [a * 100 for a in accuracies],
                      color=colors)
        ax.set_xlabel("Subject", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Improved SVM -- Per-Subject Accuracy\n"
                     f"Mean: {results['mean_accuracy']:.1%} +/- "
                     f"{results['std_accuracy']:.1%}", fontsize=14)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([f"S{s}" for s in subjects])
        ax.set_ylim(0, 100)
        ax.axhline(y=results["mean_accuracy"] * 100, color="red",
                   linestyle="--", label=f"Mean: {results['mean_accuracy']:.1%}")
        ax.legend(fontsize=10)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.0%}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        save_path = RESULTS_DIR / "loso_accuracy_improved_svm.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")

    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="BCI Improved SVM Training Pipeline"
    )
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "grid_search", "augmented", "compare"],
                        help="Training mode (default: simple)")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds for grid search")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    print_banner()
    print("  >> IMPROVED SVM TRAINING")
    print()

    # ---- Compare mode ----
    if args.mode == "compare":
        results = compare_old_vs_new()
        if results and not args.no_plots:
            plot_improvement_results(results)
        return

    # ---- Load data ----
    if args.simulate:
        X, y, sids = load_simulated_data()
        # Simulated data is already preprocessed-like, just use it
    else:
        # Start from RAW for maximum benefit
        raw_trials, y, sids = load_raw_data()
        if raw_trials is None:
            logger.error("No raw data. Use --simulate or collect data first.")
            return
        print("  Step 1: Re-preprocessing from raw (0.5-50 Hz, no ICA)...")
        X = preprocess_raw_trials(raw_trials)
        print(f"  [ok] Re-preprocessed: {X.shape}")

    if X is None:
        return

    augment = (args.mode == "augmented")
    mode = "grid_search" if args.mode in ("grid_search", "augmented") else "simple"

    results = run_improved_pipeline(
        X, y, sids,
        mode=mode,
        augment=augment,
        cv_folds=args.cv_folds
    )

    if not args.no_plots:
        plot_improvement_results(results)

    print(f"\n[ok] Done! Check results/ directory for plots.")
    print(f"[ok] Model saved to: {MODELS_DIR / 'improved_svm_model.pkl'}")


if __name__ == "__main__":
    main()
