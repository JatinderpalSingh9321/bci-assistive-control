"""
BCI Assistive Control — Training Orchestration
================================================
Orchestrates the full training pipeline: loads preprocessed data,
trains models (SVM, CNN, LSTM), runs LOSO evaluation, generates
visualizations, and saves trained models.

Usage:
  python -m src.train --model cnn
  python -m src.train --model svm
  python -m src.train --model all
  python -m src.train --model cnn --simulate

Group No. 7 | 8th Semester Major Project
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils import (
    PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    CNN_MODEL_PATH, SVM_MODEL_PATH, RESULTS_CSV,
    N_CLASSES, CLASS_NAMES, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_PATIENCE,
    EPOCH_SAMPLES, setup_logger, print_banner
)
from src.models import (
    build_svm, build_1d_cnn, build_lstm, build_cnn_lstm,
    augment_dataset, MODEL_BUILDERS
)
from src.feature_extraction import extract_features_batch
from src.evaluation import (
    loso_evaluate, plot_confusion_matrix, plot_subject_accuracy,
    plot_roc_curve, plot_comparison, save_results_csv
)

logger = setup_logger("train")


def load_dataset(simulate=False):
    """
    Load the preprocessed dataset (or generate simulated data).

    Parameters
    ----------
    simulate : bool
        If True, generate and use simulated data.

    Returns
    -------
    X : np.ndarray
        Shape (n_trials, n_timepoints).
    y : np.ndarray
        Shape (n_trials,).
    subject_ids : np.ndarray
        Shape (n_trials,).
    """
    if simulate:
        logger.info("Using simulated dataset...")
        from src.acquisition import generate_simulated_dataset
        X, y, subject_ids = generate_simulated_dataset()
        return X, y, subject_ids

    data_path = PREPROCESSED_DIR / "all_clean_epochs.npz"

    if not data_path.exists():
        # Try simulated dataset as fallback
        sim_path = PREPROCESSED_DIR / "simulated_dataset.npz"
        if sim_path.exists():
            logger.info(f"Loading simulated dataset: {sim_path}")
            data_path = sim_path
        else:
            logger.error(f"No dataset found at {data_path}")
            logger.info("Run preprocessing first: python -m src.preprocessing --all-subjects")
            logger.info("Or generate simulated data: python -m src.acquisition --simulate")
            return None, None, None

    loaded = np.load(data_path)
    X = loaded["X"]
    y = loaded["y"]
    subject_ids = loaded["subject_ids"]

    logger.info(f"Loaded dataset: {X.shape[0]} trials, "
                f"{len(np.unique(subject_ids))} subjects, "
                f"shape {X.shape}")
    return X, y, subject_ids


def train_and_evaluate(model_type, X, y, subject_ids):
    """
    Train and evaluate a model using LOSO cross-validation.

    Parameters
    ----------
    model_type : str
        One of: "svm", "cnn", "lstm", "cnn_lstm".
    X, y, subject_ids : np.ndarray
        Dataset arrays.

    Returns
    -------
    results : dict
        LOSO evaluation results.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  TRAINING: {model_type.upper()}")
    logger.info(f"{'=' * 60}")

    builder = lambda: MODEL_BUILDERS[model_type]() if model_type != "svm" else build_svm()

    start = time.time()
    results = loso_evaluate(
        X, y, subject_ids,
        model_builder=builder,
        model_type=model_type
    )
    elapsed = time.time() - start
    results["training_time_sec"] = round(elapsed, 1)

    logger.info(f"Training completed in {elapsed:.1f}s")
    return results


def train_final_model(model_type, X, y, subject_ids):
    """
    Train a final model on ALL data (for deployment / demo).

    Parameters
    ----------
    model_type : str
        Model type.
    X, y, subject_ids : np.ndarray
        Full dataset.
    """
    logger.info(f"Training final {model_type.upper()} on all data for deployment...")

    if model_type == "svm":
        model = build_svm()
        X_feat = extract_features_batch(X)
        model.fit(X_feat, y)
        with open(SVM_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"✓ Final SVM saved: {SVM_MODEL_PATH}")

    elif model_type in ("cnn", "lstm", "cnn_lstm"):
        builder_map = {"cnn": build_1d_cnn, "lstm": build_lstm, "cnn_lstm": build_cnn_lstm}
        model = builder_map[model_type]()

        X_in = X[..., np.newaxis]
        y_cat = tf.keras.utils.to_categorical(y, N_CLASSES)

        # Augment for final training
        X_aug, y_aug = augment_dataset(X, y, augment_factor=3)
        X_aug_in = X_aug[..., np.newaxis]
        y_aug_cat = tf.keras.utils.to_categorical(y_aug, N_CLASSES)

        model.fit(
            X_aug_in, y_aug_cat,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            verbose=1,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=CNN_PATIENCE,
                    restore_best_weights=True
                )
            ]
        )

        save_path = MODELS_DIR / f"{model_type}_model.h5"
        model.save(save_path)
        logger.info(f"✓ Final {model_type.upper()} saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BCI Model Training Pipeline")
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["svm", "cnn", "lstm", "cnn_lstm", "all"],
                        help="Model to train (default: cnn)")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data for testing")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--save-final", action="store_true",
                        help="Train and save a final model on all data")
    args = parser.parse_args()

    print_banner()

    # Load data
    X, y, subject_ids = load_dataset(simulate=args.simulate)
    if X is None:
        return

    # Determine which models to train
    if args.model == "all":
        models_to_train = ["svm", "cnn", "lstm"]
    else:
        models_to_train = [args.model]

    # Train and evaluate each model
    all_results = []
    for model_type in models_to_train:
        results = train_and_evaluate(model_type, X, y, subject_ids)
        all_results.append(results)

        if not args.no_plots:
            plot_confusion_matrix(results)
            plot_subject_accuracy(results)
            plot_roc_curve(results)

    # Comparison plot (if multiple models)
    if len(all_results) > 1 and not args.no_plots:
        plot_comparison(all_results)

    # Save results CSV
    save_results_csv(all_results)

    # Train final deployment model
    if args.save_final:
        for model_type in models_to_train:
            train_final_model(model_type, X, y, subject_ids)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("  TRAINING SUMMARY")
    logger.info(f"{'=' * 60}")
    for r in all_results:
        logger.info(f"  {r['model_type'].upper():10s}  "
                    f"Accuracy: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%}  "
                    f"Time: {r['training_time_sec']:.1f}s")
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Models saved to:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
