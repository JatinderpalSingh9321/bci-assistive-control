"""
BCI Assistive Control — Model Evaluation
==========================================
Implements Leave-One-Subject-Out (LOSO) cross-validation,
classification metrics, and visualization for motor imagery models.

Group No. 7 | 8th Semester Major Project
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
import tensorflow as tf

from src.utils import (
    N_CLASSES, CLASS_NAMES, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_PATIENCE,
    RESULTS_DIR, MODELS_DIR, setup_logger
)
from src.feature_extraction import extract_features_batch

logger = setup_logger("evaluation")


# ──────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ──────────────────────────────────────────────

def loso_evaluate(X, y, subject_ids, model_builder, model_type="cnn",
                  epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
                  patience=CNN_PATIENCE):
    """
    Leave-One-Subject-Out cross-validation.

    For each subject, train on all other subjects and test on the
    held-out subject. This is the gold standard for BCI evaluation.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed epochs, shape (n_total, n_timepoints).
    y : np.ndarray
        Labels, shape (n_total,).
    subject_ids : np.ndarray
        Subject ID per trial, shape (n_total,).
    model_builder : callable
        Function that returns a fresh model.
    model_type : str
        "cnn", "lstm", or "svm".
    epochs : int
        Max training epochs (deep learning only).
    batch_size : int
        Batch size (deep learning only).
    patience : int
        Early stopping patience (deep learning only).

    Returns
    -------
    results : dict
        Contains per-subject accuracies, overall metrics, and predictions.
    """
    unique_subjects = np.unique(subject_ids)
    all_preds = []
    all_true = []
    all_probs = []
    per_subject_acc = {}

    logger.info(f"Starting LOSO evaluation: {len(unique_subjects)} subjects, "
                f"model={model_type}")

    for sub in unique_subjects:
        test_mask  = subject_ids == sub
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        model = model_builder()

        if model_type in ("cnn", "lstm", "cnn_lstm"):
            # Reshape for Conv1D/LSTM: (batch, timepoints, 1)
            X_train_in = X_train[..., np.newaxis]
            X_test_in  = X_test[..., np.newaxis]
            y_train_cat = tf.keras.utils.to_categorical(y_train, N_CLASSES)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=patience,
                    restore_best_weights=True,
                    monitor="val_loss"
                )
            ]

            model.fit(
                X_train_in, y_train_cat,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1,
                callbacks=callbacks
            )

            probs = model.predict(X_test_in, verbose=0)
            preds = np.argmax(probs, axis=1)

        elif model_type == "svm":
            # SVM uses hand-crafted features
            X_train_feat = extract_features_batch(X_train)
            X_test_feat  = extract_features_batch(X_test)

            model.fit(X_train_feat, y_train)
            preds = model.predict(X_test_feat)
            probs = model.predict_proba(X_test_feat)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        acc = accuracy_score(y_test, preds)
        per_subject_acc[int(sub)] = round(acc, 4)

        all_preds.extend(preds)
        all_true.extend(y_test)
        if probs is not None:
            all_probs.extend(probs)

        logger.info(f"  Subject {int(sub):3d}: accuracy = {acc:.1%} "
                    f"({np.sum(preds == y_test)}/{len(y_test)})")

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs) if all_probs else None

    overall_acc = accuracy_score(all_true, all_preds)
    acc_values = list(per_subject_acc.values())
    mean_acc = np.mean(acc_values)
    std_acc  = np.std(acc_values)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Overall accuracy:  {overall_acc:.1%}")
    logger.info(f"Mean ± Std:        {mean_acc:.1%} ± {std_acc:.1%}")
    logger.info(f"{'=' * 50}")

    report = classification_report(
        all_true, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    report_str = classification_report(
        all_true, all_preds,
        target_names=CLASS_NAMES
    )
    logger.info(f"\n{report_str}")

    results = {
        "model_type": model_type,
        "per_subject_accuracy": per_subject_acc,
        "overall_accuracy": round(overall_acc, 4),
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "predictions": all_preds,
        "true_labels": all_true,
        "probabilities": all_probs,
        "classification_report": report,
    }

    return results


# ──────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────

def plot_confusion_matrix(results, save_path=None):
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    results : dict
        Output from loso_evaluate().
    save_path : str or Path or None
        Where to save the figure. Auto-generated if None.
    """
    cm = confusion_matrix(results["true_labels"], results["predictions"])
    model_type = results["model_type"].upper()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, square=True, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_type}\n"
                 f"Accuracy: {results['overall_accuracy']:.1%}", fontsize=14)

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"confusion_matrix_{results['model_type']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def plot_subject_accuracy(results, save_path=None):
    """
    Plot per-subject accuracy as a bar chart.

    Parameters
    ----------
    results : dict
        Output from loso_evaluate().
    save_path : str or Path or None
        Where to save the figure.
    """
    subjects = list(results["per_subject_accuracy"].keys())
    accuracies = list(results["per_subject_accuracy"].values())
    model_type = results["model_type"].upper()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#00d2ff" if a >= 0.8 else "#f5a623" if a >= 0.7 else "#e94560"
              for a in accuracies]
    bars = ax.bar(range(len(subjects)), [a * 100 for a in accuracies], color=colors)

    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"LOSO Per-Subject Accuracy — {model_type}\n"
                 f"Mean: {results['mean_accuracy']:.1%} ± {results['std_accuracy']:.1%}",
                 fontsize=14)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_ylim(0, 100)
    ax.axhline(y=results["mean_accuracy"] * 100, color="red", linestyle="--",
               label=f"Mean: {results['mean_accuracy']:.1%}")
    ax.legend(fontsize=10)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"loso_accuracy_{results['model_type']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy chart: {save_path}")


def plot_roc_curve(results, save_path=None):
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    results : dict
        Output from loso_evaluate().
    save_path : str or Path or None
        Where to save the figure.
    """
    if results["probabilities"] is None:
        logger.warning("No probabilities available for ROC curve.")
        return

    probs = results["probabilities"]
    true = results["true_labels"]
    model_type = results["model_type"].upper()

    # Use probability of class 1 (RIGHT)
    if probs.ndim == 2:
        prob_positive = probs[:, 1]
    else:
        prob_positive = probs

    fpr, tpr, _ = roc_curve(true, prob_positive)
    auc = roc_auc_score(true, prob_positive)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#00d2ff", lw=2, label=f"{model_type} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_type}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"roc_curve_{results['model_type']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curve: {save_path}")


def plot_comparison(results_list, save_path=None):
    """
    Plot accuracy comparison across multiple models.

    Parameters
    ----------
    results_list : list of dict
        List of outputs from loso_evaluate() for different models.
    save_path : str or Path or None
        Where to save the figure.
    """
    model_names = [r["model_type"].upper() for r in results_list]
    means = [r["mean_accuracy"] * 100 for r in results_list]
    stds  = [r["std_accuracy"] * 100 for r in results_list]

    colors = ["#00d2ff", "#f5a623", "#e94560", "#7b68ee"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                  color=colors[:len(model_names)], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("LOSO Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison — LOSO Cross-Validation", fontsize=14)
    ax.set_ylim(0, 100)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison chart: {save_path}")


def save_results_csv(results_list, save_path=None):
    """Save all LOSO results to a CSV file."""
    import pandas as pd

    rows = []
    for r in results_list:
        for sub, acc in r["per_subject_accuracy"].items():
            rows.append({
                "model": r["model_type"],
                "subject": sub,
                "accuracy": acc
            })

    df = pd.DataFrame(rows)

    if save_path is None:
        save_path = MODELS_DIR / "results.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Saved results CSV: {save_path}")
    return df


if __name__ == "__main__":
    logger.info("Evaluation module loaded. Use loso_evaluate() to run cross-validation.")
