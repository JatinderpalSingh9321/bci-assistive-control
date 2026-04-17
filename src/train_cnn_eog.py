"""
BCI Assistive Control — CNN Blink/Wink Training Pipeline
==========================================================
Trains a 1D CNN on blink/wink data collected across multiple sessions.

Based on: Eye Winking BCI Quick Reference Card
Adapted for: 1-channel Fp1 referential, ~1754 samples/trial, 3 classes

  Class 0: BLINK
  Class 1: WINK_LEFT
  Class 2: WINK_RIGHT

Usage:
  python -m src.train_cnn_eog --subject 1
  python -m src.train_cnn_eog --subject 1 --epochs 200
  python -m src.train_cnn_eog --subject 1 --compare

Group No. 7 | 8th Semester Major Project
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from src.utils import (
    SAMPLING_RATE, RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    setup_logger
)

logger = setup_logger("train_cnn_eog")

CLASS_NAMES = ["BLINK", "WINK_LEFT", "WINK_RIGHT"]

# Fixed length for all trials (pad/truncate to this)
FIXED_LENGTH = 1750  # ~3 seconds at actual sample rate


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_all_sessions(subject_id, max_sessions=10):
    """Load blink/wink data from all available sessions."""
    data_dir = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    all_data = []
    all_labels = []

    for sess in range(1, max_sessions + 1):
        path = data_dir / f"session_{sess:02d}_blink_wink.npz"
        if not path.exists():
            continue

        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"]
        labels = loaded["labels"]

        all_data.extend(data)
        all_labels.extend(labels)

        logger.info(f"  ✓ Session {sess}: {len(labels)} trials loaded")

    if not all_data:
        logger.error(f"No blink/wink data found in {data_dir}")
        return None, None

    all_labels = np.array(all_labels, dtype=np.int32)

    logger.info(f"\n  Total: {len(all_labels)} trials")
    for cid in range(3):
        count = np.sum(all_labels == cid)
        logger.info(f"    {CLASS_NAMES[cid]:12s}: {count} trials")

    return all_data, all_labels


def preprocess_data(raw_data, fixed_len=FIXED_LENGTH):
    """
    Preprocess raw 1-channel epochs into a fixed-size matrix.

    Steps:
    1. Notch filter (50 Hz)
    2. Bandpass filter (0.5 - 45 Hz)
    3. Baseline correction (subtract mean)
    4. Normalize (divide by std)
    5. Pad or truncate to fixed length

    Returns: np.ndarray of shape (n_trials, fixed_len, 1)
    """
    processed = []

    for i, epoch in enumerate(raw_data):
        sig = np.array(epoch, dtype=np.float64).flatten()

        # Notch filter (50 Hz power line)
        b, a = scipy_signal.iirnotch(50.0, 30, SAMPLING_RATE)
        sig = scipy_signal.filtfilt(b, a, sig)

        # Bandpass 0.5 - 45 Hz
        nyq = SAMPLING_RATE / 2
        low = max(0.5 / nyq, 0.001)
        high = min(45.0 / nyq, 0.999)
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        sig = scipy_signal.filtfilt(b, a, sig)

        # Baseline correction
        sig = sig - np.mean(sig)

        # Normalize
        std = np.std(sig)
        if std > 1e-10:
            sig = sig / std

        # Pad or truncate to fixed length
        if len(sig) >= fixed_len:
            sig = sig[:fixed_len]
        else:
            sig = np.pad(sig, (0, fixed_len - len(sig)), mode='constant')

        processed.append(sig)

    X = np.array(processed, dtype=np.float32)
    # Add channel dimension: (n_trials, fixed_len) → (n_trials, fixed_len, 1)
    X = X[:, :, np.newaxis]

    logger.info(f"  Preprocessed shape: {X.shape}")
    return X


# ──────────────────────────────────────────────
# CNN MODEL
# ──────────────────────────────────────────────

def build_cnn(input_shape, n_classes=3):
    """Build 1D CNN for blink/wink classification."""
    from tensorflow import keras

    model = keras.Sequential([
        # Conv block 1
        keras.layers.Conv1D(16, 25, activation='relu',
                          input_shape=input_shape, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(4),
        keras.layers.Dropout(0.2),

        # Conv block 2
        keras.layers.Conv1D(32, 15, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(4),
        keras.layers.Dropout(0.2),

        # Conv block 3
        keras.layers.Conv1D(64, 7, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(4),
        keras.layers.Dropout(0.3),

        # Dense head
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ──────────────────────────────────────────────
# SVM MODEL (for comparison)
# ──────────────────────────────────────────────

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM for comparison."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report

    # Flatten for SVM
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_ts_flat = X_test.reshape(X_test.shape[0], -1)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=10, gamma="scale", kernel="rbf",
                    probability=True, random_state=42)),
    ])

    svm.fit(X_tr_flat, y_train)
    y_pred = svm.predict(X_ts_flat)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"\n  SVM Test Accuracy: {acc:.1%}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    return svm, acc


# ──────────────────────────────────────────────
# MAIN TRAINING
# ──────────────────────────────────────────────

def train(args):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )

    # ── Load data ──
    logger.info("=" * 60)
    logger.info("  CNN BLINK/WINK TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"\n  Loading data for subject {args.subject}...")

    raw_data, labels = load_all_sessions(args.subject)
    if raw_data is None:
        return

    # ── Preprocess ──
    logger.info("\n  Preprocessing...")
    X = preprocess_data(raw_data)

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42
    )

    logger.info(f"\n  Train: {len(y_train)} trials")
    logger.info(f"  Test:  {len(y_test)} trials")
    for cid in range(3):
        n_tr = np.sum(y_train == cid)
        n_ts = np.sum(y_test == cid)
        logger.info(f"    {CLASS_NAMES[cid]:12s}: {n_tr} train, {n_ts} test")

    # ── Train CNN ──
    logger.info("\n" + "─" * 60)
    logger.info("  TRAINING 1D CNN")
    logger.info("─" * 60)

    from tensorflow import keras

    model = build_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_classes=3
    )
    model.summary(print_fn=lambda x: logger.info(f"  {x}"))

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=1
    )

    train_time = time.time() - start_time
    logger.info(f"\n  Training time: {train_time:.1f}s")
    logger.info(f"  Epochs trained: {len(history.history['loss'])}")

    # ── Evaluate CNN ──
    logger.info("\n" + "─" * 60)
    logger.info("  CNN RESULTS")
    logger.info("─" * 60)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    cnn_acc = accuracy_score(y_test, y_pred)

    logger.info(f"\n  CNN Test Accuracy: {cnn_acc:.1%}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info("  Confusion Matrix:")
    logger.info(f"             {'  '.join(f'{n:>10s}' for n in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        logger.info(f"  {CLASS_NAMES[i]:12s} {'  '.join(f'{v:>10d}' for v in row)}")

    # ── Save CNN Model ──
    cnn_path = MODELS_DIR / "cnn_eog_model.h5"
    model.save(cnn_path)
    logger.info(f"\n  ✓ CNN model saved: {cnn_path}")

    # Also save as pickle with metadata for mouse controller
    eog_model_path = MODELS_DIR / "eog_cnn_model.pkl"
    with open(eog_model_path, "wb") as f:
        pickle.dump({
            "model_type": "cnn",
            "model_path": str(cnn_path),
            "class_names": CLASS_NAMES,
            "train_accuracy": float(cnn_acc),
            "fixed_length": FIXED_LENGTH,
            "source": "train_cnn_eog.py",
        }, f)

    # ── Compare with SVM ──
    svm_acc = None
    if args.compare:
        logger.info("\n" + "─" * 60)
        logger.info("  COMPARISON: SVM vs CNN")
        logger.info("─" * 60)

        svm_model, svm_acc = train_svm(X_train, y_train, X_test, y_test)

        # Save SVM model too (as backup)
        svm_path = MODELS_DIR / "eog_3class_model.pkl"
        with open(svm_path, "wb") as f:
            pickle.dump({
                "model": svm_model,
                "class_names": CLASS_NAMES,
                "n_features": X_train.shape[1] * X_train.shape[2],
                "train_accuracy": float(svm_acc),
                "source": "train_cnn_eog.py",
                "data_type": "1ch_fp1_preprocessed",
            }, f)
        logger.info(f"  ✓ SVM model saved: {svm_path}")

    # ── Training Curves ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / "cnn_eog_training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"\n  ✓ Training curves saved: {plot_path}")
    except Exception as e:
        logger.warning(f"  Could not save plots: {e}")

    # ── Save Results ──
    results = {
        "total_samples": int(len(labels)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "cnn_accuracy": float(cnn_acc),
        "svm_accuracy": float(svm_acc) if svm_acc else None,
        "epochs_trained": len(history.history['loss']),
        "training_time_sec": float(train_time),
        "class_counts": {
            CLASS_NAMES[c]: int(np.sum(labels == c)) for c in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
    }

    results_path = RESULTS_DIR / "cnn_eog_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✓ Results saved: {results_path}")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Samples:    {len(labels)}")
    print(f"  CNN Test Accuracy: {cnn_acc:.1%}")
    if svm_acc:
        print(f"  SVM Test Accuracy: {svm_acc:.1%}")
        winner = "CNN" if cnn_acc > svm_acc else "SVM"
        print(f"  Winner:            {winner}")
    print(f"  Training Time:     {train_time:.1f}s")
    print(f"  Epochs:            {len(history.history['loss'])}")
    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"  Results saved to: {results_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN + SVM on Blink/Wink Data (1-Channel Fp1)"
    )
    parser.add_argument("--subject", type=int, required=True,
                        help="Subject ID")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default: 100)")
    parser.add_argument("--compare", action="store_true",
                        help="Also train SVM for comparison")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
