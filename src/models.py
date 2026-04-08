"""
BCI Assistive Control — Model Definitions
==========================================
Defines all ML model architectures for motor imagery classification:
  - SVM (baseline) with hand-crafted spectral features
  - 1D-CNN (main thesis model) operating on raw preprocessed epochs
  - LSTM (bonus) for temporal sequence modeling

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import (
    EPOCH_SAMPLES, N_CLASSES,
    CNN_LEARNING_RATE, CNN_DROPOUT,
    SVM_KERNEL, SVM_C, SVM_GAMMA,
    setup_logger
)

logger = setup_logger("models")


# ──────────────────────────────────────────────
# SVM BASELINE
# ──────────────────────────────────────────────

def build_svm():
    """
    Build an SVM classification pipeline with standardization.

    The pipeline applies StandardScaler before an RBF-kernel SVM.
    Designed to work with hand-crafted features from feature_extraction.py.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted-ready SVM pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True,   # Enable predict_proba for confidence scores
            random_state=42
        ))
    ])
    logger.info(f"Built SVM pipeline: kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA}")
    return pipeline


# ──────────────────────────────────────────────
# 1D-CNN (MAIN MODEL)
# ──────────────────────────────────────────────

def build_1d_cnn(n_timepoints=EPOCH_SAMPLES, n_classes=N_CLASSES):
    """
    Build the 1D Convolutional Neural Network for motor imagery classification.

    Architecture:
        Input (n_timepoints, 1)
        → Conv1D(64, k=5, ReLU) → BatchNorm → Dropout(0.3)
        → Conv1D(128, k=5, ReLU) → BatchNorm → MaxPool(2) → Dropout(0.3)
        → Conv1D(64, k=3, ReLU) → GlobalAveragePooling1D
        → Dense(128, ReLU) → Dropout(0.3)
        → Dense(n_classes, Softmax)

    Parameters
    ----------
    n_timepoints : int
        Number of time samples per epoch (default: 500 at 125 Hz).
    n_classes : int
        Number of output classes (default: 2 = LEFT/RIGHT).

    Returns
    -------
    tf.keras.Model
        Compiled 1D-CNN model.
    """
    inputs = keras.Input(shape=(n_timepoints, 1), name="eeg_input")

    # Block 1: Temporal feature extraction
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu",
                      name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop1")(x)

    # Block 2: Deeper temporal features
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop2")(x)

    # Block 3: Refined features + global pooling
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Classifier head
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop_fc")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="BCI_1D_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_params = model.count_params()
    logger.info(f"Built 1D-CNN: input=({n_timepoints}, 1), "
                f"classes={n_classes}, params={total_params:,}")
    return model


# ──────────────────────────────────────────────
# LSTM (BONUS MODEL)
# ──────────────────────────────────────────────

def build_lstm(n_timepoints=EPOCH_SAMPLES, n_classes=N_CLASSES):
    """
    Build an LSTM model for temporal motor imagery classification.

    Architecture:
        Input (n_timepoints, 1)
        → LSTM(64, return_sequences=True) → Dropout(0.3)
        → LSTM(32) → Dropout(0.3)
        → Dense(64, ReLU) → Dropout(0.3)
        → Dense(n_classes, Softmax)

    Parameters
    ----------
    n_timepoints : int
        Number of time samples per epoch.
    n_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """
    inputs = keras.Input(shape=(n_timepoints, 1), name="eeg_input")

    x = layers.LSTM(64, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(CNN_DROPOUT, name="drop_lstm1")(x)

    x = layers.LSTM(32, name="lstm2")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop_lstm2")(x)

    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop_fc")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="BCI_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_params = model.count_params()
    logger.info(f"Built LSTM: input=({n_timepoints}, 1), "
                f"classes={n_classes}, params={total_params:,}")
    return model


# ──────────────────────────────────────────────
# HYBRID CNN-LSTM (OPTIONAL)
# ──────────────────────────────────────────────

def build_cnn_lstm(n_timepoints=EPOCH_SAMPLES, n_classes=N_CLASSES):
    """
    Build a hybrid CNN-LSTM model that combines spatial feature
    extraction (Conv1D) with temporal sequence modeling (LSTM).

    Parameters
    ----------
    n_timepoints : int
        Number of time samples per epoch.
    n_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Compiled CNN-LSTM model.
    """
    inputs = keras.Input(shape=(n_timepoints, 1), name="eeg_input")

    # CNN feature extraction
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # LSTM temporal modeling
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="BCI_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Built CNN-LSTM: params={model.count_params():,}")
    return model


# ──────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────

def augment_epoch(epoch, noise_std=0.05, max_shift=10):
    """
    Augment a single EEG epoch for training.

    Applies Gaussian noise and random time-shifting.

    Parameters
    ----------
    epoch : np.ndarray
        1D signal array.
    noise_std : float
        Standard deviation of Gaussian noise.
    max_shift : int
        Maximum time-shift in samples (positive or negative).

    Returns
    -------
    np.ndarray
        Augmented epoch.
    """
    # Add Gaussian noise
    noisy = epoch + np.random.normal(0, noise_std, epoch.shape)

    # Random time-shift
    shift = np.random.randint(-max_shift, max_shift + 1)
    shifted = np.roll(noisy, shift)

    return shifted.astype(np.float32)


def augment_dataset(X, y, augment_factor=2, noise_std=0.05, max_shift=10):
    """
    Augment an entire dataset by generating additional samples.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_timepoints).
    y : np.ndarray
        Shape (n_samples,).
    augment_factor : int
        How many augmented copies to generate per original sample.

    Returns
    -------
    X_aug : np.ndarray
        Original + augmented data.
    y_aug : np.ndarray
        Corresponding labels.
    """
    X_list = [X]
    y_list = [y]

    for _ in range(augment_factor):
        X_new = np.array([augment_epoch(x, noise_std, max_shift) for x in X])
        X_list.append(X_new)
        y_list.append(y.copy())

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    logger.info(f"Augmented: {X.shape[0]} → {X_aug.shape[0]} samples "
                f"(factor={augment_factor})")
    return X_aug, y_aug


# ──────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────

MODEL_BUILDERS = {
    "svm": build_svm,
    "cnn": build_1d_cnn,
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
}


def get_model(name):
    """Get a model builder by name."""
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[name]()


if __name__ == "__main__":
    print("Available models:")
    for name in MODEL_BUILDERS:
        print(f"  - {name}")

    print("\n--- 1D-CNN Summary ---")
    cnn = build_1d_cnn()
    cnn.summary()

    print("\n--- SVM Pipeline ---")
    svm = build_svm()
    print(svm)
