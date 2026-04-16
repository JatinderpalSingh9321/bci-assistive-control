"""
BCI Assistive Control — Flask REST API
=======================================
Serves real-time motor imagery predictions via REST endpoints.
Loads pre-trained CNN and SVM models at startup and provides:
  - /health    — API status check
  - /predict   — Classify an EEG epoch
  - /calibrate — Fine-tune model for a specific user

Group No. 7 | 8th Semester Major Project
"""

import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    API_HOST, API_PORT, CONFIDENCE_THRESHOLD,
    CNN_MODEL_PATH, SVM_MODEL_PATH,
    N_CLASSES, CLASS_NAMES, EPOCH_SAMPLES,
    setup_logger
)
from src.preprocessing import EEGPreprocessor

logger = setup_logger("api")

# ──────────────────────────────────────────────
# APP INITIALIZATION
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# Global model references (loaded at startup)
cnn_model = None
svm_model = None
preprocessor = EEGPreprocessor()


def load_models():
    """Load trained models from disk at server startup."""
    global cnn_model, svm_model

    # Load CNN
    if CNN_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            cnn_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
            logger.info(f"✓ CNN model loaded: {CNN_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load CNN model: {e}")
    else:
        logger.warning(f"CNN model not found at {CNN_MODEL_PATH}")

    # Load SVM
    if SVM_MODEL_PATH.exists():
        try:
            with open(SVM_MODEL_PATH, "rb") as f:
                svm_model = pickle.load(f)
            logger.info(f"✓ SVM model loaded: {SVM_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load SVM model: {e}")
    else:
        logger.warning(f"SVM model not found at {SVM_MODEL_PATH}")

    if cnn_model is None and svm_model is None:
        logger.warning("⚠ No models loaded. /predict endpoint will return errors.")
        logger.info("Train models first: python -m src.train --model cnn --simulate --save-final")


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns API status and loaded models."""
    available_models = []
    if cnn_model is not None:
        available_models.append("cnn")
    if svm_model is not None:
        available_models.append("svm")

    return jsonify({
        "status": "ok",
        "models": available_models,
        "epoch_size": EPOCH_SAMPLES,
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single EEG epoch.

    Request body:
        {
            "epoch": [float, float, ...],   # raw or preprocessed EEG samples
            "model": "cnn" | "svm",         # optional, default "cnn"
            "preprocess": true | false      # optional, default false
        }

    Response:
        {
            "prediction": "LEFT" | "RIGHT" | "NEUTRAL",
            "confidence": 0.82,
            "probabilities": {"LEFT": 0.82, "RIGHT": 0.18},
            "latency_ms": 48.3
        }
    """
    start = time.time()

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if "epoch" not in data:
        return jsonify({"error": "Missing 'epoch' field in request body"}), 400

    epoch = np.array(data["epoch"], dtype=np.float32)
    model_choice = data.get("model", "cnn")
    should_preprocess = data.get("preprocess", False)

    # Optional preprocessing
    if should_preprocess:
        try:
            epoch = preprocessor.preprocess_epoch(epoch)
        except Exception as e:
            return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    # Pad or truncate to expected length
    if len(epoch) < EPOCH_SAMPLES:
        epoch = np.pad(epoch, (0, EPOCH_SAMPLES - len(epoch)), mode="edge")
    elif len(epoch) > EPOCH_SAMPLES:
        epoch = epoch[:EPOCH_SAMPLES]

    # Predict motor imagery (LEFT / RIGHT) using trained models
    try:
        if model_choice == "svm" and svm_model is not None:
            from src.feature_extraction import extract_band_power
            features = extract_band_power(epoch).reshape(1, -1)
            probs = svm_model.predict_proba(features)[0]
        elif cnn_model is not None:
            epoch_input = epoch[np.newaxis, :, np.newaxis]  # (1, T, 1)
            probs = cnn_model.predict(epoch_input, verbose=0)[0]
        elif svm_model is not None:
            # Fallback to SVM if CNN not available
            from src.feature_extraction import extract_band_power
            features = extract_band_power(epoch).reshape(1, -1)
            probs = svm_model.predict_proba(features)[0]
        else:
            return jsonify({"error": "No models loaded. Train a model first."}), 503

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Determine prediction with confidence thresholding
    confidence = float(np.max(probs))
    predicted_idx = int(np.argmax(probs))

    if confidence < CONFIDENCE_THRESHOLD:
        label = "NEUTRAL"
    else:
        label = CLASS_NAMES[predicted_idx]

    latency_ms = (time.time() - start) * 1000

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 3),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 3)
            for i in range(len(CLASS_NAMES))
        },
        "latency_ms": round(latency_ms, 1),
        "model_used": model_choice
    })


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """
    Fine-tune the CNN model for a specific user.

    Request body:
        {
            "epochs": [[float, ...], [float, ...], ...],
            "labels": [0, 1, 0, ...]
        }

    Response:
        {"status": "calibrated", "n_samples": 30}
    """
    global cnn_model

    if cnn_model is None:
        return jsonify({"error": "No CNN model loaded for calibration"}), 503

    try:
        data = request.get_json(force=True)
        epochs = np.array(data["epochs"], dtype=np.float32)
        labels = np.array(data["labels"], dtype=np.int32)
    except Exception as e:
        return jsonify({"error": f"Invalid calibration data: {str(e)}"}), 400

    if len(epochs) != len(labels):
        return jsonify({"error": "Epochs and labels must have same length"}), 400

    if len(epochs) < 5:
        return jsonify({"error": "Need at least 5 calibration trials"}), 400

    try:
        import tensorflow as tf

        # Freeze all layers except the final classifier head
        for layer in cnn_model.layers[:-2]:
            layer.trainable = False

        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Prepare data
        X_cal = epochs[..., np.newaxis]
        y_cal = tf.keras.utils.to_categorical(labels, N_CLASSES)

        # Fine-tune (few epochs, small learning rate)
        cnn_model.fit(X_cal, y_cal, epochs=5, batch_size=8, verbose=0)

        # Unfreeze for future predictions
        for layer in cnn_model.layers:
            layer.trainable = True

        logger.info(f"✓ Calibrated with {len(labels)} samples")

        return jsonify({
            "status": "calibrated",
            "n_samples": len(labels)
        })

    except Exception as e:
        return jsonify({"error": f"Calibration failed: {str(e)}"}), 500


@app.route("/simulate", methods=["GET"])
def simulate_prediction():
    """
    Generate a simulated motor imagery prediction (for demo/testing).
    Only produces LEFT or RIGHT — blink/wink detection is handled
    separately by the camera-based eye_tracker module.
    """
    from src.acquisition import generate_simulated_epoch

    label = np.random.randint(0, 2)
    epoch = generate_simulated_epoch(label=label)

    conf = 0.65 + np.random.random() * 0.25
    if label == 0:
        probs = [conf, 1 - conf]
    else:
        probs = [1 - conf, conf]

    return jsonify({
        "prediction": CLASS_NAMES[label],
        "confidence": round(float(conf), 3),
        "probabilities": {
            "LEFT": round(float(probs[0]), 3),
            "RIGHT": round(float(probs[1]), 3)
        },
        "latency_ms": round(np.random.uniform(30, 80), 1),
        "model_used": "simulated",
        "epoch_preview": epoch[:20].tolist()
    })


# ──────────────────────────────────────────────
# SERVER STARTUP
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import print_banner
    print_banner()

    logger.info("Starting BCI API server...")
    load_models()

    logger.info(f"Server running at http://{API_HOST}:{API_PORT}")
    logger.info("Endpoints:")
    logger.info("  GET  /health     — Status check")
    logger.info("  POST /predict    — Classify EEG epoch")
    logger.info("  POST /calibrate  — Fine-tune for user")
    logger.info("  GET  /simulate   — Simulated prediction (demo)")

    app.run(host=API_HOST, port=API_PORT, debug=False)
