"""
BCI Assistive Control — 4-Direction Mouse Controller
======================================================
Real-time mouse control using 4-channel EEG:

  C3/C4 (Motor Imagery) → Cursor Movement:
    LEFT  imagery → move cursor left
    RIGHT imagery → move cursor right
    UP    imagery → move cursor up
    DOWN  imagery → move cursor down

  Fp1/Fp2 (EOG) → Mouse Actions:
    BLINK       → left click
    WINK_LEFT   → right click
    WINK_RIGHT  → double click

Requires:
  - Trained models: mi_4class_model.pkl, eog_3class_model.pkl
  - Arduino running eeg_4channel.ino firmware
  - Or use --simulate for demo mode

Usage:
  python -m src.mouse_control_4ch --simulate --speed 30
  python -m src.mouse_control_4ch --port COM7 --speed 40
  python -m src.mouse_control_4ch --simulate --speed 30 --no-eog

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import pickle
import threading
import queue
from collections import deque

import numpy as np
import pyautogui

from src.utils import (
    SERIAL_PORT, BAUD_RATE, SAMPLING_RATE, MODELS_DIR,
    setup_logger
)
from src.train_4ch import extract_mi_features, extract_eog_features

logger = setup_logger("mouse_4ch")

# Safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

MI_ACTIONS = {
    0: ("LEFT",  (-1,  0)),   # dx, dy
    1: ("RIGHT", ( 1,  0)),
    2: ("UP",    ( 0, -1)),
    3: ("DOWN",  ( 0,  1)),
}

EOG_ACTIONS = {
    0: "CLICK",         # BLINK → left click
    1: "RIGHT_CLICK",   # WINK_LEFT → right click
    2: "DOUBLE_CLICK",  # WINK_RIGHT → double click
}

# Sliding window for prediction
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
STRIDE_SECONDS = 0.5
STRIDE_SAMPLES = int(STRIDE_SECONDS * SAMPLING_RATE)

# EOG detection window (shorter — blinks are fast)
EOG_WINDOW_SECONDS = 1.0
EOG_WINDOW_SAMPLES = int(EOG_WINDOW_SECONDS * SAMPLING_RATE)

# Confidence thresholds
MI_CONFIDENCE_THRESHOLD = 0.4   # Lower for 4-class
EOG_CONFIDENCE_THRESHOLD = 0.6
EOG_COOLDOWN_SECONDS = 1.5  # Prevent rapid-fire clicks


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

def load_models():
    """Load trained MI and EOG models."""
    models = {}

    mi_path = MODELS_DIR / "mi_4class_model.pkl"
    if mi_path.exists():
        with open(mi_path, "rb") as f:
            mi_data = pickle.load(f)
        models["mi"] = mi_data["model"]
        logger.info(f"✓ MI model loaded (accuracy: {mi_data.get('train_accuracy', '?'):.1%})")
    else:
        logger.warning(f"MI model not found: {mi_path}")

    eog_path = MODELS_DIR / "eog_3class_model.pkl"
    if eog_path.exists():
        with open(eog_path, "rb") as f:
            eog_data = pickle.load(f)
        models["eog"] = eog_data["model"]
        logger.info(f"✓ EOG model loaded (accuracy: {eog_data.get('train_accuracy', '?'):.1%})")
    else:
        logger.warning(f"EOG model not found: {eog_path}")

    return models


# ──────────────────────────────────────────────
# SERIAL READER (BACKGROUND THREAD)
# ──────────────────────────────────────────────

class SerialReader(threading.Thread):
    """Background thread that reads 4-channel serial data into a shared buffer."""

    def __init__(self, port, baud, buffer_size=5000):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buffer = deque(maxlen=buffer_size)
        self._running = False
        self._lock = threading.Lock()

    def run(self):
        import serial as ser_lib
        self._running = True

        try:
            ser = ser_lib.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)

            # Skip header
            for _ in range(20):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line and not line.startswith("#"):
                    break

            logger.info(f"✓ Serial reader connected to {self.port}")

            while self._running:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and not line.startswith("#"):
                        parts = line.split(",")
                        if len(parts) >= 4:
                            sample = [float(p) for p in parts[:4]]
                        elif len(parts) == 1:
                            val = float(parts[0])
                            sample = [val, val, val, val]
                        else:
                            continue

                        with self._lock:
                            self.buffer.append(sample)
                except (ValueError, UnicodeDecodeError):
                    pass

            ser.close()
        except Exception as e:
            logger.error(f"Serial reader error: {e}")
            self._running = False

    def stop(self):
        self._running = False

    def get_window(self, n_samples):
        """Get the last n_samples from the buffer."""
        with self._lock:
            if len(self.buffer) < n_samples:
                return None
            data = list(self.buffer)[-n_samples:]
        return np.array(data, dtype=np.float32)


# ──────────────────────────────────────────────
# SIMULATED DATA STREAM
# ──────────────────────────────────────────────

class SimulatedReader:
    """Simulated 4-channel data stream for testing without hardware."""

    def __init__(self):
        self.buffer = deque(maxlen=5000)
        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._lock = threading.Lock()
        self._current_class = 0
        self._class_timer = time.time()

    def start(self):
        self._thread.start()
        logger.info("✓ Simulated reader started")

    def _generate_loop(self):
        from src.experiment_4ch import generate_simulated_4ch

        while self._running:
            # Change simulated direction every 3 seconds
            if time.time() - self._class_timer > 3.0:
                self._current_class = np.random.randint(0, 7)
                self._class_timer = time.time()

            # Generate a small chunk
            chunk = generate_simulated_4ch(self._current_class, 25)
            with self._lock:
                for sample in chunk:
                    self.buffer.append(sample.tolist())

            time.sleep(25 / SAMPLING_RATE)  # ~real-time

    def stop(self):
        self._running = False

    def get_window(self, n_samples):
        with self._lock:
            if len(self.buffer) < n_samples:
                return None
            data = list(self.buffer)[-n_samples:]
        return np.array(data, dtype=np.float32)


# ──────────────────────────────────────────────
# MAIN CONTROL LOOP
# ──────────────────────────────────────────────

def run_mouse_control(args):
    """Main real-time mouse control loop."""

    # ── Load models ──
    models = load_models()

    if not models:
        logger.error("No models found. Run: python -m src.train_4ch --simulate")
        return

    has_mi = "mi" in models
    has_eog = "eog" in models and not args.no_eog

    # ── Setup data source ──
    if args.simulate:
        reader = SimulatedReader()
        reader.start()
    else:
        reader = SerialReader(args.port, BAUD_RATE)
        reader.start()

    # Wait for buffer to fill
    logger.info("Waiting for data buffer to fill...")
    time.sleep(3)

    # ── Banner ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BCI 4-Direction Mouse Control")
    logger.info("=" * 60)
    logger.info(f"  Mode:    {'SIMULATED' if args.simulate else 'HARDWARE'}")
    logger.info(f"  Speed:   {args.speed} px per prediction")
    logger.info(f"  MI Model:  {'✓' if has_mi else '✗'} (LEFT/RIGHT/UP/DOWN)")
    logger.info(f"  EOG Model: {'✓' if has_eog else '✗'} (BLINK/WINK)")
    logger.info("")
    logger.info("  Cursor Movement (Motor Imagery):")
    logger.info("    LEFT  imagery → ← cursor left")
    logger.info("    RIGHT imagery → → cursor right")
    logger.info("    UP    imagery → ↑ cursor up")
    logger.info("    DOWN  imagery → ↓ cursor down")
    if has_eog:
        logger.info("  Mouse Actions (Eye Actions):")
        logger.info("    BLINK       → Left click")
        logger.info("    WINK LEFT   → Right click")
        logger.info("    WINK RIGHT  → Double click")
    logger.info("")
    logger.info("  Safety: Move mouse to screen corner to abort")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    # ── State ──
    speed = args.speed
    total_actions = 0
    last_eog_time = 0
    prediction_history = deque(maxlen=5)  # Smoothing

    try:
        while True:
            # ── Motor Imagery Prediction ──
            if has_mi:
                window = reader.get_window(WINDOW_SAMPLES)
                if window is not None and window.shape[0] >= WINDOW_SAMPLES:
                    try:
                        features = extract_mi_features(window).reshape(1, -1)
                        mi_model = models["mi"]
                        pred = mi_model.predict(features)[0]
                        proba = mi_model.predict_proba(features)[0]
                        confidence = proba[pred]

                        if confidence >= MI_CONFIDENCE_THRESHOLD:
                            prediction_history.append(pred)

                            # Majority vote from recent predictions
                            if len(prediction_history) >= 2:
                                from collections import Counter
                                votes = Counter(prediction_history)
                                majority_pred = votes.most_common(1)[0][0]
                                majority_count = votes.most_common(1)[0][1]

                                if majority_count >= 2:
                                    action_name, (dx, dy) = MI_ACTIONS[majority_pred]
                                    move_x = dx * speed
                                    move_y = dy * speed
                                    pyautogui.moveRel(move_x, move_y, duration=0.05)
                                    total_actions += 1

                                    if total_actions % 5 == 1:
                                        logger.info(
                                            f"[MOVE] {action_name:5s} "
                                            f"({confidence:.0%}) "
                                            f"dx={move_x:+d} dy={move_y:+d} "
                                            f"(#{total_actions})"
                                        )
                    except Exception as e:
                        if total_actions == 0:
                            logger.warning(f"MI prediction failed: {e}")

            # ── EOG Prediction ──
            if has_eog:
                eog_window = reader.get_window(EOG_WINDOW_SAMPLES)
                now = time.time()

                if (eog_window is not None and
                        eog_window.shape[0] >= EOG_WINDOW_SAMPLES and
                        now - last_eog_time > EOG_COOLDOWN_SECONDS):

                    # Check if there's actually EOG activity (amplitude spike)
                    fp1 = eog_window[:, 2] if eog_window.shape[1] > 2 else eog_window[:, 0]
                    fp2 = eog_window[:, 3] if eog_window.shape[1] > 3 else eog_window[:, 0]
                    peak_amplitude = max(np.max(np.abs(fp1)), np.max(np.abs(fp2)))
                    mean_amplitude = np.mean(np.abs(np.concatenate([fp1, fp2])))

                    # Only classify if there's a significant peak (blink/wink)
                    if peak_amplitude > mean_amplitude * 3:
                        try:
                            features = extract_eog_features(eog_window).reshape(1, -1)
                            eog_model = models["eog"]
                            pred = eog_model.predict(features)[0]
                            proba = eog_model.predict_proba(features)[0]
                            confidence = proba[pred]

                            if confidence >= EOG_CONFIDENCE_THRESHOLD:
                                action = EOG_ACTIONS[pred]
                                last_eog_time = now
                                total_actions += 1

                                if action == "CLICK":
                                    pyautogui.click(button='left')
                                    logger.info(f"[CLICK]  Blink → Left Click "
                                               f"({confidence:.0%}) (#{total_actions})")
                                elif action == "RIGHT_CLICK":
                                    pyautogui.click(button='right')
                                    logger.info(f"[RCLICK] Wink Left → Right Click "
                                               f"({confidence:.0%}) (#{total_actions})")
                                elif action == "DOUBLE_CLICK":
                                    pyautogui.doubleClick()
                                    logger.info(f"[DBLCLK] Wink Right → Double Click "
                                               f"({confidence:.0%}) (#{total_actions})")
                        except Exception as e:
                            pass

            # Control loop rate
            time.sleep(STRIDE_SECONDS)

    except KeyboardInterrupt:
        logger.info("\n✓ Mouse control stopped by user (Ctrl+C)")
    except pyautogui.FailSafeException:
        logger.error("\n✗ Failsafe! Mouse at screen corner. Stopping.")
    finally:
        reader.stop()
        logger.info(f"Total actions: {total_actions}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BCI 4-Direction Mouse Control — Brain + Eyes"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data (no hardware)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--speed", type=int, default=30,
                        help="Pixels per movement (default: 30)")
    parser.add_argument("--no-eog", action="store_true",
                        help="Disable EOG (movement only, no clicks)")
    args = parser.parse_args()

    run_mouse_control(args)


if __name__ == "__main__":
    main()
