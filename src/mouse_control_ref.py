"""
BCI Assistive Control — Real-Time Mouse Controller (Referential + Webcam)
==========================================================================
Controls mouse cursor using:
  - 1× BioAmp EXG Pill (C3 referential to earlobe) → LEFT/RIGHT/UP/DOWN
  - Webcam (MediaPipe eye tracking) → BLINK/WINK for clicks

This matches the actual hardware the user has.
No code changes to Arduino firmware needed — uses same eeg_stream.ino.

Usage:
  python -m src.mouse_control_ref --simulate --speed 30
  python -m src.mouse_control_ref --port COM7 --speed 40
  python -m src.mouse_control_ref --port COM7 --speed 40 --preview

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import pickle
import threading
from collections import deque, Counter

import numpy as np
import pyautogui

from src.utils import SERIAL_PORT, BAUD_RATE, SAMPLING_RATE, MODELS_DIR, setup_logger
from src.train_referential import extract_features

logger = setup_logger("mouse_ref")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# ──────────────────────────────────────────────
# MAPPING
# ──────────────────────────────────────────────

MI_ACTIONS = {
    0: ("LEFT",  (-1,  0)),
    1: ("RIGHT", ( 1,  0)),
    2: ("UP",    ( 0, -1)),
    3: ("DOWN",  ( 0,  1)),
}

SYMBOLS = {0: "←", 1: "→", 2: "↑", 3: "↓"}

# Prediction settings
WINDOW_SEC = 2.0
WINDOW_N   = int(WINDOW_SEC * SAMPLING_RATE)   # 500 samples
STRIDE_SEC = 0.4
MIN_CONFIDENCE = 0.35
VOTE_SIZE = 3  # Majority voting window


# ──────────────────────────────────────────────
# SERIAL READER THREAD (1-channel)
# ──────────────────────────────────────────────

class EEGReader(threading.Thread):
    """Background thread for reading 1-channel EEG via serial."""

    def __init__(self, port, baud=BAUD_RATE, buf_size=5000):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buf = deque(maxlen=buf_size)
        self._running = False
        self._lock = threading.Lock()

    def run(self):
        import serial as s
        self._running = True
        try:
            ser = s.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            # Skip header
            for _ in range(20):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line and not line.startswith("#"):
                    break

            logger.info(f"✓ EEG reader connected ({self.port})")

            while self._running:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and not line.startswith("#"):
                        val = float(line.split(",")[0])  # Works with both firmwares
                        with self._lock:
                            self.buf.append(val)
                except (ValueError, UnicodeDecodeError):
                    pass
            ser.close()
        except Exception as e:
            logger.error(f"Serial error: {e}")
            self._running = False

    def stop(self):
        self._running = False

    def get_window(self, n):
        with self._lock:
            if len(self.buf) < n:
                return None
            return np.array(list(self.buf)[-n:], dtype=np.float32)


class SimReader:
    """Simulated EEG for testing."""

    def __init__(self):
        self.buf = deque(maxlen=5000)
        self._running = True
        self._class = 0
        self._timer = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        logger.info("✓ Simulated reader started")

    def _loop(self):
        from src.experiment_referential import simulate_referential_epoch
        while self._running:
            if time.time() - self._timer > 3:
                self._class = np.random.randint(0, 4)
                self._timer = time.time()
            chunk = simulate_referential_epoch(self._class, 25)
            with threading.Lock():
                for v in chunk:
                    self.buf.append(float(v))
            time.sleep(25 / SAMPLING_RATE)

    def stop(self):
        self._running = False

    def get_window(self, n):
        if len(self.buf) < n:
            return None
        return np.array(list(self.buf)[-n:], dtype=np.float32)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run(args):
    # Load model
    model_path = MODELS_DIR / "mi_referential_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run first: python -m src.train_referential --simulate")
        return

    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model = pkg["model"]
    logger.info(f"✓ Model loaded (accuracy: {pkg.get('cv_accuracy', '?'):.1%})")

    # Start eye tracker if available
    eye_tracker = None
    if not args.no_camera:
        try:
            from src.eye_tracker import EyeTracker
            eye_tracker = EyeTracker(
                camera_index=args.camera,
                show_preview=args.preview
            )
            eye_tracker.start()
            logger.info("✓ Eye tracker started (BLINK→click, WINK→hold)")
        except Exception as e:
            logger.warning(f"Camera unavailable: {e}")
            logger.warning("Continuing without camera (movement only)")

    # Start data source
    if args.simulate:
        reader = SimReader()
        reader.start()
    else:
        reader = EEGReader(args.port)
        reader.start()

    time.sleep(3)  # Fill buffer

    # Banner
    logger.info("")
    logger.info("=" * 60)
    logger.info("  🧠 BCI Mouse Control — Referential C3 + Webcam")
    logger.info("=" * 60)
    logger.info(f"  Mode: {'SIMULATED' if args.simulate else 'HARDWARE'}")
    logger.info(f"  Speed: {args.speed} px | Window: {WINDOW_SEC}s")
    logger.info("")
    logger.info("  Brain (C3 EEG):")
    logger.info("    LEFT  imagery ← cursor left")
    logger.info("    RIGHT imagery → cursor right")
    logger.info("    UP    imagery ↑ cursor up")
    logger.info("    DOWN  imagery ↓ cursor down")
    if eye_tracker:
        logger.info("  Eyes (Webcam):")
        logger.info("    BLINK      → left click")
        logger.info("    WINK       → hold / release")
    logger.info("")
    logger.info("  Ctrl+C to stop | Screen corner = failsafe")
    logger.info("=" * 60)

    speed = args.speed
    actions = 0
    votes = deque(maxlen=VOTE_SIZE)
    is_holding = False
    last_eye_t = 0

    try:
        while True:
            # ── Eye events ──
            if eye_tracker:
                while True:
                    ev = eye_tracker.get_event(timeout=0.01)
                    if ev is None:
                        break
                    now = time.time()
                    if now - last_eye_t < 1.0:
                        continue
                    last_eye_t = now
                    actions += 1

                    if ev == "BLINK":
                        if is_holding:
                            pyautogui.mouseUp(button='left')
                            is_holding = False
                        pyautogui.click(button='left')
                        logger.info(f"[CLICK]  Blink → Left Click (#{actions})")
                    elif ev == "WINK":
                        if not is_holding:
                            pyautogui.mouseDown(button='left')
                            is_holding = True
                            logger.info(f"[GRAB]   Wink → Hold (#{actions})")
                        else:
                            pyautogui.mouseUp(button='left')
                            is_holding = False
                            logger.info(f"[DROP]   Wink → Release (#{actions})")

            # ── Brain prediction ──
            window = reader.get_window(WINDOW_N)
            if window is not None:
                try:
                    feats = extract_features(window).reshape(1, -1)
                    pred = model.predict(feats)[0]
                    proba = model.predict_proba(feats)[0]
                    conf = proba[pred]

                    if conf >= MIN_CONFIDENCE:
                        votes.append(pred)

                        if len(votes) >= 2:
                            majority = Counter(votes).most_common(1)[0]
                            if majority[1] >= 2:
                                direction = majority[0]
                                name, (dx, dy) = MI_ACTIONS[direction]
                                mx, my = dx * speed, dy * speed
                                pyautogui.moveRel(mx, my, duration=0.05)
                                actions += 1

                                if actions % 3 == 1:
                                    state = "DRAG" if is_holding else "MOVE"
                                    logger.info(
                                        f"[{state}]  {SYMBOLS[direction]}  {name:5s} "
                                        f"({conf:.0%}) "
                                        f"dx={mx:+d} dy={my:+d} (#{actions})"
                                    )
                except Exception:
                    pass

            time.sleep(STRIDE_SEC)

    except KeyboardInterrupt:
        if is_holding:
            pyautogui.mouseUp(button='left')
        logger.info(f"\n✓ Stopped. Total actions: {actions}")
    except pyautogui.FailSafeException:
        if is_holding:
            pyautogui.mouseUp(button='left')
        logger.error("\n✗ Failsafe triggered!")
    finally:
        reader.stop()
        if eye_tracker:
            eye_tracker.stop()


def main():
    parser = argparse.ArgumentParser(
        description="BCI Mouse Control — 1-Channel C3 + Webcam"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Simulated mode (no hardware)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--speed", type=int, default=30,
                        help="Pixels per move (default: 30)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable webcam (brain only)")
    parser.add_argument("--preview", action="store_true",
                        help="Show camera preview")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
