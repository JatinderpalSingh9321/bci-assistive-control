"""
BCI Assistive Control — EOG Mouse Controller (Fp1 Forehead)
=============================================================
Real-time blink/wink detection using forehead electrode.

Uses threshold-based peak detection (more reliable than ML for
single-channel EOG):

  Single BLINK  → Left Click
  Double BLINK  → Right Click  (2 blinks within 0.8s)
  Long BLINK    → Double Click (eyes closed > 0.6s)

Electrode Placement:
  IN+  → Fp1 (left forehead, above left eyebrow)
  IN-  → Left earlobe (reference)
  GND  → Right earlobe

Usage:
  python -m src.mouse_control_eog --port COM7
  python -m src.mouse_control_eog --port COM7 --sensitivity 2.5
  python -m src.mouse_control_eog --port COM7 --debug

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import threading
from collections import deque

import numpy as np

from src.utils import SERIAL_PORT, BAUD_RATE, SAMPLING_RATE, setup_logger

logger = setup_logger("mouse_eog")


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# Circular buffer: ~10 seconds of data
BUF_SIZE = 5000

# Baseline window: use last N samples for running mean
BASELINE_WINDOW = 500  # ~2 seconds

# Blink detection
BLINK_COOLDOWN = 0.3       # Min seconds between blink events
MULTI_BLINK_WINDOW = 0.8   # Max seconds between blinks to count as multi-blink

# Polling rate
POLL_INTERVAL = 0.02  # 50 Hz check rate


# ──────────────────────────────────────────────
# SERIAL READER
# ──────────────────────────────────────────────

class SerialReader(threading.Thread):
    """Background thread reading 1-channel serial data."""

    def __init__(self, port, baud=BAUD_RATE):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buf = deque(maxlen=BUF_SIZE)
        self.timestamps = deque(maxlen=BUF_SIZE)
        self._running = False
        self._lock = threading.Lock()

    def run(self):
        import serial as s
        self._running = True
        try:
            ser = s.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            for _ in range(20):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line and not line.startswith("#"):
                    break
            logger.info(f"✓ Serial connected ({self.port})")

            while self._running:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and not line.startswith("#"):
                        val = float(line.split(",")[0])
                        now = time.time()
                        with self._lock:
                            self.buf.append(val)
                            self.timestamps.append(now)
                except (ValueError, UnicodeDecodeError):
                    pass
            ser.close()
        except Exception as e:
            logger.error(f"Serial error: {e}")
            self._running = False

    def stop(self):
        self._running = False

    def get_recent(self, n):
        """Get last n samples and their timestamps."""
        with self._lock:
            if len(self.buf) < n:
                return None, None
            data = np.array(list(self.buf)[-n:], dtype=np.float64)
            times = np.array(list(self.timestamps)[-n:])
        return data, times

    def get_actual_sample_rate(self):
        """Estimate actual sampling rate from timestamps."""
        with self._lock:
            if len(self.timestamps) < 100:
                return SAMPLING_RATE
            ts = list(self.timestamps)[-100:]
        dt = ts[-1] - ts[0]
        if dt > 0:
            return (len(ts) - 1) / dt
        return SAMPLING_RATE


# ──────────────────────────────────────────────
# BLINK DETECTOR (threshold-based)
# ──────────────────────────────────────────────

class BlinkDetector:
    """
    Detects blinks using adaptive threshold on baseline-corrected signal.

    How it works:
    1. Maintain a running baseline (mean of recent samples)
    2. Subtract baseline -> centered signal
    3. When centered signal exceeds threshold -> blink start
    4. When it drops back below -> blink end
    5. Count blinks within a time window:
       - 1 blink  -> left click
       - 2 blinks -> right click
       - 3 blinks -> double click
    """

    def __init__(self, sensitivity=3.0, debug=False):
        self.sensitivity = sensitivity
        self.debug = debug

        # Adaptive threshold state
        self.noise_level = 10.0  # Running estimate of noise floor (std)
        self.baseline = 500.0    # Running baseline (mean)

        # Blink state machine
        self.in_blink = False
        self.blink_start_time = 0
        self.last_blink_time = 0
        self.blink_count = 0     # Count of blinks in current sequence
        self.sequence_start = 0  # When the current blink sequence started

        # Calibration
        self.calibrated = False
        self.cal_samples = []
        self.cal_needed = 500    # ~2 seconds of calibration

    def calibrate(self, samples):
        """Update calibration from a batch of samples."""
        self.cal_samples.extend(samples.tolist())
        if len(self.cal_samples) >= self.cal_needed:
            arr = np.array(self.cal_samples[-self.cal_needed:])
            self.baseline = np.mean(arr)
            self.noise_level = np.std(arr)
            self.calibrated = True
            threshold = self.noise_level * self.sensitivity
            logger.info(f"  Calibrated: baseline={self.baseline:.1f}, "
                       f"noise={self.noise_level:.1f}, "
                       f"threshold={threshold:.1f}")
            return True
        return False

    def update_baseline(self, recent_samples):
        """Update running baseline from recent rest samples (only during non-blink)."""
        if not self.in_blink:
            alpha = 0.01
            new_mean = np.mean(recent_samples[-50:])
            new_std = np.std(recent_samples[-200:]) if len(recent_samples) >= 200 else self.noise_level
            self.baseline = (1 - alpha) * self.baseline + alpha * new_mean
            self.noise_level = (1 - alpha) * self.noise_level + alpha * new_std

    def get_threshold(self):
        """Current adaptive threshold."""
        return self.noise_level * self.sensitivity

    def process(self, samples, timestamps):
        """
        Process a batch of samples and return detected events.

        Returns list of events: 'SINGLE_BLINK', 'DOUBLE_BLINK', 'TRIPLE_BLINK'
        """
        events = []
        now = time.time()

        if not self.calibrated:
            return events

        # Baseline-correct
        centered = samples - self.baseline
        abs_centered = np.abs(centered)
        threshold = self.get_threshold()

        # Check if any sample exceeds threshold
        max_deflection = np.max(abs_centered)
        is_active = max_deflection > threshold

        if self.debug and int(now * 2) % 2 == 0:
            logger.info(
                f"  [DBG] baseline={self.baseline:.0f} noise={self.noise_level:.1f} "
                f"thresh={threshold:.1f} max_defl={max_deflection:.1f} "
                f"active={'YES' if is_active else 'no '} "
                f"in_blink={self.in_blink} count={self.blink_count}"
            )

        # State machine
        if not self.in_blink and is_active:
            # Blink START
            self.in_blink = True
            self.blink_start_time = now

        elif self.in_blink and not is_active:
            # Blink END
            self.in_blink = False
            blink_duration = now - self.blink_start_time

            if blink_duration > 0.10:  # Ignore very short noise spikes (< 100ms)
                # Check if this blink is part of an ongoing sequence
                # We check the gap between the end of the last blink and the start of this one
                if self.blink_count > 0 and (self.blink_start_time - self.last_blink_time) < MULTI_BLINK_WINDOW:
                    # Continue sequence
                    self.blink_count += 1
                else:
                    # Start new sequence
                    self.blink_count = 1
                    self.sequence_start = now

                self.last_blink_time = now

                # Always log blink sequence progress
                logger.info(f"  [SEQ] Blink #{self.blink_count} detected")

                # Triple blink detected immediately
                if self.blink_count >= 3:
                    events.append("TRIPLE_BLINK")
                    self.blink_count = 0
                    logger.info(f"  [SEQ] >>> TRIPLE BLINK!")

        # Check if blink sequence timed out
        if (self.blink_count > 0 and not self.in_blink and
                (now - self.last_blink_time) > MULTI_BLINK_WINDOW):
            if self.blink_count == 1:
                events.append("SINGLE_BLINK")
                if self.debug:
                    logger.info(f"  [DBG] Confirmed: single blink")
            elif self.blink_count == 2:
                events.append("DOUBLE_BLINK")
                if self.debug:
                    logger.info(f"  [DBG] Confirmed: double blink")
            self.blink_count = 0

        # Update baseline (only during rest)
        if not self.in_blink:
            self.update_baseline(samples)

        return events


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run(args):
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02

    # Start reader
    reader = SerialReader(args.port)
    reader.start()

    # Wait for connection
    time.sleep(3)

    # Detect actual sample rate
    actual_rate = reader.get_actual_sample_rate()
    logger.info(f"  Actual sample rate: {actual_rate:.0f} Hz")

    # Banner
    logger.info("")
    logger.info("=" * 60)
    logger.info("  👁 EOG Blink Controller — Fp1 Forehead")
    logger.info("=" * 60)
    logger.info(f"  Port:        {args.port}")
    logger.info(f"  Sensitivity: {args.sensitivity}x (lower = more sensitive)")
    logger.info(f"  Cooldown:    {BLINK_COOLDOWN}s")
    logger.info("")
    logger.info("  Controls:")
    logger.info("    1 Blink   -> Left Click")
    logger.info("    2 Blinks  -> Right Click")
    logger.info("    3 Blinks  -> Double Click")
    logger.info("")
    logger.info("  Calibrating... keep eyes open and relaxed")
    logger.info("=" * 60)

    detector = BlinkDetector(sensitivity=args.sensitivity, debug=args.debug)

    total_actions = 0
    last_action_time = 0
    check_size = 15  # Keep window small to detect the fast gap between double blinks

    try:
        # ── Calibration phase ──
        while not detector.calibrated:
            data, ts = reader.get_recent(detector.cal_needed)
            if data is not None:
                detector.calibrate(data)
            time.sleep(0.2)

        logger.info("")
        logger.info("  ✓ Ready! Start blinking to control mouse.")
        logger.info("    Ctrl+C to stop | Screen corner = failsafe")
        logger.info("")

        # ── Main loop ──
        while True:
            data, ts = reader.get_recent(check_size)

            if data is not None:
                events = detector.process(data, ts)

                for event in events:
                    now = time.time()
                    if now - last_action_time < BLINK_COOLDOWN:
                        continue

                    total_actions += 1
                    last_action_time = now

                    if event == "SINGLE_BLINK":
                        pyautogui.click(button='left')
                        logger.info(
                            f"  [{total_actions:3d}]  [x1] BLINK -> Left Click"
                        )
                    elif event == "DOUBLE_BLINK":
                        pyautogui.click(button='right')
                        logger.info(
                            f"  [{total_actions:3d}]  [x2] DOUBLE -> Right Click"
                        )
                    elif event == "TRIPLE_BLINK":
                        pyautogui.doubleClick()
                        logger.info(
                            f"  [{total_actions:3d}]  [x3] TRIPLE -> Double Click"
                        )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info(f"\n✓ Stopped. Total actions: {total_actions}")
    except pyautogui.FailSafeException:
        logger.error("\n✗ Failsafe triggered (mouse at screen corner)")
    finally:
        reader.stop()


def main():
    parser = argparse.ArgumentParser(
        description="EOG Blink Controller — Threshold-Based (Fp1 Forehead)"
    )
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--sensitivity", type=float, default=3.0,
                        help="Blink detection sensitivity as noise multiplier. "
                             "Lower = more sensitive (default: 3.0, try 2.0-5.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Show real-time signal debug info")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
