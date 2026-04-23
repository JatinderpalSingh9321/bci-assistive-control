"""
BCI Assistive Control — Buffered EOG Mouse Controller
=====================================================
Windowed blink detection for precise control:
- 4s Detection Window: Count all blinks.
- 3s Cooldown Window: Ignore blinks and wait.

Mapping:
  1 blink  → Left Click
  2 blinks → Right Click
  3+ blinks→ Double Click

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import threading
from collections import deque

import numpy as np
import pyautogui

from src.utils import SERIAL_PORT, BAUD_RATE, SAMPLING_RATE, setup_logger

logger = setup_logger("mouse_eog_buffered")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
BUF_SIZE = 5000
BASELINE_WINDOW = 500
POLL_INTERVAL = 0.02

# ──────────────────────────────────────────────
# SERIAL READER
# ──────────────────────────────────────────────

class SerialReader(threading.Thread):
    def __init__(self, port, baud=BAUD_RATE, simulate=False):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.simulate = simulate
        self.buf = deque(maxlen=BUF_SIZE)
        self.timestamps = deque(maxlen=BUF_SIZE)
        self._running = False
        self._lock = threading.Lock()

    def run(self):
        if self.simulate:
            self._running = True
            logger.info("✓ Simulation mode active (No hardware needed)")
            while self._running:
                # Add dummy noise
                val = 512 + np.random.normal(0, 2)
                now = time.time()
                with self._lock:
                    self.buf.append(val)
                    self.timestamps.append(now)
                time.sleep(1/SAMPLING_RATE)
            return

        import serial as s
        self._running = True
        try:
            ser = s.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            # Clear buffer
            for _ in range(20):
                ser.readline()
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
        with self._lock:
            if len(self.buf) < n:
                return None, None
            data = np.array(list(self.buf)[-n:], dtype=np.float64)
            times = np.array(list(self.timestamps)[-n:])
        return data, times

    def get_actual_sample_rate(self):
        with self._lock:
            if len(self.timestamps) < 100:
                return SAMPLING_RATE
            ts = list(self.timestamps)[-100:]
        dt = ts[-1] - ts[0]
        return (len(ts) - 1) / dt if dt > 0 else SAMPLING_RATE


# ──────────────────────────────────────────────
# BLINK DETECTOR (Threshold Only)
# ──────────────────────────────────────────────

class BlinkDetector:
    def __init__(self, sensitivity=3.1, debug=False):
        self.sensitivity = sensitivity
        self.debug = debug
        self.noise_level = 10.0
        self.baseline = 512.0
        self.min_abs_thresh = 40.0 # Prevents triggering on micro-noise
        self.in_blink = False
        self.blink_start_time = 0
        self.last_blink_end_time = 0
        self.calibrated = False
        self.cal_samples = []
        self.cal_needed = 500

    def calibrate(self, samples):
        self.cal_samples.extend(samples.tolist())
        if len(self.cal_samples) >= self.cal_needed:
            arr = np.array(self.cal_samples[-self.cal_needed:])
            self.baseline = np.mean(arr)
            self.noise_level = np.std(arr)
            self.calibrated = True
            logger.info(f"  Calibrated: baseline={self.baseline:.1f}, noise={self.noise_level:.1f}, thresh={self.noise_level*self.sensitivity:.1f}")
            return True
        return False

    def update_baseline(self, samples):
        if not self.in_blink:
            alpha = 0.01
            self.baseline = (1 - alpha) * self.baseline + alpha * np.mean(samples[-50:])
            self.noise_level = (1 - alpha) * self.noise_level + alpha * np.std(samples[-200:])

    def process(self, samples):
        """Returns True if a valid blink JUST finished."""
        now = time.time()
        if not self.calibrated: return False

        # Refractory period: Don't start a new blink too soon after the last one (debounce)
        if not self.in_blink and (now - self.last_blink_end_time) < 0.25:
            return False

        centered = samples - self.baseline
        max_defl = np.max(np.abs(centered))
        
        # Adaptive threshold with a safety floor
        threshold = max(self.noise_level * self.sensitivity, self.min_abs_thresh)
        is_active = max_defl > threshold

        if self.debug and int(now * 5) % 5 == 0:
            logger.info(f"  [DBG] baseline={self.baseline:.0f} noise={self.noise_level:.1f} defl={max_defl:.1f} act={is_active}")

        blink_finished = False
        if not self.in_blink and is_active:
            self.in_blink = True
            self.blink_start_time = now
        elif self.in_blink and not is_active:
            self.in_blink = False
            self.last_blink_end_time = now
            duration = now - self.blink_start_time
            
            if 0.08 < duration < 2.0: # Catch natural variability and "heavy" blinks
                blink_finished = True
            else:
                if self.debug or duration > 0.02:
                    logger.info(f"\n  [DETECTOR] Blink rejected: duration {duration:.3f}s (target 0.08-2.0s)")
        
        if not self.in_blink:
            self.update_baseline(samples)

        return blink_finished


# ──────────────────────────────────────────────
# BUFFERED CONTROLLER
# ──────────────────────────────────────────────

class BufferedController:
    def __init__(self, detector, reader, det_time, cool_time):
        self.detector = detector
        self.reader = reader
        self.det_time = det_time
        self.cool_time = cool_time
        
        self.blink_count = 0
        self.state = "DETECTING" # "DETECTING" or "COOLDOWN"
        self.phase_start = time.time()
        self.last_countdown = -1
        self.total_actions = 0

    def tick(self):
        now = time.time()
        elapsed = now - self.phase_start

        # Continuous data processing (Batch size 30 for stability)
        data, ts = self.reader.get_recent(30)
        if data is not None:
            if self.detector.process(data):
                if self.state == "DETECTING":
                    self.blink_count += 1
                    logger.info(f"  [BUFFER] Blink #{self.blink_count} detected")
                else:
                    remaining = self.cool_time - elapsed
                    logger.info(f"  [BUFFER] Blink ignored (Cooldown: {remaining:.1f}s left)")

        # Phase transitions
        if self.state == "DETECTING" and elapsed >= self.det_time:
            self.execute_action()
            self.state = "COOLDOWN"
            self.phase_start = now
            self.last_countdown = -1 # Reset countdown trigger

        elif self.state == "COOLDOWN":
            remaining = int(self.cool_time - elapsed + 0.5)
            if remaining > 0 and remaining != self.last_countdown:
                logger.info(f"  [WAIT] Calibration/Reseting... Ready in {remaining}s")
                self.last_countdown = remaining

            if elapsed >= self.cool_time:
                self.blink_count = 0
                self.state = "DETECTING"
                self.phase_start = now
                logger.info("\n" + "#"*50)
                logger.info("  >>> [START] DETECTION WINDOW OPEN - BLINK NOW! <<<")
                logger.info("#"*50 + "\n")

    def execute_action(self):
        if self.blink_count == 0:
            logger.info("  [ACTION] No blinks detected. Skipping.")
            return

        self.total_actions += 1
        logger.info(f"  [ACTION] Processing {self.blink_count} blinks...")

        if self.blink_count == 1:
            pyautogui.click(button='left')
            logger.info(f"  [{self.total_actions:3d}]  ACTION: Left Click")
        elif self.blink_count == 2:
            pyautogui.click(button='right')
            logger.info(f"  [{self.total_actions:3d}]  ACTION: Right Click")
        elif self.blink_count >= 3:
            pyautogui.doubleClick()
            logger.info(f"  [{self.total_actions:3d}]  ACTION: Double Click")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=SERIAL_PORT)
    parser.add_argument("--sensitivity", type=float, default=3.1)
    parser.add_argument("--detection", type=float, default=3.0)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    reader = SerialReader(args.port, simulate=args.simulate)
    reader.start()

    detector = BlinkDetector(sensitivity=args.sensitivity, debug=args.debug)

    # Calibration
    logger.info(f"Calibrating system... Please stay relaxed.")
    time.sleep(2)
    while not detector.calibrated:
        data, ts = reader.get_recent(500)
        if data is not None:
            detector.calibrate(data)
        time.sleep(0.1)

    controller = BufferedController(detector, reader, args.detection, args.cooldown)
    
    logger.info(f"\n✓ Buffer System Ready!")
    logger.info(f"  Mode:      {'SIMULATION' if args.simulate else 'HARDWARE'}")
    logger.info(f"  Window:    {args.detection}s Detection + {args.cooldown}s Cooldown")
    logger.info(f"  Mapping:   1=Left, 2=Right, 3+=Double\n")
    
    logger.info(f"="*40)
    logger.info(f"  [PHASE] >>> Switching to DETECTION ({args.detection}s)")
    logger.info(f"="*40)

    try:
        while True:
            controller.tick()
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info(f"\n✓ Stopped. Total actions: {controller.total_actions}")
    except pyautogui.FailSafeException:
        logger.error("\n✗ Failsafe triggered!")
    finally:
        reader.stop()

if __name__ == "__main__":
    main()
