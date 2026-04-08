"""
BCI Assistive Control — EEG Signal Acquisition
===============================================
Handles serial communication with the Upside Down Labs BioAmp EXG Pill
via Arduino. Provides real-time streaming, signal quality checks,
and a threaded acquisition loop for concurrent processing.

Group No. 7 | 8th Semester Major Project
"""

import argparse
import threading
import queue
import time

import numpy as np
import serial
import serial.tools.list_ports

from src.utils import (
    SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT,
    SAMPLING_RATE, TARGET_SAMPLING_RATE,
    EPOCH_SAMPLES, setup_logger
)

logger = setup_logger("acquisition")

# ──────────────────────────────────────────────
# SERIAL CONNECTION
# ──────────────────────────────────────────────

def list_serial_ports():
    """List all available serial ports on the system."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        logger.warning("No serial ports found.")
    for p in ports:
        logger.info(f"  {p.device:15s}  {p.description}")
    return ports


def connect(port=SERIAL_PORT, baud=BAUD_RATE, timeout=SERIAL_TIMEOUT):
    """
    Open a serial connection to the BioAmp EXG Pill via Arduino.

    Returns
    -------
    serial.Serial
        The open serial connection object.
    """
    logger.info(f"Connecting to {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # Wait for Arduino reset

        # Skip firmware header lines (lines starting with '#')
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line and not line.startswith("#"):
                break
            if line:
                logger.info(f"  Firmware: {line}")

        logger.info(f"✓ Connected to {port}")
        return ser
    except serial.SerialException as e:
        logger.error(f"✗ Failed to connect: {e}")
        raise


def read_sample(ser):
    """
    Read a single sample from the serial stream.

    Returns
    -------
    float or None
        The ADC value, or None if the line was unparseable.
    """
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            return float(line)
    except (ValueError, UnicodeDecodeError):
        pass
    return None


# ──────────────────────────────────────────────
# SIGNAL QUALITY CHECK
# ──────────────────────────────────────────────

def check_signal_quality(ser=None, duration=10, fs=SAMPLING_RATE, port=SERIAL_PORT):
    """
    Collect a short segment of EEG and compute signal-to-noise ratio.

    Parameters
    ----------
    ser : serial.Serial or None
        An open serial connection. If None, opens a new one.
    duration : int
        Seconds to collect.
    fs : int
        Expected sampling rate.
    port : str
        Serial port (used only if `ser` is None).

    Returns
    -------
    dict
        Signal quality metrics: n_samples, mean_amplitude, snr_db, status.
    """
    own_connection = False
    if ser is None:
        ser = connect(port)
        own_connection = True

    logger.info(f"Collecting {duration}s of signal for quality check...")
    samples = []
    start = time.time()

    while time.time() - start < duration:
        val = read_sample(ser)
        if val is not None:
            samples.append(val)

    if own_connection:
        ser.close()

    data = np.array(samples)
    if len(data) < 10:
        logger.error("✗ Too few samples received. Check wiring.")
        return {"n_samples": len(data), "snr_db": 0, "status": "FAIL"}

    # Convert ADC to approximate µV (10-bit ADC, 3.3V ref, ~1000× gain)
    voltage = (data / 1023.0) * 3.3 * 1e6 / 1000  # rough µV estimate

    mean_amp = np.mean(np.abs(voltage))
    variance = np.var(voltage)
    snr = 10 * np.log10(np.mean(voltage ** 2) / (variance + 1e-10))

    effective_rate = len(data) / duration
    status = "GOOD" if snr > 8 else "POOR"

    logger.info(f"  Samples collected:  {len(data)}")
    logger.info(f"  Effective rate:     {effective_rate:.0f} Hz (target: {fs} Hz)")
    logger.info(f"  Mean amplitude:     {mean_amp:.1f} µV")
    logger.info(f"  SNR estimate:       {snr:.1f} dB")
    logger.info(f"  Status:             {'✓ ' + status if status == 'GOOD' else '✗ ' + status + ' — check electrode contact'}")

    return {
        "n_samples": len(data),
        "effective_rate_hz": round(effective_rate, 1),
        "mean_amplitude_uv": round(mean_amp, 2),
        "snr_db": round(snr, 1),
        "status": status,
        "raw_data": data
    }


# ──────────────────────────────────────────────
# REAL-TIME STREAMING (THREADED)
# ──────────────────────────────────────────────

class EEGStreamer:
    """
    Threaded EEG acquisition from the Upside Down Labs kit.

    Continuously reads samples from serial and fills a queue
    with complete epochs (sliding window with 50% overlap).

    Parameters
    ----------
    port : str
        Serial port name.
    buffer_size : int
        Samples per epoch (default: EPOCH_SAMPLES = 500 at 125 Hz).
    overlap : float
        Overlap fraction between consecutive epochs (default: 0.5).
    """

    def __init__(self, port=SERIAL_PORT, buffer_size=EPOCH_SAMPLES, overlap=0.5):
        self.port = port
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.epoch_queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread = None
        self._ser = None

    def start(self):
        """Start the streaming thread."""
        if self._running:
            logger.warning("Streamer already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info("✓ EEG streaming started.")

    def stop(self):
        """Stop the streaming thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._ser and self._ser.is_open:
            self._ser.close()
        logger.info("Streaming stopped.")

    def get_epoch(self, timeout=10):
        """
        Get the next available epoch from the queue.

        Returns
        -------
        np.ndarray
            Shape (buffer_size,) — one epoch of EEG data.
        """
        try:
            return self.epoch_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Epoch queue timeout — no data received.")
            return None

    def _stream_loop(self):
        """Internal streaming loop (runs in a background thread)."""
        try:
            self._ser = connect(self.port)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            self._running = False
            return

        buffer = []
        hop = int(self.buffer_size * (1 - self.overlap))

        while self._running:
            val = read_sample(self._ser)
            if val is not None:
                buffer.append(val)

                if len(buffer) >= self.buffer_size:
                    epoch = np.array(buffer[:self.buffer_size], dtype=np.float32)

                    # Put epoch in queue (non-blocking — drops oldest if full)
                    if self.epoch_queue.full():
                        try:
                            self.epoch_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.epoch_queue.put(epoch)

                    # Slide the window
                    buffer = buffer[hop:]


# ──────────────────────────────────────────────
# SIMULATED DATA (for dev/testing without hardware)
# ──────────────────────────────────────────────

def generate_simulated_epoch(label=0, n_samples=EPOCH_SAMPLES, fs=TARGET_SAMPLING_RATE):
    """
    Generate a simulated motor imagery EEG epoch for testing.

    Produces synthetic alpha/beta oscillations with lateralized
    ERD patterns mimicking real motor imagery signals.

    Parameters
    ----------
    label : int
        0 = LEFT, 1 = RIGHT.
    n_samples : int
        Number of samples per epoch.
    fs : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Simulated EEG epoch of shape (n_samples,).
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

    # Base alpha (10 Hz) and beta (20 Hz) oscillations
    alpha = 15 * np.sin(2 * np.pi * 10 * t)
    beta  = 8  * np.sin(2 * np.pi * 20 * t)

    # Simulate ERD: reduce alpha/beta power for the active hemisphere
    if label == 0:  # LEFT imagery → ERD at C3 (right hemisphere more active)
        alpha *= 0.5  # Suppress alpha (ERD)
        beta  *= 0.6
    else:            # RIGHT imagery → ERD at C4
        alpha *= 0.8
        beta  *= 0.4  # Stronger beta suppression

    # Add pink noise (1/f) characteristic of EEG
    noise = np.cumsum(np.random.randn(n_samples)) * 0.3
    noise -= np.mean(noise)

    # Add white noise
    white = np.random.randn(n_samples) * 2

    signal = alpha + beta + noise + white
    return signal.astype(np.float32)


def generate_simulated_dataset(n_subjects=6, trials_per_subject=80, n_samples=EPOCH_SAMPLES):
    """
    Generate a full simulated dataset for pipeline testing.

    Returns
    -------
    X : np.ndarray
        Shape (n_total_trials, n_samples).
    y : np.ndarray
        Shape (n_total_trials,) — labels (0=LEFT, 1=RIGHT).
    subject_ids : np.ndarray
        Shape (n_total_trials,) — subject ID per trial.
    """
    all_epochs = []
    all_labels = []
    all_subjects = []

    for sub in range(1, n_subjects + 1):
        for trial in range(trials_per_subject):
            label = trial % 2  # Balanced LEFT/RIGHT
            epoch = generate_simulated_epoch(label=label, n_samples=n_samples)
            # Add subject-specific variation
            epoch += np.random.randn() * 3  # DC offset
            epoch *= (0.8 + 0.4 * np.random.rand())  # Amplitude scaling

            all_epochs.append(epoch)
            all_labels.append(label)
            all_subjects.append(sub)

    X = np.array(all_epochs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    subject_ids = np.array(all_subjects, dtype=np.int32)

    logger.info(f"Generated simulated dataset: {X.shape[0]} trials, "
                f"{n_subjects} subjects, {n_samples} samples/trial")
    return X, y, subject_ids


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BCI EEG Acquisition — Upside Down Labs")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    parser.add_argument("--check-quality", action="store_true", help="Run 10s signal quality check")
    parser.add_argument("--port", type=str, default=SERIAL_PORT, help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--duration", type=int, default=10, help="Quality check duration in seconds")
    parser.add_argument("--simulate", action="store_true", help="Generate and save simulated data")
    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
    elif args.check_quality:
        check_signal_quality(port=args.port, duration=args.duration)
    elif args.simulate:
        X, y, sids = generate_simulated_dataset()
        from src.utils import PREPROCESSED_DIR
        np.savez(PREPROCESSED_DIR / "simulated_dataset.npz", X=X, y=y, subject_ids=sids)
        logger.info(f"Saved simulated dataset to {PREPROCESSED_DIR / 'simulated_dataset.npz'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
