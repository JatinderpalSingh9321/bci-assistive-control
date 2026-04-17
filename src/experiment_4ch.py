"""
BCI Assistive Control — 4-Channel Data Collection Experiment
=============================================================
Collects motor imagery (LEFT/RIGHT/UP/DOWN) from C3/C4 and
eye actions (BLINK/WINK_LEFT/WINK_RIGHT) from Fp1/Fp2 using
a visual cue paradigm with Pygame.

4-Electrode Setup:
  C3  (A0) — left motor cortex   → motor imagery
  C4  (A1) — right motor cortex  → motor imagery
  Fp1 (A2) — left forehead       → blink / wink detection
  Fp2 (A3) — right forehead      → blink / wink detection

Motor Imagery Mapping:
  LEFT  — imagine squeezing left fist  (ERD at C4, contralateral)
  RIGHT — imagine squeezing right fist (ERD at C3, contralateral)
  UP    — imagine curling tongue upward (bilateral pattern change)
  DOWN  — imagine wiggling toes / feet  (vertex activation, some at C3/C4)

Eye Action Mapping:
  BLINK      — close both eyes briefly     → left click
  WINK_LEFT  — close left eye only         → scroll up / hold
  WINK_RIGHT — close right eye only        → scroll down / release

Usage:
  python -m src.experiment_4ch --subject 1 --port COM7
  python -m src.experiment_4ch --subject 1 --simulate
  python -m src.experiment_4ch --subject 1 --simulate --block eog

Group No. 7 | 8th Semester Major Project
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    import serial
except ImportError:
    serial = None

from src.utils import (
    SERIAL_PORT, BAUD_RATE, SAMPLING_RATE,
    PROJECT_ROOT, RAW_DATA_DIR,
    setup_logger
)

logger = setup_logger("experiment_4ch")

# ──────────────────────────────────────────────
# EXPERIMENT PARAMETERS
# ──────────────────────────────────────────────

N_CHANNELS = 4  # C3, C4, Fp1, Fp2
CHANNEL_NAMES = ["C3", "C4", "Fp1", "Fp2"]

# Motor imagery classes (from C3/C4)
MI_CLASSES = {
    0: {"name": "LEFT",  "instruction": "Squeeze LEFT Fist",  "arrow": "←"},
    1: {"name": "RIGHT", "instruction": "Squeeze RIGHT Fist", "arrow": "→"},
    2: {"name": "UP",    "instruction": "Curl Tongue Upward",  "arrow": "↑"},
    3: {"name": "DOWN",  "instruction": "Wiggle Your Toes",    "arrow": "↓"},
}

# Eye action classes (from Fp1/Fp2)
EOG_CLASSES = {
    4: {"name": "BLINK",      "instruction": "Close BOTH Eyes Briefly", "icon": "◉◉"},
    5: {"name": "WINK_LEFT",  "instruction": "Close LEFT Eye Only",     "icon": "◉ ○"},
    6: {"name": "WINK_RIGHT", "instruction": "Close RIGHT Eye Only",    "icon": "○ ◉"},
}

ALL_CLASSES = {**MI_CLASSES, **EOG_CLASSES}

# Timing (seconds)
READY_DURATION   = 2.0   # "Get Ready" screen
CUE_DURATION     = 1.0   # Show direction/action
IMAGINE_DURATION = 4.0   # Motor imagery / eye action period (data recorded)
REST_DURATION    = 2.0   # Inter-trial rest

# Trials
MI_TRIALS_PER_CLASS  = 25   # 25 × 4 = 100 motor imagery trials
EOG_TRIALS_PER_CLASS = 20   # 20 × 3 = 60 eye action trials

# ──────────────────────────────────────────────
# DISPLAY CONFIGURATION
# ──────────────────────────────────────────────

SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 800

# Colors (dark theme with accent colors per direction)
COLOR_BG       = (15, 15, 25)
COLOR_TEXT     = (210, 210, 220)
COLOR_LEFT     = (0, 200, 255)     # Cyan
COLOR_RIGHT    = (255, 166, 0)     # Orange
COLOR_UP       = (100, 255, 150)   # Green
COLOR_DOWN     = (255, 100, 100)   # Red
COLOR_BLINK    = (255, 255, 100)   # Yellow
COLOR_WINK     = (200, 150, 255)   # Purple
COLOR_IMAGINE  = (100, 255, 100)   # Bright green
COLOR_READY    = (180, 180, 190)
COLOR_REST     = (60, 60, 80)
COLOR_ACCENT   = (233, 69, 96)     # Red accent
COLOR_PROGRESS = (0, 180, 255)

DIRECTION_COLORS = {
    0: COLOR_LEFT, 1: COLOR_RIGHT, 2: COLOR_UP, 3: COLOR_DOWN,
    4: COLOR_BLINK, 5: COLOR_WINK, 6: COLOR_WINK,
}


def init_display():
    """Initialize Pygame display and fonts."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🧠 BCI 4-Channel Data Collection — Motor Imagery + EOG")

    fonts = {
        "title":   pygame.font.SysFont("Segoe UI", 44, bold=True),
        "cue":     pygame.font.SysFont("Segoe UI", 120, bold=True),
        "action":  pygame.font.SysFont("Segoe UI", 36),
        "status":  pygame.font.SysFont("Segoe UI", 26),
        "info":    pygame.font.SysFont("Consolas", 18),
        "small":   pygame.font.SysFont("Consolas", 14),
    }
    return screen, fonts


def draw_text(screen, font, text, color, x, y, center=True):
    """Draw text at position. If center=True, x,y is center point."""
    surface = font.render(text, True, color)
    if center:
        rect = surface.get_rect(center=(x, y))
    else:
        rect = surface.get_rect(topleft=(x, y))
    screen.blit(surface, rect)


def draw_progress_bar(screen, current, total, y=SCREEN_HEIGHT - 40):
    """Draw a progress bar."""
    bar_w = SCREEN_WIDTH - 100
    bar_h = 14
    x = 50
    progress = current / max(total, 1)

    pygame.draw.rect(screen, (40, 40, 60), (x, y, bar_w, bar_h), border_radius=7)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        pygame.draw.rect(screen, COLOR_PROGRESS, (x, y, fill_w, bar_h), border_radius=7)

    # Percentage
    pct_text = f"{progress * 100:.0f}%"
    font = pygame.font.SysFont("Consolas", 14)
    draw_text(screen, font, pct_text, COLOR_TEXT, SCREEN_WIDTH // 2, y - 12)


def draw_arrow(screen, direction, cx, cy, size=80, color=(255, 255, 255)):
    """Draw a large directional arrow."""
    if direction == 0:  # LEFT
        points = [(cx + size, cy - size//2), (cx - size, cy), (cx + size, cy + size//2)]
    elif direction == 1:  # RIGHT
        points = [(cx - size, cy - size//2), (cx + size, cy), (cx - size, cy + size//2)]
    elif direction == 2:  # UP
        points = [(cx - size//2, cy + size), (cx, cy - size), (cx + size//2, cy + size)]
    elif direction == 3:  # DOWN
        points = [(cx - size//2, cy - size), (cx, cy + size), (cx + size//2, cy - size)]
    else:
        return
    pygame.draw.polygon(screen, color, points)
    # Outline
    pygame.draw.polygon(screen, (255, 255, 255), points, 3)


def draw_signal_monitor(screen, samples_buffer, y_start=650):
    """Draw a small real-time signal trace at the bottom."""
    if len(samples_buffer) < 2:
        return

    font = pygame.font.SysFont("Consolas", 12)
    ch_colors = [COLOR_LEFT, COLOR_RIGHT, COLOR_BLINK, COLOR_WINK]
    ch_height = 25
    plot_width = SCREEN_WIDTH - 200

    for ch_idx in range(N_CHANNELS):
        y_center = y_start + ch_idx * (ch_height + 5)

        # Channel label
        draw_text(screen, font, CHANNEL_NAMES[ch_idx], ch_colors[ch_idx],
                  40, y_center, center=True)

        # Plot last 200 samples
        data = [s[ch_idx] for s in samples_buffer[-200:] if len(s) > ch_idx]
        if len(data) < 2:
            continue

        data = np.array(data)
        # Normalize to display range
        if np.ptp(data) > 0:
            norm = (data - np.mean(data)) / (np.ptp(data) + 1e-10) * ch_height
        else:
            norm = np.zeros_like(data)

        points = []
        for i, v in enumerate(norm):
            px = 70 + int(i * plot_width / len(norm))
            py = int(y_center + v)
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(screen, ch_colors[ch_idx], False, points, 1)


# ──────────────────────────────────────────────
# SERIAL COMMUNICATION (4-CHANNEL)
# ──────────────────────────────────────────────

def connect_4ch(port=SERIAL_PORT, baud=BAUD_RATE):
    """Open serial connection to 4-channel Arduino firmware."""
    logger.info(f"Connecting to {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for Arduino reset

        # Skip firmware header lines
        for _ in range(20):
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line and line.startswith("#"):
                logger.info(f"  Firmware: {line}")
            elif line and "," in line:
                # First data line — we're connected
                break

        logger.info(f"✓ Connected to {port} (4-channel mode)")
        return ser
    except serial.SerialException as e:
        logger.error(f"✗ Failed to connect: {e}")
        raise


def read_4ch_sample(ser):
    """
    Read one 4-channel sample from serial.

    Returns
    -------
    list of float or None
        [c3, c4, fp1, fp2] ADC values, or None if parse failed.
    """
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            parts = line.split(",")
            if len(parts) >= 4:
                return [float(p) for p in parts[:4]]
            elif len(parts) == 1:
                # Fallback: single-channel firmware, duplicate to all channels
                val = float(parts[0])
                return [val, val, val, val]
    except (ValueError, UnicodeDecodeError):
        pass
    return None


# ──────────────────────────────────────────────
# SIMULATED DATA GENERATION
# ──────────────────────────────────────────────

def generate_simulated_4ch(label, n_samples, fs=SAMPLING_RATE):
    """
    Generate simulated 4-channel EEG data for a given class.

    Parameters
    ----------
    label : int
        0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=BLINK, 5=WINK_LEFT, 6=WINK_RIGHT
    n_samples : int
        Number of samples to generate.
    fs : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 4) — columns are [C3, C4, Fp1, Fp2].
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

    # Base oscillations
    alpha_10 = np.sin(2 * np.pi * 10 * t)
    beta_20  = np.sin(2 * np.pi * 20 * t)
    noise    = np.random.randn(n_samples) * 0.5

    # Baseline for all channels
    c3  = 15 * alpha_10 + 8 * beta_20 + noise * 3
    c4  = 15 * alpha_10 + 8 * beta_20 + np.random.randn(n_samples) * 3
    fp1 = np.random.randn(n_samples) * 5  # Forehead: mostly noise baseline
    fp2 = np.random.randn(n_samples) * 5

    if label == 0:  # LEFT imagery → ERD at C4 (contralateral)
        c4 *= 0.4   # Strong suppression at C4
        c3 *= 0.9   # Mild change at C3
    elif label == 1:  # RIGHT imagery → ERD at C3 (contralateral)
        c3 *= 0.4   # Strong suppression at C3
        c4 *= 0.9
    elif label == 2:  # UP (tongue) → bilateral pattern change, more frontal beta
        c3 *= 0.7
        c4 *= 0.7
        c3 += 10 * np.sin(2 * np.pi * 25 * t)  # Extra high-beta
        c4 += 10 * np.sin(2 * np.pi * 25 * t)
    elif label == 3:  # DOWN (feet) → different bilateral pattern, lower frequency
        c3 *= 0.85
        c4 *= 0.85
        c3 += 8 * np.sin(2 * np.pi * 7 * t)   # Extra theta
        c4 += 8 * np.sin(2 * np.pi * 7 * t)
    elif label == 4:  # BLINK → large symmetric deflection on Fp1 & Fp2
        blink_times = np.random.choice(range(100, n_samples - 100), size=3, replace=False)
        for bt in blink_times:
            blink_artifact = 200 * np.exp(-0.5 * ((np.arange(n_samples) - bt) / 30) ** 2)
            fp1 += blink_artifact
            fp2 += blink_artifact
    elif label == 5:  # WINK_LEFT → larger deflection on Fp1 than Fp2
        wink_times = np.random.choice(range(100, n_samples - 100), size=3, replace=False)
        for wt in wink_times:
            artifact = 180 * np.exp(-0.5 * ((np.arange(n_samples) - wt) / 25) ** 2)
            fp1 += artifact * 1.0
            fp2 += artifact * 0.3
    elif label == 6:  # WINK_RIGHT → larger deflection on Fp2 than Fp1
        wink_times = np.random.choice(range(100, n_samples - 100), size=3, replace=False)
        for wt in wink_times:
            artifact = 180 * np.exp(-0.5 * ((np.arange(n_samples) - wt) / 25) ** 2)
            fp1 += artifact * 0.3
            fp2 += artifact * 1.0

    data = np.column_stack([c3, c4, fp1, fp2])
    return data.astype(np.float32)


# ──────────────────────────────────────────────
# EXPERIMENT RUNNER
# ──────────────────────────────────────────────

def run_experiment(subject_id, session_id=1, port=SERIAL_PORT,
                   use_hardware=True, block="all",
                   mi_trials=MI_TRIALS_PER_CLASS,
                   eog_trials=EOG_TRIALS_PER_CLASS):
    """
    Run the 4-channel data collection experiment.

    Parameters
    ----------
    subject_id : int
        Subject number (1-indexed).
    session_id : int
        Session number.
    port : str
        Serial port for Arduino.
    use_hardware : bool
        If False, use simulated data.
    block : str
        Which block to run: 'mi' (motor imagery only),
        'eog' (eye actions only), or 'all' (both).
    mi_trials : int
        Trials per MI class (default 25 → 100 total for 4 classes).
    eog_trials : int
        Trials per EOG class (default 20 → 60 total for 3 classes).
    """
    if pygame is None:
        logger.error("Pygame not installed. Run: pip install pygame")
        return

    logger.info(f"Starting 4-Channel Experiment: Subject {subject_id:03d}, Session {session_id}")
    logger.info(f"  Block: {block} | MI trials/class: {mi_trials} | EOG trials/class: {eog_trials}")

    screen, fonts = init_display()
    clock = pygame.time.Clock()

    # ── Serial connection ──
    ser = None
    if use_hardware:
        if serial is None:
            logger.error("pyserial not installed. Run: pip install pyserial")
            return
        try:
            ser = connect_4ch(port)
        except Exception as e:
            logger.error(f"Serial failed: {e}. Falling back to simulated.")
            use_hardware = False

    # ── Build trial list ──
    trials = []

    if block in ("mi", "all"):
        for class_id in range(4):
            for _ in range(mi_trials):
                trials.append(class_id)

    if block in ("eog", "all"):
        for class_id in range(4, 7):
            for _ in range(eog_trials):
                trials.append(class_id)

    np.random.shuffle(trials)
    total_trials = len(trials)

    logger.info(f"  Total trials: {total_trials}")

    # Storage
    all_data = []      # List of np.ndarray, each shape (n_samples, 4)
    all_labels = []    # List of int
    all_metadata = []  # List of dict per trial

    # Signal monitor buffer
    signal_buffer = []

    # ─────────── WELCOME SCREEN ───────────

    screen.fill(COLOR_BG)
    draw_text(screen, fonts["title"], "🧠 BCI 4-Channel Data Collection", COLOR_ACCENT,
              SCREEN_WIDTH // 2, 80)
    draw_text(screen, fonts["status"], f"Subject {subject_id:03d}  ·  {total_trials} trials  ·  Session {session_id}",
              COLOR_TEXT, SCREEN_WIDTH // 2, 140)

    y_off = 200
    if block in ("mi", "all"):
        draw_text(screen, fonts["status"], "── Motor Imagery (C3/C4) ──", COLOR_PROGRESS,
                  SCREEN_WIDTH // 2, y_off)
        y_off += 40
        for cid, info in MI_CLASSES.items():
            color = DIRECTION_COLORS[cid]
            draw_text(screen, fonts["info"],
                      f"  {info['arrow']}  {info['name']:6s} — {info['instruction']}",
                      color, SCREEN_WIDTH // 2, y_off)
            y_off += 30

    if block in ("eog", "all"):
        y_off += 15
        draw_text(screen, fonts["status"], "── Eye Actions (Fp1/Fp2) ──", COLOR_PROGRESS,
                  SCREEN_WIDTH // 2, y_off)
        y_off += 40
        for cid, info in EOG_CLASSES.items():
            color = DIRECTION_COLORS[cid]
            draw_text(screen, fonts["info"],
                      f"  {info['icon']}  {info['name']:12s} — {info['instruction']}",
                      color, SCREEN_WIDTH // 2, y_off)
            y_off += 30

    y_off += 40
    hw_text = "HARDWARE" if use_hardware else "SIMULATED"
    draw_text(screen, fonts["info"], f"Mode: {hw_text}  |  Port: {port}",
              COLOR_REST, SCREEN_WIDTH // 2, y_off)
    y_off += 50
    draw_text(screen, fonts["status"], "Press SPACE to begin  ·  ESC to abort",
              COLOR_READY, SCREEN_WIDTH // 2, y_off)

    pygame.display.flip()

    # Wait for spacebar
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                if ser:
                    ser.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    # ─────────── TRIAL LOOP ───────────

    for trial_n, class_id in enumerate(trials):
        class_info = ALL_CLASSES[class_id]
        class_name = class_info["name"]
        is_mi = class_id < 4
        color = DIRECTION_COLORS[class_id]

        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                logger.info("Experiment aborted by user.")
                _save_data(subject_id, session_id, all_data, all_labels,
                           all_metadata, block)
                pygame.quit()
                if ser:
                    ser.close()
                return

        # ── READY phase ──
        screen.fill(COLOR_BG)
        draw_text(screen, fonts["cue"], "+", COLOR_READY,
                  SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40)
        draw_text(screen, fonts["status"], f"Trial {trial_n + 1} / {total_trials}",
                  COLOR_REST, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        block_name = "Motor Imagery" if is_mi else "Eye Action"
        draw_text(screen, fonts["info"], f"Next: {block_name} — {class_name}",
                  COLOR_REST, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100)
        draw_progress_bar(screen, trial_n, total_trials)
        draw_signal_monitor(screen, signal_buffer)
        pygame.display.flip()
        time.sleep(READY_DURATION)

        # ── CUE phase ──
        screen.fill(COLOR_BG)

        if is_mi:
            # Draw large arrow
            draw_arrow(screen, class_id,
                       SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30,
                       size=100, color=color)
            draw_text(screen, fonts["action"], class_info["instruction"],
                      color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 110)
        else:
            # Draw eye icon text
            draw_text(screen, fonts["cue"], class_info["icon"],
                      color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)
            draw_text(screen, fonts["action"], class_info["instruction"],
                      color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)

        draw_progress_bar(screen, trial_n, total_trials)
        pygame.display.flip()
        time.sleep(CUE_DURATION)

        # ── IMAGINE / PERFORM phase (recording data) ──
        screen.fill(COLOR_BG)

        if is_mi:
            draw_arrow(screen, class_id,
                       SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                       size=100, color=color)
            draw_text(screen, fonts["action"], "IMAGINE NOW",
                      COLOR_IMAGINE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 90)
        else:
            draw_text(screen, fonts["cue"], class_info["icon"],
                      color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
            draw_text(screen, fonts["action"], "PERFORM NOW",
                      COLOR_IMAGINE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70)

        # Countdown timer
        draw_text(screen, fonts["info"], f"{IMAGINE_DURATION:.0f}s",
                  COLOR_REST, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 140)
        draw_progress_bar(screen, trial_n, total_trials)
        pygame.display.flip()

        # ── Collect samples ──
        epoch_samples = []
        start_time = time.time()

        if use_hardware and ser:
            while time.time() - start_time < IMAGINE_DURATION:
                sample = read_4ch_sample(ser)
                if sample is not None:
                    epoch_samples.append(sample)
                    signal_buffer.append(sample)
                    # Keep buffer manageable
                    if len(signal_buffer) > 500:
                        signal_buffer = signal_buffer[-500:]

                # Update countdown every 0.5s
                elapsed = time.time() - start_time
                remaining = IMAGINE_DURATION - elapsed
                if int(elapsed * 2) != int((elapsed - 0.01) * 2):
                    # Redraw countdown
                    screen.fill(COLOR_BG)
                    if is_mi:
                        draw_arrow(screen, class_id,
                                   SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                                   size=100, color=color)
                        draw_text(screen, fonts["action"], "IMAGINE NOW",
                                  COLOR_IMAGINE, SCREEN_WIDTH // 2,
                                  SCREEN_HEIGHT // 2 + 90)
                    else:
                        draw_text(screen, fonts["cue"], class_info["icon"],
                                  color, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
                        draw_text(screen, fonts["action"], "PERFORM NOW",
                                  COLOR_IMAGINE, SCREEN_WIDTH // 2,
                                  SCREEN_HEIGHT // 2 + 70)
                    draw_text(screen, fonts["status"], f"{remaining:.1f}s",
                              COLOR_READY, SCREEN_WIDTH // 2,
                              SCREEN_HEIGHT // 2 + 140)
                    draw_text(screen, fonts["small"],
                              f"Samples: {len(epoch_samples)}",
                              COLOR_REST, SCREEN_WIDTH // 2,
                              SCREEN_HEIGHT // 2 + 170)
                    draw_progress_bar(screen, trial_n, total_trials)
                    draw_signal_monitor(screen, signal_buffer)
                    pygame.display.flip()

                # Handle events during recording
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        logger.info("Aborted during recording.")
                        _save_data(subject_id, session_id, all_data, all_labels,
                                   all_metadata, block)
                        pygame.quit()
                        ser.close()
                        return
        else:
            # Simulated mode
            n_sim = int(IMAGINE_DURATION * SAMPLING_RATE)
            sim_data = generate_simulated_4ch(class_id, n_sim)
            epoch_samples = sim_data.tolist()

            # Add to signal buffer for display
            for s in sim_data[-200:]:
                signal_buffer.append(s.tolist())
            if len(signal_buffer) > 500:
                signal_buffer = signal_buffer[-500:]

            time.sleep(IMAGINE_DURATION)

        # Convert to array
        epoch_array = np.array(epoch_samples, dtype=np.float32)

        all_data.append(epoch_array)
        all_labels.append(class_id)
        all_metadata.append({
            "trial": trial_n + 1,
            "class_id": class_id,
            "class_name": class_name,
            "n_samples": len(epoch_samples),
            "duration_s": round(time.time() - start_time, 2),
            "is_motor_imagery": is_mi,
        })

        logger.info(f"  Trial {trial_n + 1:3d}/{total_trials} | "
                     f"{class_name:12s} | {len(epoch_samples):4d} samples | "
                     f"{epoch_array.shape}")

        # ── REST phase ──
        screen.fill(COLOR_BG)
        draw_text(screen, fonts["status"], "Rest...", COLOR_REST,
                  SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)
        draw_text(screen, fonts["info"],
                  f"Completed: {trial_n + 1}/{total_trials}",
                  COLOR_REST, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)
        draw_progress_bar(screen, trial_n + 1, total_trials)
        draw_signal_monitor(screen, signal_buffer)
        pygame.display.flip()
        time.sleep(REST_DURATION)

    # ─────────── SAVE DATA ───────────

    save_path = _save_data(subject_id, session_id, all_data, all_labels,
                           all_metadata, block)

    # ─────────── COMPLETION SCREEN ───────────

    screen.fill(COLOR_BG)
    draw_text(screen, fonts["title"], "✓ Data Collection Complete!",
              COLOR_IMAGINE, SCREEN_WIDTH // 2, 200)
    draw_text(screen, fonts["status"],
              f"{len(all_labels)} trials collected  ·  Subject {subject_id:03d}",
              COLOR_TEXT, SCREEN_WIDTH // 2, 270)

    # Summary per class
    y_off = 330
    for cid in sorted(set(all_labels)):
        count = all_labels.count(cid)
        info = ALL_CLASSES[cid]
        color = DIRECTION_COLORS[cid]
        draw_text(screen, fonts["info"],
                  f"{info['name']:12s}: {count} trials",
                  color, SCREEN_WIDTH // 2, y_off)
        y_off += 28

    draw_text(screen, fonts["info"], f"Saved: {save_path}",
              COLOR_REST, SCREEN_WIDTH // 2, y_off + 30)
    draw_text(screen, fonts["status"], "Press any key to exit",
              COLOR_READY, SCREEN_WIDTH // 2, y_off + 80)
    pygame.display.flip()

    # Wait for keypress
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False

    pygame.quit()
    if ser:
        ser.close()

    logger.info(f"Experiment complete: {len(all_labels)} trials saved to {save_path}")


def _save_data(subject_id, session_id, all_data, all_labels, all_metadata, block):
    """Save collected data to disk."""
    if not all_data:
        logger.warning("No data to save.")
        return None

    save_dir = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save as npz with variable-length trials
    data_path = save_dir / f"session_{session_id:02d}_4ch_{block}.npz"

    np.savez(
        data_path,
        data=np.array(all_data, dtype=object),
        labels=np.array(all_labels, dtype=np.int32),
        channel_names=np.array(CHANNEL_NAMES),
    )
    logger.info(f"✓ Data saved: {data_path}")

    # Save metadata JSON
    meta = {
        "subject_id": subject_id,
        "session_id": session_id,
        "block": block,
        "n_channels": N_CHANNELS,
        "channel_names": CHANNEL_NAMES,
        "sampling_rate_hz": SAMPLING_RATE,
        "n_trials": len(all_labels),
        "class_counts": {},
        "class_map": {str(k): v["name"] for k, v in ALL_CLASSES.items()},
        "trial_timing": {
            "ready_s": READY_DURATION,
            "cue_s": CUE_DURATION,
            "imagine_s": IMAGINE_DURATION,
            "rest_s": REST_DURATION,
        },
        "electrode_setup": {
            "C3": "A0 — left motor cortex, ref: A1 earlobe",
            "C4": "A1 — right motor cortex, ref: A2 earlobe",
            "Fp1": "A2 — left forehead, ref: A1 earlobe",
            "Fp2": "A3 — right forehead, ref: A2 earlobe",
        },
        "trials": all_metadata,
    }

    for cid in sorted(set(all_labels)):
        count = all_labels.count(cid)
        name = ALL_CLASSES[cid]["name"]
        meta["class_counts"][name] = count

    meta_path = save_dir / f"session_{session_id:02d}_4ch_{block}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"✓ Metadata saved: {meta_path}")

    return data_path


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BCI 4-Channel Data Collection — Motor Imagery + EOG"
    )
    parser.add_argument("--subject", type=int, required=True,
                        help="Subject ID (1-indexed)")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number (default: 1)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data (no hardware needed)")
    parser.add_argument("--block", type=str, default="all",
                        choices=["mi", "eog", "all"],
                        help="Block to run: 'mi'=motor imagery, 'eog'=eye, 'all'=both")
    parser.add_argument("--mi-trials", type=int, default=MI_TRIALS_PER_CLASS,
                        help=f"Trials per MI class (default: {MI_TRIALS_PER_CLASS})")
    parser.add_argument("--eog-trials", type=int, default=EOG_TRIALS_PER_CLASS,
                        help=f"Trials per EOG class (default: {EOG_TRIALS_PER_CLASS})")
    args = parser.parse_args()

    run_experiment(
        subject_id=args.subject,
        session_id=args.session,
        port=args.port,
        use_hardware=not args.simulate,
        block=args.block,
        mi_trials=args.mi_trials,
        eog_trials=args.eog_trials,
    )


if __name__ == "__main__":
    main()
