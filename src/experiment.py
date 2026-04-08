"""
BCI Assistive Control — Motor Imagery Experiment Paradigm
=========================================================
Runs the data collection paradigm using Pygame. Displays visual
cues (LEFT/RIGHT arrows) and records EEG epochs from the
Upside Down Labs kit during motor imagery periods.

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
    TRIALS_PER_SUBJECT, RELAX_DURATION, CUE_DURATION,
    IMAGERY_DURATION, REST_DURATION,
    subject_dir, save_session_metadata, setup_logger
)

logger = setup_logger("experiment")

# ──────────────────────────────────────────────
# DISPLAY CONFIGURATION
# ──────────────────────────────────────────────

SCREEN_WIDTH  = 1024
SCREEN_HEIGHT = 700

# Colors (dark theme)
COLOR_BG      = (18, 18, 28)
COLOR_TEXT    = (200, 200, 210)
COLOR_CUE_L  = (0, 210, 255)    # Cyan for LEFT
COLOR_CUE_R  = (255, 166, 0)    # Orange for RIGHT
COLOR_IMAGINE = (100, 255, 100)  # Green for IMAGINE
COLOR_READY   = (180, 180, 190)
COLOR_REST    = (60, 60, 80)
COLOR_ACCENT  = (233, 69, 96)   # Red accent


def init_display():
    """Initialize Pygame display and fonts."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🧠 BCI Motor Imagery — Data Collection")

    fonts = {
        "title":  pygame.font.SysFont("Segoe UI", 42, bold=True),
        "cue":    pygame.font.SysFont("Segoe UI", 96, bold=True),
        "status": pygame.font.SysFont("Segoe UI", 28),
        "info":   pygame.font.SysFont("Consolas", 18),
    }
    return screen, fonts


def draw_centered_text(screen, font, text, color, y_offset=0):
    """Draw text centered horizontally on screen."""
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset))
    screen.blit(surface, rect)


def draw_progress_bar(screen, current, total, y=SCREEN_HEIGHT - 50):
    """Draw a progress bar at the bottom of the screen."""
    bar_width = SCREEN_WIDTH - 100
    bar_height = 12
    x = 50
    progress = current / max(total, 1)

    # Background
    pygame.draw.rect(screen, (40, 40, 60), (x, y, bar_width, bar_height), border_radius=6)
    # Fill
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        pygame.draw.rect(screen, COLOR_ACCENT, (x, y, fill_width, bar_height), border_radius=6)


# ──────────────────────────────────────────────
# EXPERIMENT RUNNER
# ──────────────────────────────────────────────

def run_experiment(subject_id, n_trials=TRIALS_PER_SUBJECT, session_id=1,
                   port=SERIAL_PORT, use_hardware=True):
    """
    Run the motor imagery data collection experiment.

    Parameters
    ----------
    subject_id : int
        Subject number (1-indexed).
    n_trials : int
        Total number of trials (must be even for balanced classes).
    session_id : int
        Session number for this subject.
    port : str
        Serial port for EEG data.
    use_hardware : bool
        If False, uses simulated data instead of serial.
    """
    if pygame is None:
        logger.error("Pygame is not installed. Run: pip install pygame")
        return

    logger.info(f"Starting experiment: Subject {subject_id:03d}, "
                f"{n_trials} trials, Session {session_id}")

    # Initialize display
    screen, fonts = init_display()
    clock = pygame.time.Clock()

    # Initialize serial connection
    ser = None
    if use_hardware:
        if serial is None:
            logger.error("pyserial is not installed. Run: pip install pyserial")
            return
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            # Skip header lines
            for _ in range(10):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line and not line.startswith("#"):
                    break
            logger.info(f"✓ Connected to {port}")
        except Exception as e:
            logger.error(f"Could not connect to serial port: {e}")
            logger.info("Falling back to simulated data.")
            use_hardware = False

    # Generate balanced, randomized trial order
    labels = []
    for i in range(n_trials):
        labels.append(i % 2)  # 0 = LEFT, 1 = RIGHT
    np.random.shuffle(labels)

    # Storage
    all_trials = []
    all_labels = []

    # ─────────── WELCOME SCREEN ───────────

    screen.fill(COLOR_BG)
    draw_centered_text(screen, fonts["title"], "🧠 Motor Imagery BCI", COLOR_ACCENT, -80)
    draw_centered_text(screen, fonts["status"], f"Subject {subject_id:03d}  ·  {n_trials} trials", COLOR_TEXT, -20)
    draw_centered_text(screen, fonts["status"], "When you see LEFT ← : imagine squeezing your LEFT fist", COLOR_CUE_L, 40)
    draw_centered_text(screen, fonts["status"], "When you see RIGHT → : imagine squeezing your RIGHT fist", COLOR_CUE_R, 80)
    draw_centered_text(screen, fonts["info"], "Press SPACE to begin  ·  Press ESC to abort", COLOR_REST, 150)
    pygame.display.flip()

    # Wait for spacebar
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                if ser:
                    ser.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    # ─────────── TRIAL LOOP ───────────

    for trial_n, label in enumerate(labels):
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                logger.info("Experiment aborted by user.")
                pygame.quit()
                if ser:
                    ser.close()
                return

        # ── RELAX phase ──
        screen.fill(COLOR_BG)
        draw_centered_text(screen, fonts["cue"], "Ready", COLOR_READY, -30)
        draw_centered_text(screen, fonts["info"],
                          f"Trial {trial_n + 1} / {n_trials}", COLOR_REST, 60)
        draw_progress_bar(screen, trial_n, n_trials)
        pygame.display.flip()
        time.sleep(RELAX_DURATION)

        # ── CUE phase ──
        cue_text = "← LEFT" if label == 0 else "RIGHT →"
        cue_color = COLOR_CUE_L if label == 0 else COLOR_CUE_R

        screen.fill(COLOR_BG)
        draw_centered_text(screen, fonts["cue"], cue_text, cue_color, -30)
        draw_progress_bar(screen, trial_n, n_trials)
        pygame.display.flip()
        time.sleep(CUE_DURATION)

        # ── MOTOR IMAGERY phase (record EEG) ──
        screen.fill(COLOR_BG)
        draw_centered_text(screen, fonts["cue"], "IMAGINE", COLOR_IMAGINE, -30)
        side_text = "Left Hand" if label == 0 else "Right Hand"
        draw_centered_text(screen, fonts["status"], side_text, cue_color, 50)
        draw_progress_bar(screen, trial_n, n_trials)
        pygame.display.flip()

        # Collect EEG samples during imagery period
        epoch_samples = []
        start = time.time()

        if use_hardware and ser:
            while time.time() - start < IMAGERY_DURATION:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and not line.startswith("#"):
                        epoch_samples.append(float(line))
                except (ValueError, UnicodeDecodeError):
                    pass
        else:
            # Simulated: generate ~1000 samples (4s × 250 Hz)
            n_sim = int(IMAGERY_DURATION * SAMPLING_RATE)
            from src.acquisition import generate_simulated_epoch
            epoch_samples = generate_simulated_epoch(
                label=label, n_samples=n_sim, fs=SAMPLING_RATE
            ).tolist()
            time.sleep(IMAGERY_DURATION)  # Maintain timing

        all_trials.append(epoch_samples)
        all_labels.append(label)

        side = "LEFT" if label == 0 else "RIGHT"
        logger.info(f"  Trial {trial_n + 1:3d}/{n_trials} | {side:5s} | "
                    f"{len(epoch_samples)} samples")

        # ── REST phase ──
        screen.fill(COLOR_BG)
        draw_centered_text(screen, fonts["info"], "Rest...", COLOR_REST, 0)
        draw_progress_bar(screen, trial_n + 1, n_trials)
        pygame.display.flip()
        time.sleep(REST_DURATION)

    # ─────────── SAVE DATA ───────────

    save_dir = subject_dir(subject_id)
    data_path = save_dir / f"session_{session_id:02d}.npz"

    np.savez(
        data_path,
        data=np.array(all_trials, dtype=object),
        labels=np.array(all_labels, dtype=np.int32)
    )
    logger.info(f"✓ Data saved: {data_path}")

    # Save metadata
    meta_path = save_session_metadata(
        subject_id, session_id,
        n_trials_collected=len(all_trials),
        trial_order=[("LEFT" if l == 0 else "RIGHT") for l in all_labels],
        notes="Collected via experiment.py"
    )
    logger.info(f"✓ Metadata saved: {meta_path}")

    # ─────────── COMPLETION SCREEN ───────────

    screen.fill(COLOR_BG)
    draw_centered_text(screen, fonts["title"], "✓ Complete!", COLOR_IMAGINE, -40)
    draw_centered_text(screen, fonts["status"],
                      f"{len(all_trials)} trials collected  ·  Subject {subject_id:03d}",
                      COLOR_TEXT, 20)
    draw_centered_text(screen, fonts["info"], "Press any key to exit", COLOR_REST, 80)
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

    logger.info(f"Experiment complete: Subject {subject_id:03d}, "
                f"{len(all_trials)} trials saved.")


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BCI Motor Imagery Data Collection")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID (1-indexed)")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_SUBJECT,
                        help=f"Number of trials (default: {TRIALS_PER_SUBJECT})")
    parser.add_argument("--session", type=int, default=1, help="Session number (default: 1)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data (no hardware needed)")
    args = parser.parse_args()

    run_experiment(
        subject_id=args.subject,
        n_trials=args.trials,
        session_id=args.session,
        port=args.port,
        use_hardware=not args.simulate
    )


if __name__ == "__main__":
    main()
