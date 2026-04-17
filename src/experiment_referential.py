"""
BCI Assistive Control — Data Collection (1-Channel Referential + Webcam)
=========================================================================
Collects motor imagery data from a SINGLE referential EEG channel (C3)
and eye actions via webcam — matched to actual hardware available.

Hardware Setup:
  BioAmp EXG Pill → Arduino A0 (single channel)
    IN+  → C3 electrode (scalp, left motor cortex)
    IN-  → A1 earlobe (reference)
    GND  → A2 earlobe (ground)
  
  Webcam → Eye tracking via MediaPipe
    BLINK      → both eyes close
    WINK_LEFT  → left eye close
    WINK_RIGHT → right eye close

Motor Imagery Classes (all from C3 referential):
  LEFT  — imagine squeezing LEFT fist  → cursor LEFT
  RIGHT — imagine squeezing RIGHT fist → cursor RIGHT
  UP    — imagine curling tongue up     → cursor UP
  DOWN  — imagine wiggling toes         → cursor DOWN

Why this works better than bipolar:
  - Bipolar (C3-C4) cancels common signals, loses lateralization info
  - Referential (C3 vs earlobe) preserves the actual cortical signal
  - LEFT imagery → strong ERD at C3 (ipsilateral, less suppression)
  - RIGHT imagery → weak ERD at C3 (contralateral, more suppression)
  - UP/DOWN → different frequency patterns (beta vs theta at C3)

Usage:
  python -m src.experiment_referential --subject 1 --port COM7
  python -m src.experiment_referential --subject 1 --simulate
  python -m src.experiment_referential --subject 1 --port COM7 --block mi
  python -m src.experiment_referential --subject 1 --simulate --trials 15

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
    RAW_DATA_DIR, setup_logger
)

logger = setup_logger("experiment_ref")

# ──────────────────────────────────────────────
# EXPERIMENT PARAMETERS
# ──────────────────────────────────────────────

# Motor imagery classes
MI_CLASSES = {
    0: {"name": "LEFT",  "task": "Squeeze LEFT Fist",    "arrow": "←",
        "tip": "Imagine clenching your left hand tightly"},
    1: {"name": "RIGHT", "task": "Squeeze RIGHT Fist",   "arrow": "→",
        "tip": "Imagine clenching your right hand tightly"},
    2: {"name": "UP",    "task": "Curl Tongue Upward",    "arrow": "↑",
        "tip": "Imagine pressing tongue firmly to roof of mouth"},
    3: {"name": "DOWN",  "task": "Wiggle Your Toes",      "arrow": "↓",
        "tip": "Imagine curling and wiggling all your toes"},
}

# Eye action classes (collected via webcam OR Fp1/Fp2 if available)
EOG_CLASSES = {
    4: {"name": "BLINK",      "task": "Close BOTH Eyes (1 sec)", "icon": "◉◉"},
    5: {"name": "WINK_LEFT",  "task": "Close LEFT Eye Only",     "icon": "◉ ○"},
    6: {"name": "WINK_RIGHT", "task": "Close RIGHT Eye Only",    "icon": "○ ◉"},
}

ALL_CLASSES = {**MI_CLASSES, **EOG_CLASSES}

# Trial timing (seconds)
READY_DURATION   = 2.0
CUE_DURATION     = 1.0
IMAGINE_DURATION = 4.0
REST_DURATION    = 2.0

# Default trials per class
DEFAULT_MI_TRIALS  = 25   # 25 × 4 = 100 MI trials
DEFAULT_EOG_TRIALS = 20   # 20 × 3 = 60 EOG trials

# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

SCREEN_W = 1200
SCREEN_H = 800

# Dark theme colors
BG       = (15, 15, 25)
TEXT     = (210, 210, 220)
REST_C   = (60, 60, 80)
READY_C  = (180, 180, 190)
ACCENT   = (233, 69, 96)
IMAGINE_C = (100, 255, 100)
PROGRESS_C = (0, 180, 255)

CLASS_COLORS = {
    0: (0, 200, 255),    # LEFT — cyan
    1: (255, 166, 0),    # RIGHT — orange
    2: (100, 255, 150),  # UP — green
    3: (255, 100, 100),  # DOWN — red
    4: (255, 255, 100),  # BLINK — yellow
    5: (200, 150, 255),  # WINK_L — purple
    6: (200, 150, 255),  # WINK_R — purple
}


def init_display():
    """Initialize Pygame."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("🧠 BCI Data Collection — Referential + Webcam")
    fonts = {
        "title":  pygame.font.SysFont("Segoe UI", 44, bold=True),
        "cue":    pygame.font.SysFont("Segoe UI", 120, bold=True),
        "action": pygame.font.SysFont("Segoe UI", 36),
        "status": pygame.font.SysFont("Segoe UI", 26),
        "info":   pygame.font.SysFont("Consolas", 18),
        "tip":    pygame.font.SysFont("Segoe UI", 22, italic=True),
        "small":  pygame.font.SysFont("Consolas", 14),
    }
    return screen, fonts


def txt(screen, font, text, color, x, y):
    """Draw centered text."""
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(x, y))
    screen.blit(surf, rect)


def draw_arrow(screen, direction, cx, cy, size=100, color=(255, 255, 255)):
    """Draw a directional arrow."""
    s = size
    if direction == 0:    # LEFT
        pts = [(cx+s, cy-s//2), (cx-s, cy), (cx+s, cy+s//2)]
    elif direction == 1:  # RIGHT
        pts = [(cx-s, cy-s//2), (cx+s, cy), (cx-s, cy+s//2)]
    elif direction == 2:  # UP
        pts = [(cx-s//2, cy+s), (cx, cy-s), (cx+s//2, cy+s)]
    elif direction == 3:  # DOWN
        pts = [(cx-s//2, cy-s), (cx, cy+s), (cx+s//2, cy-s)]
    else:
        return
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (255, 255, 255), pts, 3)


def draw_progress(screen, current, total):
    """Draw progress bar at bottom."""
    y = SCREEN_H - 40
    w = SCREEN_W - 100
    h = 14
    x = 50
    frac = current / max(total, 1)

    pygame.draw.rect(screen, (40, 40, 60), (x, y, w, h), border_radius=7)
    fill = int(w * frac)
    if fill > 0:
        pygame.draw.rect(screen, PROGRESS_C, (x, y, fill, h), border_radius=7)

    font = pygame.font.SysFont("Consolas", 14)
    txt(screen, font, f"{frac*100:.0f}%", TEXT, SCREEN_W//2, y - 12)


def draw_signal_trace(screen, buffer, y_start=700):
    """Draw small real-time EEG trace."""
    if len(buffer) < 10:
        return
    font = pygame.font.SysFont("Consolas", 12)
    txt(screen, font, "C3 (ref)", (0, 200, 255), 45, y_start)

    data = np.array(buffer[-300:])
    if np.ptp(data) > 0:
        norm = (data - np.mean(data)) / (np.ptp(data) + 1e-10) * 20
    else:
        norm = np.zeros_like(data)

    plot_w = SCREEN_W - 200
    points = []
    for i, v in enumerate(norm):
        px = 90 + int(i * plot_w / len(norm))
        py = int(y_start + v)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(screen, (0, 200, 255), False, points, 1)


# ──────────────────────────────────────────────
# SERIAL (1-CHANNEL)
# ──────────────────────────────────────────────

def connect_serial(port=SERIAL_PORT, baud=BAUD_RATE):
    """Open serial connection (same firmware as before)."""
    logger.info(f"Connecting to {port} @ {baud}...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)

    for _ in range(20):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and line.startswith("#"):
            logger.info(f"  Firmware: {line}")
        elif line:
            break

    logger.info(f"✓ Connected to {port}")
    return ser


def read_sample(ser):
    """Read one ADC value from serial."""
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            return float(line)
    except (ValueError, UnicodeDecodeError):
        pass
    return None


# ──────────────────────────────────────────────
# SIMULATED DATA
# ──────────────────────────────────────────────

def simulate_referential_epoch(label, n_samples, fs=SAMPLING_RATE):
    """
    Simulate a single-channel referential C3 epoch.

    Key: In referential montage, C3 electrode picks up:
      LEFT  imagery → ipsilateral to C3 → LESS suppression (higher mu power)
      RIGHT imagery → contralateral   → MORE suppression (lower mu power, ERD)
      UP    imagery → tongue → bilateral high-beta increase
      DOWN  imagery → feet  → theta/low-alpha increase
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

    # Base oscillations present at C3
    mu    = 20 * np.sin(2 * np.pi * 10 * t)   # Mu rhythm (10 Hz)
    beta  = 10 * np.sin(2 * np.pi * 20 * t)   # Beta (20 Hz)
    theta = 5  * np.sin(2 * np.pi * 6 * t)    # Theta (6 Hz)

    # Background noise
    pink = np.cumsum(np.random.randn(n_samples)) * 0.2
    pink -= np.mean(pink)
    white = np.random.randn(n_samples) * 2

    if label == 0:   # LEFT imagery → C3 ipsilateral → LESS ERD (mu stays)
        signal = mu * 0.85 + beta * 0.8 + theta * 0.3
    elif label == 1: # RIGHT imagery → C3 contralateral → STRONG ERD (mu drops)
        signal = mu * 0.35 + beta * 0.4 + theta * 0.3
    elif label == 2: # UP (tongue) → high beta burst at C3
        signal = mu * 0.6 + beta * 0.5 + theta * 0.3
        signal += 12 * np.sin(2 * np.pi * 26 * t)  # High beta burst
    elif label == 3: # DOWN (feet) → theta increase, mu/beta mild change
        signal = mu * 0.7 + beta * 0.7 + theta * 1.5
        signal += 8 * np.sin(2 * np.pi * 4 * t)   # Strong theta
    else:
        signal = mu + beta + theta  # Baseline (rest / EOG)

    signal = signal + pink + white

    # Add subject variability
    signal += np.random.randn() * 2   # DC offset
    signal *= (0.8 + 0.4 * np.random.rand())  # Amplitude

    return signal.astype(np.float32)


# ──────────────────────────────────────────────
# EXPERIMENT
# ──────────────────────────────────────────────

def run_experiment(subject_id, session_id=1, port=SERIAL_PORT,
                   use_hardware=True, block="all",
                   mi_trials=DEFAULT_MI_TRIALS, eog_trials=DEFAULT_EOG_TRIALS):
    """
    Run data collection experiment.

    Parameters
    ----------
    block : str
        'mi' = motor imagery only (EEG)
        'eog' = eye actions only (webcam timing cues)
        'all' = both blocks
    """
    if pygame is None:
        logger.error("Pygame not installed. Run: pip install pygame")
        return

    logger.info(f"Starting experiment: Subject {subject_id:03d}, Session {session_id}")

    screen, fonts = init_display()

    # Serial connection (for MI block)
    ser = None
    if use_hardware and block in ("mi", "all"):
        if serial is None:
            logger.error("pyserial not installed.")
            return
        try:
            ser = connect_serial(port)
        except Exception as e:
            logger.error(f"Serial failed: {e}. Falling back to simulated.")
            use_hardware = False

    # Build trial list
    trials = []
    if block in ("mi", "all"):
        for cid in range(4):
            trials.extend([cid] * mi_trials)
    if block in ("eog", "all"):
        for cid in range(4, 7):
            trials.extend([cid] * eog_trials)

    np.random.shuffle(trials)
    total = len(trials)
    logger.info(f"  Total trials: {total}")

    # Storage
    all_eeg = []        # List of 1D np arrays (EEG from C3)
    all_labels = []     # Class IDs
    all_meta = []       # Per-trial metadata
    signal_buf = []     # For live trace display

    # ─────────── WELCOME SCREEN ───────────

    screen.fill(BG)
    txt(screen, fonts["title"], "🧠 BCI Data Collection", ACCENT, SCREEN_W//2, 60)
    txt(screen, fonts["status"],
        f"Subject {subject_id:03d}  ·  {total} trials  ·  Session {session_id}",
        TEXT, SCREEN_W//2, 110)

    y = 170
    txt(screen, fonts["status"], "── Electrode Setup ──", PROGRESS_C, SCREEN_W//2, y)
    y += 35
    txt(screen, fonts["info"], "BioAmp EXG Pill:  IN+→C3 (scalp)  IN-→Earlobe (ref)  GND→Other earlobe",
        (0, 200, 255), SCREEN_W//2, y)
    y += 25
    mode = "HARDWARE" if use_hardware else "SIMULATED"
    txt(screen, fonts["info"], f"Mode: {mode}   Webcam: blink/wink detection",
        REST_C, SCREEN_W//2, y)

    y += 50
    if block in ("mi", "all"):
        txt(screen, fonts["status"], "── Motor Imagery (EEG from C3) ──", PROGRESS_C, SCREEN_W//2, y)
        y += 35
        for cid, info in MI_CLASSES.items():
            color = CLASS_COLORS[cid]
            txt(screen, fonts["info"],
                f"  {info['arrow']}  {info['name']:6s} — {info['task']}",
                color, SCREEN_W//2, y)
            y += 28

    if block in ("eog", "all"):
        y += 10
        txt(screen, fonts["status"], "── Eye Actions (Webcam) ──", PROGRESS_C, SCREEN_W//2, y)
        y += 35
        for cid, info in EOG_CLASSES.items():
            color = CLASS_COLORS[cid]
            txt(screen, fonts["info"],
                f"  {info['icon']}  {info['name']:12s} — {info['task']}",
                color, SCREEN_W//2, y)
            y += 28

    y += 40
    txt(screen, fonts["status"], "Press SPACE to start  ·  ESC to abort",
        READY_C, SCREEN_W//2, y)
    pygame.display.flip()

    # Wait for space
    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or \
               (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                if ser: ser.close()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                waiting = False

    # ─────────── TRIAL LOOP ───────────

    for trial_n, cid in enumerate(trials):
        info = ALL_CLASSES[cid]
        is_mi = cid < 4
        color = CLASS_COLORS[cid]

        # Check quit
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or \
               (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                _save(subject_id, session_id, all_eeg, all_labels, all_meta, block)
                pygame.quit()
                if ser: ser.close()
                return

        # ── READY ("+" fixation) ──
        screen.fill(BG)
        txt(screen, fonts["cue"], "+", READY_C, SCREEN_W//2, SCREEN_H//2 - 50)
        txt(screen, fonts["status"], f"Trial {trial_n+1} / {total}",
            REST_C, SCREEN_W//2, SCREEN_H//2 + 40)
        block_type = "Motor Imagery" if is_mi else "Eye Action"
        txt(screen, fonts["info"], f"Next: {block_type} — {info['name']}",
            REST_C, SCREEN_W//2, SCREEN_H//2 + 80)
        draw_progress(screen, trial_n, total)
        draw_signal_trace(screen, signal_buf)
        pygame.display.flip()
        time.sleep(READY_DURATION)

        # ── CUE (show what to do) ──
        screen.fill(BG)
        if is_mi:
            draw_arrow(screen, cid, SCREEN_W//2, SCREEN_H//2 - 40, size=100, color=color)
            txt(screen, fonts["action"], info["task"], color, SCREEN_W//2, SCREEN_H//2 + 100)
            txt(screen, fonts["tip"], info["tip"], (150, 150, 170), SCREEN_W//2, SCREEN_H//2 + 150)
        else:
            txt(screen, fonts["cue"], info["icon"], color, SCREEN_W//2, SCREEN_H//2 - 40)
            txt(screen, fonts["action"], info["task"], color, SCREEN_W//2, SCREEN_H//2 + 80)
        draw_progress(screen, trial_n, total)
        pygame.display.flip()
        time.sleep(CUE_DURATION)

        # ── IMAGINE / PERFORM (recording) ──
        epoch_data = []
        start_t = time.time()

        if is_mi:
            # Record EEG during motor imagery
            screen.fill(BG)
            draw_arrow(screen, cid, SCREEN_W//2, SCREEN_H//2 - 60, size=100, color=color)
            txt(screen, fonts["action"], "IMAGINE NOW", IMAGINE_C,
                SCREEN_W//2, SCREEN_H//2 + 80)
            txt(screen, fonts["tip"], info["tip"], (150, 150, 170),
                SCREEN_W//2, SCREEN_H//2 + 130)
            draw_progress(screen, trial_n, total)
            pygame.display.flip()

            if use_hardware and ser:
                while time.time() - start_t < IMAGINE_DURATION:
                    val = read_sample(ser)
                    if val is not None:
                        epoch_data.append(val)
                        signal_buf.append(val)
                        if len(signal_buf) > 600:
                            signal_buf = signal_buf[-600:]

                    # Update display periodically
                    elapsed = time.time() - start_t
                    remaining = IMAGINE_DURATION - elapsed
                    if int(elapsed * 4) != int((elapsed - 0.001) * 4):
                        screen.fill(BG)
                        draw_arrow(screen, cid, SCREEN_W//2, SCREEN_H//2 - 60,
                                   size=100, color=color)
                        txt(screen, fonts["action"], "IMAGINE NOW", IMAGINE_C,
                            SCREEN_W//2, SCREEN_H//2 + 80)
                        txt(screen, fonts["status"], f"{remaining:.1f}s",
                            READY_C, SCREEN_W//2, SCREEN_H//2 + 130)
                        txt(screen, fonts["small"], f"Samples: {len(epoch_data)}",
                            REST_C, SCREEN_W//2, SCREEN_H//2 + 160)
                        draw_progress(screen, trial_n, total)
                        draw_signal_trace(screen, signal_buf)
                        pygame.display.flip()

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT or \
                           (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                            _save(subject_id, session_id, all_eeg, all_labels, all_meta, block)
                            pygame.quit()
                            ser.close()
                            return
            else:
                # Simulated
                n_sim = int(IMAGINE_DURATION * SAMPLING_RATE)
                epoch_data = simulate_referential_epoch(cid, n_sim).tolist()
                signal_buf.extend(epoch_data[-300:])
                if len(signal_buf) > 600:
                    signal_buf = signal_buf[-600:]
                time.sleep(IMAGINE_DURATION)
        else:
            # EOG block — just show cue, user performs eye action
            # EEG is also recorded during this if hardware is connected
            screen.fill(BG)
            txt(screen, fonts["cue"], info["icon"], color, SCREEN_W//2, SCREEN_H//2 - 60)
            txt(screen, fonts["action"], "PERFORM NOW", IMAGINE_C,
                SCREEN_W//2, SCREEN_H//2 + 60)
            txt(screen, fonts["status"], info["task"], color,
                SCREEN_W//2, SCREEN_H//2 + 110)
            draw_progress(screen, trial_n, total)
            pygame.display.flip()

            if use_hardware and ser:
                while time.time() - start_t < IMAGINE_DURATION:
                    val = read_sample(ser)
                    if val is not None:
                        epoch_data.append(val)
                    for ev in pygame.event.get():
                        pass
            else:
                # Simulated EOG (large deflections in EEG from eye movement)
                n_sim = int(IMAGINE_DURATION * SAMPLING_RATE)
                base = np.random.randn(n_sim) * 3
                if cid == 4:   # BLINK — large symmetric artifact
                    for t_blink in [n_sim//4, n_sim//2, 3*n_sim//4]:
                        artifact = 150 * np.exp(-0.5*((np.arange(n_sim)-t_blink)/30)**2)
                        base += artifact
                elif cid == 5: # WINK_LEFT
                    for t_w in [n_sim//3, 2*n_sim//3]:
                        base += 100 * np.exp(-0.5*((np.arange(n_sim)-t_w)/25)**2)
                elif cid == 6: # WINK_RIGHT
                    for t_w in [n_sim//3, 2*n_sim//3]:
                        base += 80 * np.exp(-0.5*((np.arange(n_sim)-t_w)/20)**2)
                epoch_data = base.tolist()
                time.sleep(IMAGINE_DURATION)

        # Store
        epoch_arr = np.array(epoch_data, dtype=np.float32)
        all_eeg.append(epoch_arr)
        all_labels.append(cid)
        all_meta.append({
            "trial": trial_n + 1,
            "class_id": cid,
            "class_name": info["name"],
            "n_samples": len(epoch_data),
            "duration_s": round(time.time() - start_t, 2),
            "is_motor_imagery": is_mi,
        })

        logger.info(f"  Trial {trial_n+1:3d}/{total} | {info['name']:12s} | "
                     f"{len(epoch_data):4d} samples")

        # ── REST ──
        screen.fill(BG)
        txt(screen, fonts["status"], "Rest...", REST_C, SCREEN_W//2, SCREEN_H//2 - 20)
        txt(screen, fonts["info"], f"Completed: {trial_n+1}/{total}",
            REST_C, SCREEN_W//2, SCREEN_H//2 + 25)
        draw_progress(screen, trial_n + 1, total)
        draw_signal_trace(screen, signal_buf)
        pygame.display.flip()
        time.sleep(REST_DURATION)

    # ─────────── SAVE ───────────
    save_path = _save(subject_id, session_id, all_eeg, all_labels, all_meta, block)

    # ─────────── DONE ───────────
    screen.fill(BG)
    txt(screen, fonts["title"], "✓ Collection Complete!", IMAGINE_C, SCREEN_W//2, 180)
    txt(screen, fonts["status"],
        f"{len(all_labels)} trials  ·  Subject {subject_id:03d}",
        TEXT, SCREEN_W//2, 240)

    y = 290
    for cid in sorted(set(all_labels)):
        count = sum(1 for l in all_labels if l == cid)
        info = ALL_CLASSES[cid]
        txt(screen, fonts["info"], f"{info['name']:12s}: {count} trials",
            CLASS_COLORS[cid], SCREEN_W//2, y)
        y += 28

    txt(screen, fonts["info"], f"Saved: {save_path}", REST_C, SCREEN_W//2, y + 30)
    txt(screen, fonts["status"], "Press any key to exit", READY_C, SCREEN_W//2, y + 80)
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False

    pygame.quit()
    if ser:
        ser.close()
    logger.info(f"Done: {len(all_labels)} trials saved to {save_path}")


def _save(subject_id, session_id, all_eeg, all_labels, all_meta, block):
    """Save data to disk."""
    if not all_eeg:
        return None

    save_dir = RAW_DATA_DIR / f"subject_{subject_id:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = save_dir / f"session_{session_id:02d}_ref_{block}.npz"
    np.savez(
        data_path,
        data=np.array(all_eeg, dtype=object),
        labels=np.array(all_labels, dtype=np.int32),
    )
    logger.info(f"✓ Data saved: {data_path}")

    meta = {
        "subject_id": subject_id,
        "session_id": session_id,
        "block": block,
        "n_channels": 1,
        "channel": "C3 (referential to earlobe)",
        "electrode_setup": {
            "IN+": "C3 — left motor cortex (scalp)",
            "IN-": "A1 — left earlobe (reference)",
            "GND": "A2 — right earlobe (ground)",
        },
        "montage": "referential (NOT bipolar)",
        "sampling_rate_hz": SAMPLING_RATE,
        "firmware": "eeg_stream.ino v1.0 (unchanged)",
        "n_trials": len(all_labels),
        "class_map": {str(k): v["name"] for k, v in ALL_CLASSES.items()},
        "class_counts": {},
        "trial_timing": {
            "ready_s": READY_DURATION,
            "cue_s": CUE_DURATION,
            "imagine_s": IMAGINE_DURATION,
            "rest_s": REST_DURATION,
        },
        "trials": all_meta,
    }
    for cid in sorted(set(all_labels)):
        meta["class_counts"][ALL_CLASSES[cid]["name"]] = sum(1 for l in all_labels if l == cid)

    meta_path = save_dir / f"session_{session_id:02d}_ref_{block}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"✓ Metadata saved: {meta_path}")
    return data_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BCI Data Collection — 1-Channel Referential + Webcam"
    )
    parser.add_argument("--subject", type=int, required=True,
                        help="Subject ID (1-indexed)")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number")
    parser.add_argument("--port", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulated mode (no hardware)")
    parser.add_argument("--block", type=str, default="all",
                        choices=["mi", "eog", "all"],
                        help="Block: 'mi', 'eog', or 'all'")
    parser.add_argument("--trials", type=int, default=DEFAULT_MI_TRIALS,
                        help=f"Trials per MI class (default: {DEFAULT_MI_TRIALS})")
    parser.add_argument("--eog-trials", type=int, default=DEFAULT_EOG_TRIALS,
                        help=f"Trials per EOG class (default: {DEFAULT_EOG_TRIALS})")
    args = parser.parse_args()

    run_experiment(
        subject_id=args.subject,
        session_id=args.session,
        port=args.port,
        use_hardware=not args.simulate,
        block=args.block,
        mi_trials=args.trials,
        eog_trials=args.eog_trials,
    )


if __name__ == "__main__":
    main()
