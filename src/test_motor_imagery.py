"""
BCI Test 1 — Motor Imagery Data Collection (C3 Referential)
=============================================================
Collects LEFT / RIGHT / UP / DOWN motor imagery data.

Electrode Placement:
  IN+  → C3 (left motor cortex, part the hair)
  IN-  → Left earlobe (reference)
  GND  → Right earlobe
  OUT  → Arduino A0

Mental Tasks:
  LEFT  — imagine squeezing your LEFT fist hard
  RIGHT — imagine squeezing your RIGHT fist hard
  UP    — imagine curling your tongue to the roof of mouth
  DOWN  — imagine wiggling all your toes

Usage:
  python -m src.test_motor_imagery --subject 1 --port COM7
  python -m src.test_motor_imagery --subject 1 --simulate
  python -m src.test_motor_imagery --subject 1 --port COM7 --trials 15

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

from src.utils import SERIAL_PORT, BAUD_RATE, SAMPLING_RATE, RAW_DATA_DIR, setup_logger

logger = setup_logger("test_mi")

# ──────────────────────────────────────────────
# CLASSES
# ──────────────────────────────────────────────

CLASSES = {
    0: {
        "name": "LEFT", "arrow": "←",
        "task": "Squeeze LEFT Fist",
        "tip": "Imagine clenching your left hand as hard as you can",
    },
    1: {
        "name": "RIGHT", "arrow": "→",
        "task": "Squeeze RIGHT Fist",
        "tip": "Imagine clenching your right hand as hard as you can",
    },
    2: {
        "name": "UP", "arrow": "↑",
        "task": "Curl Tongue Upward",
        "tip": "Imagine pressing your tongue firmly to the roof of your mouth",
    },
    3: {
        "name": "DOWN", "arrow": "↓",
        "task": "Wiggle Your Toes",
        "tip": "Imagine curling and wiggling all your toes at once",
    },
}

CLASS_NAMES = [CLASSES[i]["name"] for i in range(4)]

# Timing
READY_S   = 2.0   # Fixation cross
CUE_S     = 1.0   # Show arrow + instruction
IMAGINE_S = 4.0   # Record EEG (motor imagery period)
REST_S    = 2.0   # Rest between trials
TRIALS_PER_CLASS = 25  # 25 × 4 = 100 total

# Display
SW, SH = 1100, 750
BG      = (15, 15, 25)
TXT     = (210, 210, 220)
ACCENT  = (233, 69, 96)
GREEN   = (100, 255, 100)
DIM     = (60, 60, 80)
READY_C = (180, 180, 190)
BAR_C   = (0, 180, 255)

COLORS = {0: (0, 200, 255), 1: (255, 166, 0), 2: (100, 255, 150), 3: (255, 100, 100)}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _txt(screen, font, text, color, x, y):
    s = font.render(text, True, color)
    screen.blit(s, s.get_rect(center=(x, y)))


def _arrow(screen, d, cx, cy, sz=100, color=(255, 255, 255)):
    if d == 0:
        pts = [(cx+sz, cy-sz//2), (cx-sz, cy), (cx+sz, cy+sz//2)]
    elif d == 1:
        pts = [(cx-sz, cy-sz//2), (cx+sz, cy), (cx-sz, cy+sz//2)]
    elif d == 2:
        pts = [(cx-sz//2, cy+sz), (cx, cy-sz), (cx+sz//2, cy+sz)]
    else:
        pts = [(cx-sz//2, cy-sz), (cx, cy+sz), (cx+sz//2, cy-sz)]
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (255, 255, 255), pts, 3)


def _bar(screen, cur, tot):
    y, w, h = SH - 35, SW - 100, 12
    frac = cur / max(tot, 1)
    pygame.draw.rect(screen, (40, 40, 60), (50, y, w, h), border_radius=6)
    if int(w * frac) > 0:
        pygame.draw.rect(screen, BAR_C, (50, y, int(w * frac), h), border_radius=6)
    f = pygame.font.SysFont("Consolas", 13)
    _txt(screen, f, f"{frac*100:.0f}%", TXT, SW//2, y - 10)


def _trace(screen, buf, y0=680):
    if len(buf) < 10:
        return
    f = pygame.font.SysFont("Consolas", 11)
    _txt(screen, f, "C3", (0, 200, 255), 30, y0)
    d = np.array(buf[-250:])
    if np.ptp(d) > 0:
        d = (d - np.mean(d)) / (np.ptp(d) + 1e-10) * 18
    else:
        d = np.zeros_like(d)
    pw = SW - 150
    pts = [(70 + int(i * pw / len(d)), int(y0 + v)) for i, v in enumerate(d)]
    if len(pts) >= 2:
        pygame.draw.lines(screen, (0, 200, 255), False, pts, 1)


# ──────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────

def _simulate(label, n, fs=SAMPLING_RATE):
    """Simulate referential C3 epoch."""
    t = np.linspace(0, n / fs, n, endpoint=False)
    mu   = 20 * np.sin(2 * np.pi * 10 * t)
    beta = 10 * np.sin(2 * np.pi * 20 * t)
    noise = np.random.randn(n) * 3

    if label == 0:    # LEFT → less ERD at C3 (ipsilateral)
        sig = mu * 0.85 + beta * 0.8
    elif label == 1:  # RIGHT → strong ERD at C3 (contralateral)
        sig = mu * 0.35 + beta * 0.4
    elif label == 2:  # UP (tongue) → high-beta burst
        sig = mu * 0.6 + beta * 0.5 + 12 * np.sin(2 * np.pi * 26 * t)
    else:             # DOWN (feet) → theta increase
        sig = mu * 0.7 + beta * 0.7 + 12 * np.sin(2 * np.pi * 5 * t)

    return (sig + noise + np.random.randn() * 2).astype(np.float32)


# ──────────────────────────────────────────────
# SERIAL
# ──────────────────────────────────────────────

def _connect(port):
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)
    for _ in range(20):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            break
    logger.info(f"✓ Connected: {port}")
    return ser


def _read(ser):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            return float(line.split(",")[0])
    except (ValueError, UnicodeDecodeError):
        pass
    return None


# ──────────────────────────────────────────────
# EXPERIMENT
# ──────────────────────────────────────────────

def run(subject_id, session_id=1, port=SERIAL_PORT,
        use_hw=True, trials_per_class=TRIALS_PER_CLASS):

    if pygame is None:
        logger.error("pygame not installed")
        return

    screen = pygame.display.set_mode((SW, SH))
    pygame.init()
    pygame.display.set_caption("🧠 Test 1: Motor Imagery (C3)")
    fonts = {
        "title":  pygame.font.SysFont("Segoe UI", 42, bold=True),
        "cue":    pygame.font.SysFont("Segoe UI", 110, bold=True),
        "action": pygame.font.SysFont("Segoe UI", 34),
        "status": pygame.font.SysFont("Segoe UI", 24),
        "info":   pygame.font.SysFont("Consolas", 17),
        "tip":    pygame.font.SysFont("Segoe UI", 20, italic=True),
    }

    # Serial
    ser = None
    if use_hw:
        try:
            ser = _connect(port)
        except Exception as e:
            logger.error(f"Serial: {e}. Using simulated.")
            use_hw = False

    # Trials
    trials = []
    for c in range(4):
        trials.extend([c] * trials_per_class)
    np.random.shuffle(trials)
    total = len(trials)

    all_eeg, all_labels, all_meta = [], [], []
    sig_buf = []

    # ── Welcome ──
    screen.fill(BG)
    _txt(screen, fonts["title"], "TEST 1: Motor Imagery", ACCENT, SW//2, 60)
    _txt(screen, fonts["status"], f"Subject {subject_id:03d}  ·  {total} trials",
         TXT, SW//2, 110)

    y = 170
    _txt(screen, fonts["status"], "── Electrode Placement ──", BAR_C, SW//2, y); y += 35
    _txt(screen, fonts["info"], "IN+ → C3 (scalp, left side, part the hair)",
         (0, 200, 255), SW//2, y); y += 25
    _txt(screen, fonts["info"], "IN- → Left earlobe    GND → Right earlobe",
         (0, 200, 255), SW//2, y); y += 45

    _txt(screen, fonts["status"], "── Directions ──", BAR_C, SW//2, y); y += 35
    for cid, info in CLASSES.items():
        _txt(screen, fonts["info"],
             f"  {info['arrow']}  {info['name']:6s} — {info['task']}",
             COLORS[cid], SW//2, y)
        y += 28

    y += 30
    hw = "HARDWARE" if use_hw else "SIMULATED"
    _txt(screen, fonts["info"], f"Mode: {hw}  |  Port: {port}", DIM, SW//2, y)
    y += 50
    _txt(screen, fonts["status"], "Press SPACE to start  ·  ESC to quit",
         READY_C, SW//2, y)
    pygame.display.flip()

    # Wait
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                break
        else:
            continue
        break

    # ── Trial Loop ──
    for tn, cid in enumerate(trials):
        info = CLASSES[cid]
        color = COLORS[cid]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                _save_mi(subject_id, session_id, all_eeg, all_labels, all_meta)
                pygame.quit(); return

        # Ready
        screen.fill(BG)
        _txt(screen, fonts["cue"], "+", READY_C, SW//2, SH//2 - 50)
        _txt(screen, fonts["status"], f"Trial {tn+1} / {total}", DIM, SW//2, SH//2 + 40)
        _bar(screen, tn, total); _trace(screen, sig_buf)
        pygame.display.flip()
        time.sleep(READY_S)

        # Cue
        screen.fill(BG)
        _arrow(screen, cid, SW//2, SH//2 - 40, 100, color)
        _txt(screen, fonts["action"], info["task"], color, SW//2, SH//2 + 100)
        _txt(screen, fonts["tip"], info["tip"], (140, 140, 160), SW//2, SH//2 + 145)
        _bar(screen, tn, total)
        pygame.display.flip()
        time.sleep(CUE_S)

        # Imagine + Record
        samples = []
        t0 = time.time()

        if use_hw and ser:
            while time.time() - t0 < IMAGINE_S:
                val = _read(ser)
                if val is not None:
                    samples.append(val)
                    sig_buf.append(val)
                    if len(sig_buf) > 600:
                        sig_buf = sig_buf[-600:]

                elapsed = time.time() - t0
                if int(elapsed * 3) != int((elapsed - 0.01) * 3):
                    screen.fill(BG)
                    _arrow(screen, cid, SW//2, SH//2 - 50, 100, color)
                    _txt(screen, fonts["action"], "IMAGINE NOW", GREEN, SW//2, SH//2 + 80)
                    _txt(screen, fonts["status"], f"{IMAGINE_S - elapsed:.1f}s",
                         READY_C, SW//2, SH//2 + 130)
                    _txt(screen, fonts["info"], f"Samples: {len(samples)}",
                         DIM, SW//2, SH//2 + 160)
                    _bar(screen, tn, total); _trace(screen, sig_buf)
                    pygame.display.flip()

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                        _save_mi(subject_id, session_id, all_eeg, all_labels, all_meta)
                        pygame.quit(); ser.close(); return
        else:
            screen.fill(BG)
            _arrow(screen, cid, SW//2, SH//2 - 50, 100, color)
            _txt(screen, fonts["action"], "IMAGINE NOW", GREEN, SW//2, SH//2 + 80)
            _bar(screen, tn, total)
            pygame.display.flip()

            n_sim = int(IMAGINE_S * SAMPLING_RATE)
            samples = _simulate(cid, n_sim).tolist()
            sig_buf.extend(samples[-300:])
            if len(sig_buf) > 600:
                sig_buf = sig_buf[-600:]
            time.sleep(IMAGINE_S)

        arr = np.array(samples, dtype=np.float32)
        all_eeg.append(arr)
        all_labels.append(cid)
        all_meta.append({
            "trial": tn + 1, "class_id": cid,
            "class_name": info["name"],
            "n_samples": len(samples),
        })

        logger.info(f"  Trial {tn+1:3d}/{total} | {info['name']:6s} | {len(samples)} samples")

        # Rest
        screen.fill(BG)
        _txt(screen, fonts["status"], "Rest...", DIM, SW//2, SH//2)
        _bar(screen, tn + 1, total); _trace(screen, sig_buf)
        pygame.display.flip()
        time.sleep(REST_S)

    # Save
    path = _save_mi(subject_id, session_id, all_eeg, all_labels, all_meta)

    # Done
    screen.fill(BG)
    _txt(screen, fonts["title"], "✓ Test 1 Complete!", GREEN, SW//2, 200)
    _txt(screen, fonts["status"], f"{len(all_labels)} trials saved", TXT, SW//2, 260)
    for c in range(4):
        cnt = sum(1 for l in all_labels if l == c)
        _txt(screen, fonts["info"], f"{CLASS_NAMES[c]:6s}: {cnt} trials",
             COLORS[c], SW//2, 310 + c * 28)
    _txt(screen, fonts["info"], f"Saved: {path}", DIM, SW//2, 440)
    _txt(screen, fonts["status"], "Press any key to exit", READY_C, SW//2, 490)
    _txt(screen, fonts["tip"],
         "Now change electrodes to FOREHEAD and run Test 2!",
         ACCENT, SW//2, 540)
    pygame.display.flip()

    while True:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                pygame.quit()
                if ser: ser.close()
                return


def _save_mi(subj, sess, eeg, labels, meta):
    if not eeg:
        return None
    d = RAW_DATA_DIR / f"subject_{subj:03d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"session_{sess:02d}_motor_imagery.npz"
    np.savez(path,
             data=np.array(eeg, dtype=object),
             labels=np.array(labels, dtype=np.int32))
    logger.info(f"✓ Saved: {path}")

    meta_out = {
        "test": "Test 1 — Motor Imagery",
        "subject_id": subj, "session_id": sess,
        "electrode": "C3 referential (IN+=C3, IN-=earlobe)",
        "classes": CLASS_NAMES,
        "n_trials": len(labels),
        "class_counts": {CLASS_NAMES[c]: sum(1 for l in labels if l == c) for c in range(4)},
        "sampling_rate_hz": SAMPLING_RATE,
        "trials": meta,
    }
    mp = d / f"session_{sess:02d}_motor_imagery_meta.json"
    with open(mp, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    return path


def main():
    p = argparse.ArgumentParser(description="Test 1: Motor Imagery (C3)")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--port", type=str, default=SERIAL_PORT)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--trials", type=int, default=TRIALS_PER_CLASS,
                   help=f"Trials per direction (default: {TRIALS_PER_CLASS})")
    a = p.parse_args()
    run(a.subject, a.session, a.port, not a.simulate, a.trials)


if __name__ == "__main__":
    main()
