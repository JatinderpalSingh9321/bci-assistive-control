"""
BCI Test 2 — Blink & Wink Detection (Fp1 Forehead)
=====================================================
Collects BLINK / WINK_LEFT / WINK_RIGHT data from forehead electrode.

**Run this AFTER Test 1.** Move the electrodes:

Electrode Placement (change from Test 1):
  IN+  → Fp1 (left forehead, above left eyebrow)
  IN-  → Left earlobe (reference) — same as before
  GND  → Right earlobe — same as before
  OUT  → Arduino A0 — same as before

Why Fp1 works for blink/wink:
  BLINK      → HUGE signal (200+ µV) — both eyelids = massive EOG artifact
  WINK_LEFT  → LARGE signal — left eyelid close + Fp1 is ipsilateral
  WINK_RIGHT → MEDIUM signal — right eyelid only, Fp1 picks up less

  The amplitude asymmetry between wink types makes classification easy!

Usage:
  python -m src.test_blink_wink --subject 1 --port COM7
  python -m src.test_blink_wink --subject 1 --simulate
  python -m src.test_blink_wink --subject 1 --port COM7 --trials 15

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

logger = setup_logger("test_bw")

# ──────────────────────────────────────────────
# CLASSES
# ──────────────────────────────────────────────

CLASSES = {
    0: {
        "name": "BLINK",
        "icon": "◉ ◉",
        "task": "Close BOTH Eyes (1 second)",
        "tip": "Close both eyes firmly for about 1 second, then open",
    },
    1: {
        "name": "WINK_LEFT",
        "icon": "◉  ○",
        "task": "Close LEFT Eye Only",
        "tip": "Wink with your LEFT eye — keep the right eye open",
    },
    2: {
        "name": "WINK_RIGHT",
        "icon": "○  ◉",
        "task": "Close RIGHT Eye Only",
        "tip": "Wink with your RIGHT eye — keep the left eye open",
    },
}

CLASS_NAMES = [CLASSES[i]["name"] for i in range(3)]

# Timing — shorter for eye actions (they're fast)
READY_S   = 1.5
CUE_S     = 1.0
ACTION_S  = 3.0   # Shorter than MI — blinks are quick
REST_S    = 1.5
TRIALS_PER_CLASS = 25  # 25 × 3 = 75 total

# Display
SW, SH = 1100, 750
BG      = (15, 15, 25)
TXT     = (210, 210, 220)
ACCENT  = (233, 69, 96)
GREEN   = (100, 255, 100)
DIM     = (60, 60, 80)
READY_C = (180, 180, 190)
BAR_C   = (0, 180, 255)
YELLOW  = (255, 255, 100)
PURPLE  = (200, 150, 255)

COLORS = {0: YELLOW, 1: PURPLE, 2: PURPLE}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _txt(screen, font, text, color, x, y):
    s = font.render(text, True, color)
    screen.blit(s, s.get_rect(center=(x, y)))


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
    _txt(screen, f, "Fp1", YELLOW, 30, y0)
    d = np.array(buf[-250:])
    if np.ptp(d) > 0:
        d = (d - np.mean(d)) / (np.ptp(d) + 1e-10) * 18
    else:
        d = np.zeros_like(d)
    pw = SW - 150
    pts = [(70 + int(i * pw / len(d)), int(y0 + v)) for i, v in enumerate(d)]
    if len(pts) >= 2:
        pygame.draw.lines(screen, YELLOW, False, pts, 1)


def _draw_eyes(screen, cid, cx, cy, size=60):
    """Draw stylized eye icons for blink/wink."""
    gap = size * 2

    # Left eye
    left_open = (cid != 1)  # closed if WINK_LEFT
    right_open = (cid != 2)  # closed if WINK_RIGHT

    if cid == 0:  # BLINK — both closed
        left_open = False
        right_open = False

    for side, is_open, x_off in [("L", left_open, -gap//2), ("R", right_open, gap//2)]:
        ex = cx + x_off
        if is_open:
            # Open eye — circle with pupil
            pygame.draw.ellipse(screen, (255, 255, 255), (ex - size//2, cy - size//3, size, size*2//3), 2)
            pygame.draw.circle(screen, (255, 255, 255), (ex, cy), size // 6)
        else:
            # Closed eye — horizontal line with curve
            pygame.draw.arc(screen, YELLOW,
                           (ex - size//2, cy - size//4, size, size//2),
                           3.14, 6.28, 3)
            pygame.draw.line(screen, YELLOW,
                           (ex - size//2, cy), (ex + size//2, cy), 3)


# ──────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────

def _simulate(label, n, fs=SAMPLING_RATE):
    """Simulate Fp1 electrode signal for blink/wink."""
    base = np.random.randn(n) * 5  # Forehead baseline noise

    if label == 0:  # BLINK — huge symmetric peaks
        n_blinks = np.random.randint(2, 4)
        positions = np.sort(np.random.choice(
            range(n // 6, 5 * n // 6), size=n_blinks, replace=False
        ))
        for pos in positions:
            width = np.random.randint(20, 40)
            amp = np.random.uniform(150, 250)
            artifact = amp * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
            base += artifact

    elif label == 1:  # WINK_LEFT — large peaks (ipsilateral to Fp1)
        n_winks = np.random.randint(2, 4)
        positions = np.sort(np.random.choice(
            range(n // 6, 5 * n // 6), size=n_winks, replace=False
        ))
        for pos in positions:
            width = np.random.randint(15, 30)
            amp = np.random.uniform(120, 200)  # Large — same side as electrode
            artifact = amp * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
            base += artifact

    elif label == 2:  # WINK_RIGHT — smaller peaks (contralateral to Fp1)
        n_winks = np.random.randint(2, 4)
        positions = np.sort(np.random.choice(
            range(n // 6, 5 * n // 6), size=n_winks, replace=False
        ))
        for pos in positions:
            width = np.random.randint(12, 25)
            amp = np.random.uniform(60, 120)  # Smaller — opposite side
            artifact = amp * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
            base += artifact

    return base.astype(np.float32)


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
    pygame.display.set_caption("👁 Test 2: Blink & Wink (Fp1)")
    fonts = {
        "title":  pygame.font.SysFont("Segoe UI", 42, bold=True),
        "cue":    pygame.font.SysFont("Segoe UI", 80, bold=True),
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
    for c in range(3):
        trials.extend([c] * trials_per_class)
    np.random.shuffle(trials)
    total = len(trials)

    all_eeg, all_labels, all_meta = [], [], []
    sig_buf = []

    # ── Welcome ──
    screen.fill(BG)
    _txt(screen, fonts["title"], "TEST 2: Blink & Wink Detection", ACCENT, SW//2, 50)
    _txt(screen, fonts["status"], f"Subject {subject_id:03d}  ·  {total} trials",
         TXT, SW//2, 100)

    y = 160
    _txt(screen, fonts["status"], "── Move Electrodes! ──", ACCENT, SW//2, y); y += 35
    _txt(screen, fonts["info"], "IN+ → Fp1 (LEFT FOREHEAD, above left eyebrow)",
         YELLOW, SW//2, y); y += 25
    _txt(screen, fonts["info"], "IN- → Left earlobe (same as before)",
         YELLOW, SW//2, y); y += 25
    _txt(screen, fonts["info"], "GND → Right earlobe (same as before)",
         YELLOW, SW//2, y); y += 45

    _txt(screen, fonts["status"], "── Eye Actions ──", BAR_C, SW//2, y); y += 35
    for cid, info in CLASSES.items():
        _txt(screen, fonts["info"],
             f"  {info['icon']}  {info['name']:12s} — {info['task']}",
             COLORS[cid], SW//2, y)
        y += 28

    y += 30
    hw = "HARDWARE" if use_hw else "SIMULATED"
    _txt(screen, fonts["info"], f"Mode: {hw}  |  Port: {port}", DIM, SW//2, y)
    y += 50
    _txt(screen, fonts["status"], "Press SPACE to start  ·  ESC to quit",
         READY_C, SW//2, y)
    pygame.display.flip()

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
                _save_bw(subject_id, session_id, all_eeg, all_labels, all_meta)
                pygame.quit(); return

        # Ready
        screen.fill(BG)
        _txt(screen, fonts["cue"], "+", READY_C, SW//2, SH//2 - 50)
        _txt(screen, fonts["status"], f"Trial {tn+1} / {total}", DIM, SW//2, SH//2 + 40)
        _bar(screen, tn, total); _trace(screen, sig_buf)
        pygame.display.flip()
        time.sleep(READY_S)

        # Cue — show eye diagram
        screen.fill(BG)
        _draw_eyes(screen, cid, SW//2, SH//2 - 50, 60)
        _txt(screen, fonts["action"], info["task"], color, SW//2, SH//2 + 60)
        _txt(screen, fonts["tip"], info["tip"], (140, 140, 160), SW//2, SH//2 + 105)
        _bar(screen, tn, total)
        pygame.display.flip()
        time.sleep(CUE_S)

        # Perform + Record
        samples = []
        t0 = time.time()

        screen.fill(BG)
        _draw_eyes(screen, cid, SW//2, SH//2 - 60, 60)
        _txt(screen, fonts["action"], "DO IT NOW!", GREEN, SW//2, SH//2 + 60)
        _txt(screen, fonts["status"], info["name"], color, SW//2, SH//2 + 100)
        _bar(screen, tn, total)
        pygame.display.flip()

        if use_hw and ser:
            while time.time() - t0 < ACTION_S:
                val = _read(ser)
                if val is not None:
                    samples.append(val)
                    sig_buf.append(val)
                    if len(sig_buf) > 600:
                        sig_buf = sig_buf[-600:]

                elapsed = time.time() - t0
                if int(elapsed * 3) != int((elapsed - 0.01) * 3):
                    screen.fill(BG)
                    _draw_eyes(screen, cid, SW//2, SH//2 - 60, 60)
                    _txt(screen, fonts["action"], "DO IT NOW!", GREEN, SW//2, SH//2 + 60)
                    _txt(screen, fonts["status"], f"{ACTION_S - elapsed:.1f}s",
                         READY_C, SW//2, SH//2 + 100)
                    _txt(screen, fonts["info"], f"Samples: {len(samples)}",
                         DIM, SW//2, SH//2 + 130)
                    _bar(screen, tn, total); _trace(screen, sig_buf)
                    pygame.display.flip()

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                        _save_bw(subject_id, session_id, all_eeg, all_labels, all_meta)
                        pygame.quit(); ser.close(); return
        else:
            n_sim = int(ACTION_S * SAMPLING_RATE)
            samples = _simulate(cid, n_sim).tolist()
            sig_buf.extend(samples[-300:])
            if len(sig_buf) > 600:
                sig_buf = sig_buf[-600:]
            time.sleep(ACTION_S)

        arr = np.array(samples, dtype=np.float32)
        all_eeg.append(arr)
        all_labels.append(cid)
        all_meta.append({
            "trial": tn + 1, "class_id": cid,
            "class_name": info["name"],
            "n_samples": len(samples),
        })

        logger.info(f"  Trial {tn+1:3d}/{total} | {info['name']:12s} | {len(samples)} samples")

        # Rest
        screen.fill(BG)
        _txt(screen, fonts["status"], "Rest...", DIM, SW//2, SH//2)
        _bar(screen, tn + 1, total); _trace(screen, sig_buf)
        pygame.display.flip()
        time.sleep(REST_S)

    # Save
    path = _save_bw(subject_id, session_id, all_eeg, all_labels, all_meta)

    # Done
    screen.fill(BG)
    _txt(screen, fonts["title"], "✓ Test 2 Complete!", GREEN, SW//2, 200)
    _txt(screen, fonts["status"], f"{len(all_labels)} trials saved", TXT, SW//2, 260)
    for c in range(3):
        cnt = sum(1 for l in all_labels if l == c)
        _txt(screen, fonts["info"], f"{CLASS_NAMES[c]:12s}: {cnt} trials",
             COLORS[c], SW//2, 310 + c * 28)
    _txt(screen, fonts["info"], f"Saved: {path}", DIM, SW//2, 420)
    _txt(screen, fonts["status"], "Press any key to exit", READY_C, SW//2, 470)
    _txt(screen, fonts["tip"],
         "Both tests done! Now run training pipeline.",
         GREEN, SW//2, 520)
    pygame.display.flip()

    while True:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                pygame.quit()
                if ser: ser.close()
                return


def _save_bw(subj, sess, eeg, labels, meta):
    if not eeg:
        return None
    d = RAW_DATA_DIR / f"subject_{subj:03d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"session_{sess:02d}_blink_wink.npz"
    np.savez(path,
             data=np.array(eeg, dtype=object),
             labels=np.array(labels, dtype=np.int32))
    logger.info(f"✓ Saved: {path}")

    meta_out = {
        "test": "Test 2 — Blink & Wink",
        "subject_id": subj, "session_id": sess,
        "electrode": "Fp1 referential (IN+=Fp1 forehead, IN-=earlobe)",
        "classes": CLASS_NAMES,
        "n_trials": len(labels),
        "class_counts": {CLASS_NAMES[c]: sum(1 for l in labels if l == c) for c in range(3)},
        "sampling_rate_hz": SAMPLING_RATE,
        "trials": meta,
    }
    mp = d / f"session_{sess:02d}_blink_wink_meta.json"
    with open(mp, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    return path


def main():
    p = argparse.ArgumentParser(description="Test 2: Blink & Wink (Fp1)")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--port", type=str, default=SERIAL_PORT)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--trials", type=int, default=TRIALS_PER_CLASS,
                   help=f"Trials per action (default: {TRIALS_PER_CLASS})")
    a = p.parse_args()
    run(a.subject, a.session, a.port, not a.simulate, a.trials)


if __name__ == "__main__":
    main()
