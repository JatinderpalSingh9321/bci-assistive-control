"""
BCI Assistive Control — Mouse Control Client (Dual-Module)
============================================================
Merges two independent input channels into a single hands-free
mouse controller:

  MODULE 1 — Brain (Upside Down Labs EEG Kit)
    - LEFT motor imagery  ->  Move cursor left
    - RIGHT motor imagery ->  Move cursor right
    - Source: Flask API (/predict or /simulate)

  MODULE 2 — Camera (Webcam + MediaPipe)
    - BLINK (both eyes)   ->  Left click
    - WINK  (one eye)     ->  Hold / Release toggle (drag)
    - Source: eye_tracker.py (background thread)

Usage:
  python -m src.mouse_control --simulate --speed 40
  python -m src.mouse_control --simulate --speed 40 --no-camera
  python -m src.mouse_control --speed 30 --preview

Group No. 7 | 8th Semester Major Project
"""

import time
import requests
import pyautogui
import argparse

from src.utils import API_HOST, API_PORT, setup_logger

logger = setup_logger("mouse_control")

# Safety: moving mouse to any screen corner aborts the script
pyautogui.FAILSAFE = True


def main():
    parser = argparse.ArgumentParser(
        description="BCI Assistive Mouse Control — Brain + Camera (Hands-Free)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Poll /simulate endpoint for brain signals (no EEG hardware needed)"
    )
    parser.add_argument(
        "--speed", type=int, default=30,
        help="Pixels to move per prediction cycle (default: 30)"
    )
    parser.add_argument(
        "--poll-rate", type=float, default=0.5,
        help="Seconds between brain-signal API polls (default: 0.5)"
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Disable camera-based eye tracking (brain-only mode)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live camera preview window with EAR values"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0)"
    )
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{API_PORT}"
    endpoint = f"{base_url}/{'simulate' if args.simulate else 'predict'}"

    # ── Banner ──
    logger.info("=" * 60)
    logger.info("  BCI Assistive Mouse Control — Dual-Module")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  MODULE 1: Brain (Upside Down Labs EEG Kit)")
    logger.info(f"    Endpoint   : {endpoint}")
    logger.info(f"    Move Speed : {args.speed} px")
    logger.info(f"    Poll Rate  : {args.poll_rate}s")
    logger.info("")
    if not args.no_camera:
        logger.info("  MODULE 2: Camera (Webcam + MediaPipe FaceMesh)")
        logger.info(f"    Camera     : index {args.camera}")
        logger.info(f"    Preview    : {'ON' if args.preview else 'OFF'}")
    else:
        logger.info("  MODULE 2: Camera DISABLED (--no-camera)")
    logger.info("")
    logger.info("  Actions:")
    logger.info("    LEFT imagery   -> Move cursor left")
    logger.info("    RIGHT imagery  -> Move cursor right")
    if not args.no_camera:
        logger.info("    BLINK (both)   -> Left click")
        logger.info("    WINK (one eye) -> Hold / Release toggle")
    logger.info("")
    logger.info("  Safety: Move mouse to any screen corner to abort.")
    logger.info("=" * 60)

    # ── Check API ──
    try:
        health = requests.get(f"{base_url}/health", timeout=3)
        if health.status_code == 200:
            data = health.json()
            logger.info(f"[BRAIN] API connected. Models: {data.get('models', [])}")
        else:
            logger.warning(f"[BRAIN] API returned status {health.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logger.error(f"[BRAIN] Cannot connect to API at {base_url}. Is it running?")
        return

    # ── Start Camera Module ──
    eye_tracker = None
    if not args.no_camera:
        try:
            from src.eye_tracker import EyeTracker
            eye_tracker = EyeTracker(
                camera_index=args.camera,
                show_preview=args.preview
            )
            eye_tracker.start()
            logger.info("[CAMERA] Eye tracker started successfully.")
        except Exception as e:
            logger.warning(f"[CAMERA] Failed to start eye tracker: {e}")
            logger.warning("[CAMERA] Continuing in brain-only mode.")
            eye_tracker = None

    # ── State ──
    movement_speed = args.speed
    poll_interval = args.poll_rate
    is_holding = False
    total_actions = 0

    try:
        while True:
            # ─────────────────────────────────────
            # MODULE 2: Check camera for eye events
            # ─────────────────────────────────────
            if eye_tracker is not None:
                # Drain ALL pending eye events (camera is faster than poll rate)
                while True:
                    eye_event = eye_tracker.get_event(timeout=0.01)
                    if eye_event is None:
                        break

                    if eye_event == "BLINK":
                        total_actions += 1
                        if is_holding:
                            pyautogui.mouseUp(button='left')
                            is_holding = False
                            logger.info("[RELEASE] Released hold before click")
                        pyautogui.click(button='left')
                        logger.info(f"[CLICK]  Blink detected -> Left Click  (#{total_actions})")

                    elif eye_event == "WINK":
                        total_actions += 1
                        if not is_holding:
                            pyautogui.mouseDown(button='left')
                            is_holding = True
                            logger.info(f"[GRAB]   Wink detected -> Mouse Down (holding)  (#{total_actions})")
                        else:
                            pyautogui.mouseUp(button='left')
                            is_holding = False
                            logger.info(f"[DROP]   Wink detected -> Mouse Up (released)  (#{total_actions})")

            # ─────────────────────────────────────
            # MODULE 1: Check brain API for movement
            # ─────────────────────────────────────
            try:
                response = requests.get(endpoint, timeout=2)

                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction", "NEUTRAL")
                    confidence = data.get("confidence", 0.0)

                    if prediction == "LEFT":
                        total_actions += 1
                        pyautogui.moveRel(-movement_speed, 0, duration=0.05)
                        state = "DRAG" if is_holding else "MOVE"
                        logger.info(f"[{state}]   LEFT   ({confidence:.0%})  (#{total_actions})")

                    elif prediction == "RIGHT":
                        total_actions += 1
                        pyautogui.moveRel(movement_speed, 0, duration=0.05)
                        state = "DRAG" if is_holding else "MOVE"
                        logger.info(f"[{state}]   RIGHT  ({confidence:.0%})  (#{total_actions})")

                    else:
                        if total_actions > 0 and total_actions % 30 == 0:
                            logger.info(f"[IDLE]   Resting... (Hold={'ON' if is_holding else 'OFF'})  (#{total_actions})")

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                logger.warning("[BRAIN] API connection lost. Retrying...")
            except Exception as e:
                logger.error(f"[BRAIN] Error: {e}")

            # Poll interval
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        if is_holding:
            pyautogui.mouseUp(button='left')
        logger.info("\nMouse control terminated by user (Ctrl+C).")
    except pyautogui.FailSafeException:
        if is_holding:
            pyautogui.mouseUp(button='left')
        logger.error("\nFailsafe triggered! Mouse moved to screen corner. Terminating.")
    finally:
        if eye_tracker is not None:
            eye_tracker.stop()


if __name__ == "__main__":
    main()
