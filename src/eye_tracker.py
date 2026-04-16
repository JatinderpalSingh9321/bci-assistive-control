"""
BCI Assistive Control — Camera-Based Eye Tracker
==================================================
Uses a webcam + OpenCV + MediaPipe FaceLandmarker (Tasks API) to detect
eye blinks and winks in real-time.

  - BLINK (both eyes close briefly)  ->  Mouse click
  - WINK  (one eye stays closed)     ->  Mouse hold/release toggle

Uses the Eye Aspect Ratio (EAR) computed from facial landmarks
to determine whether each eye is open or closed.

Group No. 7 | 8th Semester Major Project
"""

import time
import threading
import queue
import os
import urllib.request

import cv2
import numpy as np

from src.utils import setup_logger, DATA_DIR

logger = setup_logger("eye_tracker")

# ──────────────────────────────────────────────
# MODEL DOWNLOAD (one-time)
# ──────────────────────────────────────────────

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = DATA_DIR / "face_landmarker.task"


def ensure_model():
    """Download the FaceLandmarker model if not already present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    logger.info(f"Downloading FaceLandmarker model to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(MODEL_PATH))
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise
    return str(MODEL_PATH)


# ──────────────────────────────────────────────
# LANDMARK INDICES (MediaPipe FaceLandmarker)
# ──────────────────────────────────────────────
# 6-point EAR landmarks for each eye

RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_POINTS  = [362, 385, 387, 263, 373, 380]

# ──────────────────────────────────────────────
# DETECTION THRESHOLDS
# ──────────────────────────────────────────────

EAR_THRESHOLD      = 0.19   # Below this = eye is closed
BLINK_MAX_FRAMES   = 5      # Max frames for a blink (quick close-open)
WINK_MIN_FRAMES    = 18     # Min frames for a wink (~600ms sustained one-eye close)
COOLDOWN_FRAMES    = 15     # Ignore events for N frames after one fires


# ──────────────────────────────────────────────
# EAR COMPUTATION
# ──────────────────────────────────────────────

def compute_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Compute Eye Aspect Ratio (EAR) for one eye.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Open eye: EAR ~ 0.25-0.35
    Closed eye: EAR < 0.20
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))

    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    if h == 0:
        return 0.3

    return (v1 + v2) / (2.0 * h)


# ──────────────────────────────────────────────
# EYE TRACKER CLASS
# ──────────────────────────────────────────────

class EyeTracker:
    """
    Threaded camera-based eye blink/wink detector.

    Usage:
        tracker = EyeTracker()
        tracker.start()
        event = tracker.get_event(timeout=0.1)  # "BLINK", "WINK", or None
        tracker.stop()
    """

    def __init__(self, camera_index=0, show_preview=False):
        self.camera_index = camera_index
        self.show_preview = show_preview
        self.event_queue = queue.Queue(maxsize=10)
        self._running = False
        self._thread = None

    def start(self):
        """Start the eye tracking thread."""
        if self._running:
            logger.warning("Eye tracker already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()
        logger.info("Eye tracker started (camera-based blink/wink detection).")

    def stop(self):
        """Stop the eye tracking thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Eye tracker stopped.")

    def get_event(self, timeout=0.1):
        """Get the next eye event ('BLINK', 'WINK', or None)."""
        try:
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _push_event(self, event_type):
        """Push an event to the queue."""
        if self.event_queue.full():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                pass
        self.event_queue.put(event_type)

    def _tracking_loop(self):
        """Internal loop: read frames, detect eyes, classify events."""
        import mediapipe as mp

        # Download model if needed
        model_path = ensure_model()

        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera (index={self.camera_index}).")
            self._running = False
            return

        logger.info(f"Camera opened: index={self.camera_index}")

        # Initialize FaceLandmarker (Tasks API — compatible with protobuf v7)
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        landmarker = FaceLandmarker.create_from_options(options)

        # State tracking
        left_closed_count = 0
        right_closed_count = 0
        both_closed_count = 0
        cooldown = 0
        frame_idx = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_idx += 1
                frame_h, frame_w = frame.shape[:2]

                # Convert to MediaPipe Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Detect landmarks
                timestamp_ms = int(frame_idx * (1000 / 30))  # assume ~30fps
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    landmarks = result.face_landmarks[0]

                    # Compute EAR for both eyes
                    left_ear = compute_ear(landmarks, LEFT_EYE_POINTS, frame_w, frame_h)
                    right_ear = compute_ear(landmarks, RIGHT_EYE_POINTS, frame_w, frame_h)

                    left_closed = left_ear < EAR_THRESHOLD
                    right_closed = right_ear < EAR_THRESHOLD

                    # Track closure durations
                    if left_closed:
                        left_closed_count += 1
                    else:
                        left_closed_count = 0

                    if right_closed:
                        right_closed_count += 1
                    else:
                        right_closed_count = 0

                    if left_closed and right_closed:
                        both_closed_count += 1
                    else:
                        # Check for events on transition from closed to open
                        if cooldown <= 0:
                            # BLINK: Both eyes were closed for a short burst
                            if both_closed_count > 0 and both_closed_count <= BLINK_MAX_FRAMES:
                                self._push_event("BLINK")
                                logger.info(f"[EYE] BLINK detected (frames={both_closed_count})")
                                cooldown = COOLDOWN_FRAMES

                            # WINK: Only one eye was closed for a sustained period
                            elif left_closed_count >= WINK_MIN_FRAMES and right_closed_count == 0:
                                self._push_event("WINK")
                                logger.info(f"[EYE] WINK detected - LEFT eye (frames={left_closed_count})")
                                cooldown = COOLDOWN_FRAMES

                            elif right_closed_count >= WINK_MIN_FRAMES and left_closed_count == 0:
                                self._push_event("WINK")
                                logger.info(f"[EYE] WINK detected - RIGHT eye (frames={right_closed_count})")
                                cooldown = COOLDOWN_FRAMES

                        both_closed_count = 0

                    if cooldown > 0:
                        cooldown -= 1

                    # Optional preview
                    if self.show_preview:
                        status = "OPEN"
                        color = (0, 255, 0)
                        if left_closed and right_closed:
                            status = "BOTH CLOSED"
                            color = (0, 0, 255)
                        elif left_closed:
                            status = "LEFT WINK"
                            color = (0, 165, 255)
                        elif right_closed:
                            status = "RIGHT WINK"
                            color = (0, 165, 255)

                        cv2.putText(frame, f"L-EAR: {left_ear:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"R-EAR: {right_ear:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, status, (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        cv2.imshow("BCI Eye Tracker", frame)

                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                else:
                    left_closed_count = 0
                    right_closed_count = 0
                    both_closed_count = 0

        finally:
            cap.release()
            landmarker.close()
            if self.show_preview:
                cv2.destroyAllWindows()
            logger.info("Camera released.")


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  BCI Eye Tracker Test")
    print("=" * 50)
    print("  Blink both eyes quickly  -> BLINK")
    print("  Close one eye for ~0.5s  -> WINK")
    print("  Press ESC in preview or Ctrl+C to quit")
    print("=" * 50)

    tracker = EyeTracker(show_preview=True)
    tracker.start()

    try:
        while True:
            event = tracker.get_event(timeout=0.5)
            if event:
                print(f"  >>> EVENT: {event}")
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
