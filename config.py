import cv2
import os
import sys

# ── File / folder paths ────────────────────────────────────────────────────

DATA_DIR   = "face_data"
NAMES_FILE = "names.json"
MODEL_FILE = "trained_model.xml"

# ── Image processing ───────────────────────────────────────────────────────

IMG_SIZE = (200, 200)

# ── Recognition tuning ────────────────────────────────────────────────────

CONFIDENCE_MAX  = 60
ALERT_COOLDOWN  = 5
FACE_SIZE_MIN   = 80

# ── Registration poses ─────────────────────────────────────────────────────

POSES = [
    ("FRONT",      "Look directly at the camera"),
    ("LEFT",       "Turn slightly to the left"),
    ("RIGHT",      "Turn slightly to the right"),
    ("UP",         "Tilt your head slightly up"),
    ("DOWN",       "Tilt your head slightly down"),
    ("EXPRESSION", "Give a natural smile"),
]

# ── Shared OpenCV face detector ────────────────────────────────────────────

# FIX: validate the cascade loaded correctly instead of silently continuing
# with a broken object, which causes confusing errors later at detectMultiScale.
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade  = cv2.CascadeClassifier(_cascade_path)

if face_cascade.empty():
    sys.exit(
        f"[ERROR] Failed to load Haar cascade from:\n  {_cascade_path}\n"
        "Make sure OpenCV is installed correctly."
    )

os.makedirs(DATA_DIR, exist_ok=True)
