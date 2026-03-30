import cv2
import os

# ── File / folder paths ────────────────────────────────────────────────────

# Root folder where one sub-folder per person is created to store their photos
DATA_DIR = "face_data"

# JSON file that maps integer ID (as string key) to person name
# Example content: {"0": "Alice", "1": "Bob"}
NAMES_FILE = "names.json"

# Trained LBPH model persisted to disk so it survives restarts
MODEL_FILE = "trained_model.xml"

# ── Image processing ───────────────────────────────────────────────────────

# Every face crop is resized to this before training or prediction.
# Larger = more detail but slower; 200x200 is a good balance for LBPH.
IMG_SIZE = (200, 200)

# ── Recognition tuning ────────────────────────────────────────────────────

# LBPH returns a "distance" (lower = closer match).
# Any prediction with distance BELOW this value is accepted as recognised.
# Decrease to be stricter (fewer false positives, more unknowns).
# Increase to be more lenient (more matches, higher false-positive risk).
CONFIDENCE_MAX = 80

# Seconds that must pass before a new "unknown face" alert is allowed to fire.
# Prevents the registration pop-up from re-opening while a stranger
# is still standing in front of the camera.
ALERT_COOLDOWN = 5

# Haar-cascade ignores any face bounding box smaller than this (pixels).
# Filters out tiny, distant, or partially visible faces that would give
# poor recognition results anyway.
FACE_SIZE_MIN = 60

# ── Registration poses ─────────────────────────────────────────────────────

# Six head positions the user is guided through during registration.
# More varied poses = better model generalisation.
# Each tuple is (short label displayed on screen, full instruction text).
POSES = [
    ("FRONT",      "Look directly at the camera"),
    ("LEFT",       "Turn slightly to the left"),
    ("RIGHT",      "Turn slightly to the right"),
    ("UP",         "Tilt your head slightly up"),
    ("DOWN",       "Tilt your head slightly down"),
    ("EXPRESSION", "Give a natural smile"),
]

# ── Shared OpenCV face detector ────────────────────────────────────────────

# Loaded once here and imported by both modules — avoids loading the
# XML classifier file twice and keeps initialisation in one place.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Ensure the data folder exists as soon as config is imported.
os.makedirs(DATA_DIR, exist_ok=True)
