import cv2
import os
import json
import shutil
import threading
import queue
import numpy as np

from config import (
    DATA_DIR, NAMES_FILE, MODEL_FILE,
    IMG_SIZE, FACE_SIZE_MIN, POSES,
    face_cascade,
)
from theme_song import ask_and_save_theme


#  THREAD-SAFE FRAME SHARING
#  The main loop calls set_frame() on every iteration.
#  The registration thread calls get_frame() to refresh its camera preview
#  while waiting for the user to decide whether to register or ignore.

_frame_lock   = threading.Lock()   # mutex — prevents simultaneous read + write
_latest_frame = None               # most recent BGR frame from the webcam


def set_frame(frame: np.ndarray) -> None:
    """
    Overwrite the shared frame with the latest camera capture.
    Called from the main loop on every iteration; the lock guarantees
    the registration thread never reads a partially overwritten frame.
    """
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


def get_frame():
    """
    Return a safe copy of the latest camera frame, or None if the camera
    has not produced a frame yet.  Called from the registration thread.
    """
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


#  CONCURRENCY PRIMITIVES
#  Imported by face_recognition.py so both sides share the same objects.

# The main loop puts dicts onto this queue to signal events:
#   {"type": "unknown", "roi": <grayscale face crop>}  — unknown face detected
#   {"type": "stop"}                                    — shutdown signal
event_queue = queue.Queue()

# Set while a registration prompt is open.  The main loop checks this before
# firing a new unknown-face alert so only one prompt is ever open at a time.
registration_busy = threading.Event()


#  PERSISTENCE HELPERS
#  Private to this module — only registration needs to write to disk.

def _load_names() -> dict:
    """
    Load the ID→name mapping from NAMES_FILE.
    Returns {} if the file does not exist yet (first run).
    JSON keys are always strings; callers cast to int where needed.
    """
    if os.path.exists(NAMES_FILE):
        return json.load(open(NAMES_FILE, encoding="utf-8"))
    return {}


def _save_names(names: dict) -> None:
    """Persist the current ID→name mapping to NAMES_FILE."""
    json.dump(names, open(NAMES_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def _person_folder(uid) -> str:
    """Return the image folder path for a given person ID, e.g. 'face_data/3'."""
    return os.path.join(DATA_DIR, str(uid))


def _next_id(names: dict) -> int:
    """
    Return the next free unique integer ID.
    Always max(existing_ids) + 1, so:
      - IDs never decrease or get reused after deletion.
      - Gaps in the sequence are harmless.
    Returns 0 if the database is empty.
    """
    return max((int(k) for k in names), default=-1) + 1


def _name_exists(names: dict, name: str):
    """
    Case-insensitive search for 'name' in the database.
    Returns the existing integer UID if found, None if brand new.

    Central duplicate-ID enforcement point: every registration path
    calls this before allocating a new ID, so one person always maps
    to exactly one ID regardless of how many times they are registered.
    """
    for uid_str, existing in names.items():
        if existing.lower() == name.lower():
            return int(uid_str)
    return None


#  MODEL TRAINING
#  Kept here because registration is the only action that changes the
#  training data.  After every successful registration the model is
#  retrained so the new person is immediately recognisable.

def train_model(names: dict):
    """
    Scan DATA_DIR for all saved face images, fit a fresh LBPH model,
    save it to MODEL_FILE, and return the trained recogniser.

    Returns None if there are fewer than 2 images in total (LBPH minimum).
    Retraining from scratch is fast enough for the small datasets used here
    and avoids the complexity of incremental LBPH updates.
    """
    faces, labels = [], []

    for uid_str in names:
        folder = _person_folder(uid_str)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, IMG_SIZE))
            labels.append(int(uid_str))

    if len(faces) < 2:
        print("  [!] Not enough images to train.")
        return None

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels))
    rec.save(MODEL_FILE)
    print(f"  [OK] Model trained: {len(faces)} images, {len(set(labels))} person(s).")
    return rec


#  IN-WINDOW NAME INPUT
#  Replaces the terminal input() call with a text field drawn directly
#  inside an OpenCV window so the whole interaction stays on screen.

def _ask_name_in_window(roi: np.ndarray) -> str:
    """
    Display a window containing the unknown face thumbnail and an interactive
    text field.  The user types a name and presses ENTER or ESC.

    Supported keys:
      Printable ASCII (32–126) — appended to the typed string
      BACKSPACE (8)            — removes the last character
      ENTER (13)               — confirms and closes the window
      ESC (27)                 — cancels; returns an empty string

    Parameters
    ----------
    roi : np.ndarray
        Grayscale crop of the unknown face used as the thumbnail.

    Returns
    -------
    The confirmed name string, or '' if cancelled / empty.
    """
    WIN   = "Enter Name"
    typed = ""

    # ── Build fixed layout dimensions from the thumbnail size ────────────────
    th, tw  = roi.shape[:2]
    thumb   = cv2.resize(roi, (max(tw * 2, 200), max(th * 2, 200)))
    thumb_h, thumb_w = thumb.shape[:2]
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

    PANEL_W  = max(thumb_w + 40, 340)    # window width — at least 340 px
    THUMB_Y  = 50                         # top of the face thumbnail
    INPUT_Y  = THUMB_Y + thumb_h + 20    # top of the input box
    PANEL_H  = INPUT_Y + 80              # total window height
    THUMB_X  = (PANEL_W - thumb_w) // 2  # centre the thumbnail horizontally

    while True:
        # Rebuild the panel every frame so the typed text updates in real time
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        # Title
        cv2.putText(panel, "Who is this?",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Face thumbnail
        panel[THUMB_Y:THUMB_Y + thumb_h, THUMB_X:THUMB_X + thumb_w] = thumb_bgr

        # Input box — dark filled rectangle with a cyan border
        box_y1, box_y2 = INPUT_Y, INPUT_Y + 36
        cv2.rectangle(panel, (10, box_y1), (PANEL_W - 10, box_y2), (50, 50, 50), -1)
        cv2.rectangle(panel, (10, box_y1), (PANEL_W - 10, box_y2), (0, 200, 255),  1)

        # Typed text with a static cursor character so it looks like a real field
        cv2.putText(panel, typed + "|",
                    (16, box_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # Hint at the bottom
        cv2.putText(panel, "ENTER = confirm  |  ESC = cancel",
                    (10, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if   key == 13:              # ENTER — confirm input
            break
        elif key == 27:              # ESC   — cancel
            typed = ""
            break
        elif key == 8:               # BACKSPACE — delete last character
            typed = typed[:-1]
        elif 32 <= key <= 126:       # printable ASCII — append character
            typed += chr(key)

    cv2.destroyWindow(WIN)
    return typed.strip()


#  GUIDED 6-POSE CAPTURE
#  Walks the user through POSES one by one, capturing one photo per pose.
#  Runs inside the registration thread, using the same cap object as the
#  main loop (OpenCV VideoCapture is not thread-safe for simultaneous reads,
#  but since the registration thread takes over cap exclusively during this
#  function the main loop is paused — see registration_thread below).

def register_guided(cap, uid: int) -> bool:
    """
    Guide the user through 6 poses and save one face photo per pose.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The shared webcam object.  The main loop should NOT read from cap
        while this function is running (it is blocked waiting on the thread).
    uid : int
        Person ID — photos are saved to DATA_DIR/<uid>/.

    Returns
    -------
    True  — all 6 photos saved successfully.
    False — user pressed ESC; partial photos are deleted.
    """
    folder = _person_folder(uid)
    os.makedirs(folder, exist_ok=True)

    # Start numbering from the existing photo count so we never overwrite
    # photos that were saved in a previous registration session for this person.
    base = len(os.listdir(folder))
    WIN  = "Registration"

    for i, (pose, instruction) in enumerate(POSES):
        print(f"\n  [photo {i+1}/{len(POSES)}] {pose}: {instruction}")
        print("  SPACE = capture  |  ESC = cancel")

        captured = False

        while not captured:
            ret, frame = cap.read()
            if not ret:
                cv2.destroyWindow(WIN)
                return False

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(FACE_SIZE_MIN, FACE_SIZE_MIN)
            )
            disp = frame.copy()

            # Dark header bar ─────────────────────────────────────────────────
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 80), (30, 30, 30), -1)
            cv2.putText(disp, f"Photo {i+1}/{len(POSES)}  -  {pose}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            cv2.putText(disp, instruction,
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Footer hint ─────────────────────────────────────────────────────
            cv2.putText(disp, "SPACE = capture  |  ESC = cancel",
                        (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)

            # Face bounding boxes ─────────────────────────────────────────────
            for (x, y, w, h) in faces:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (255, 200, 0), 3)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:   # ESC — abort and clean up
                cv2.destroyWindow(WIN)
                shutil.rmtree(folder, ignore_errors=True)
                return False

            if key == 32:   # SPACE — capture this pose
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
                    cv2.imwrite(os.path.join(folder, f"{base + i}.jpg"), face_img)

                    # Green flash to confirm the capture visually
                    flash = disp.copy()
                    cv2.rectangle(flash, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    cv2.putText(flash, "Captured!",
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow(WIN, flash)
                    cv2.waitKey(600)
                    captured = True
                else:
                    # Warn the user and let them reposition
                    warn = disp.copy()
                    cv2.putText(warn, "No face detected!",
                                (10, disp.shape[0] - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    cv2.imshow(WIN, warn)
                    cv2.waitKey(800)

    cv2.destroyWindow(WIN)
    return True


#  BACKGROUND REGISTRATION THREAD
#  Runs as a daemon thread started by face_recognition.py.
#  Listens on event_queue for unknown-face events, handles the full
#  R/I prompt → name input → guided capture → retrain workflow, then
#  clears registration_busy so the main loop can fire the next alert.

def registration_thread(cap, names: dict, recogniser_ref: list) -> None:
    """
    Background worker that owns the entire registration workflow.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Shared webcam object — used exclusively by this thread during capture.
    names : dict
        Live ID-> name mapping shared with face_recognition.py.
        Modified in-place here; face_recognition.py reads it each frame.
    recogniser_ref : list[recogniser | None]
        One-element list so this thread can replace the recogniser object
        and the main loop will use the updated model on its next iteration
        (Python cannot rebind a plain variable across thread boundaries
        without globals; a mutable container works cleanly instead).
    """
    WIN_PROMPT = "Unknown Face"

    while True:
        # Wait for an event; the 0.5 s timeout keeps the thread responsive
        # to the "stop" shutdown signal even when no events are queued.
        try:
            evt = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if evt.get("type") == "stop":
            break   # clean shutdown requested by main()

        roi = evt["roi"]   # grayscale face crop sent by the main loop

        # ── Show the "Unknown Face" decision window ───────────────────────────
        h, w  = roi.shape[:2]
        big   = cv2.resize(roi, (max(w * 2, 200), max(h * 2, 200)))
        bh, bw = big.shape[:2]
        disp  = np.zeros((bh + 80, bw + 20, 3), dtype=np.uint8)
        disp[40:40 + bh, 10:10 + bw] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp, "Unknown face detected",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(disp, "R = Register  |  I = Ignore",
                    (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(WIN_PROMPT, disp)

        # ── Wait for R / I decision while keeping the camera preview alive ────
        decision = None
        while decision is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                decision = "register"
            elif key in (ord('i'), 27):   # I or ESC = ignore
                decision = "ignore"

            # Refresh the "camera pending" preview so the user can see the feed
            f = get_frame()
            if f is not None:
                cv2.putText(f, "Waiting for registration decision...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow("Camera (registration pending)", f)

        cv2.destroyWindow(WIN_PROMPT)
        cv2.destroyWindow("Camera (registration pending)")

        # ── Registration flow ─────────────────────────────────────────────────
        if decision == "register":

            # Ask for the name entirely inside an OpenCV window (no terminal)
            name_input = _ask_name_in_window(roi)

            if name_input:
                # ── DUPLICATE ID PREVENTION ───────────────────────────────────
                # Check whether this name already has an ID.  If yes, reuse it
                # so the same person is never assigned two different IDs.
                # If no, allocate max_id + 1 (gaps from deletions are fine).
                existing_uid = _name_exists(names, name_input)

                if existing_uid is not None:
                    uid = existing_uid
                    print(f"  [i] '{name_input}' already registered (ID {uid}) — adding photos.")
                else:
                    uid = _next_id(names)
                    print(f"  [i] New person — assigned ID {uid}.")

                # Run the guided capture; retrain only if all 6 poses succeed
                if register_guided(cap, uid):
                    names[str(uid)] = name_input
                    _save_names(names)

                    # Replace the recogniser in the shared ref so the main loop
                    # picks up the updated model on its very next frame.
                    new_rec = train_model(names)
                    if new_rec is not None:
                        recogniser_ref[0] = new_rec

                    print(f"  [OK] '{name_input}' registered with ID {uid}!")

                    # Perguntar se o utilizador quer uma theme song
                    ask_and_save_theme(uid, name_input)
            else:
                print("  [!] Empty name — registration cancelled.")

        # Allow the main loop to fire the next unknown-face alert
        registration_busy.clear()
