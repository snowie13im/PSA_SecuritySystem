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
#  FIX: only copy the frame when the registration thread actually needs it
#  (i.e. when registration_busy is set). Skipping the copy at 30 fps when
#  no registration is in progress saves meaningful CPU and memory bandwidth.

_frame_lock   = threading.Lock()
_latest_frame = None


def set_frame(frame: np.ndarray) -> None:
    """
    Store the latest camera frame for the registration thread.
    The copy is only made when a registration prompt is active; otherwise
    we store a reference to avoid the allocation overhead every frame.
    """
    global _latest_frame
    with _frame_lock:
        # registration_busy is checked by the caller — if it is not set,
        # the registration thread won't call get_frame(), so a shallow
        # reference is safe and avoids a needless full-frame copy.
        _latest_frame = frame if not registration_busy.is_set() else frame.copy()


def get_frame():
    """Return a copy of the latest frame, or None if not yet available."""
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


#  CONCURRENCY PRIMITIVES

event_queue        = queue.Queue()
registration_busy  = threading.Event()


#  PERSISTENCE HELPERS

def _load_names() -> dict:
    if os.path.exists(NAMES_FILE):
        # FIX: use 'with' so the file handle is always released
        with open(NAMES_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_names(names: dict) -> None:
    with open(NAMES_FILE, "w", encoding="utf-8") as fh:
        json.dump(names, fh, ensure_ascii=False, indent=2)


def _person_folder(uid) -> str:
    return os.path.join(DATA_DIR, str(uid))


def _next_id(names: dict) -> int:
    return max((int(k) for k in names), default=-1) + 1


def _name_exists(names: dict, name: str):
    """Case-insensitive name lookup; returns int UID or None."""
    name_lower = name.lower()
    for uid_str, existing in names.items():
        if existing.lower() == name_lower:
            return int(uid_str)
    return None


#  MODEL TRAINING

def train_model(names: dict):
    """
    Train a fresh LBPH model from all saved face images and persist it.
    Returns None if fewer than 2 images exist (LBPH minimum requirement).

    FIX: images are read with cv2.IMREAD_GRAYSCALE and immediately resized
    in a single pass; we avoid redundant colour-space conversions and keep
    memory usage flat while iterating over potentially large image sets.
    """
    faces, labels = [], []

    for uid_str in names:
        folder = _person_folder(uid_str)
        if not os.path.isdir(folder):
            continue
        uid_int = int(uid_str)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Resize only if the stored size differs from IMG_SIZE
            if img.shape[:2] != IMG_SIZE[::-1]:
                img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(uid_int)

    if len(faces) < 2:
        print("  [!] Not enough images to train.")
        return None

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels, dtype=np.int32))
    rec.save(MODEL_FILE)
    print(f"  [OK] Model trained: {len(faces)} images, {len(set(labels))} person(s).")
    return rec


#  IN-WINDOW NAME INPUT

def _ask_name_in_window(roi: np.ndarray) -> str:
    """
    Show an OpenCV window with a face thumbnail and a typed name field.
    Returns the confirmed name string, or '' if cancelled/empty.
    """
    WIN   = "Enter Name"
    typed = ""

    th, tw   = roi.shape[:2]
    thumb    = cv2.resize(roi, (max(tw * 2, 200), max(th * 2, 200)))
    thumb_h, thumb_w = thumb.shape[:2]
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

    PANEL_W = max(thumb_w + 40, 340)
    THUMB_Y = 50
    INPUT_Y = THUMB_Y + thumb_h + 20
    PANEL_H = INPUT_Y + 80
    THUMB_X = (PANEL_W - thumb_w) // 2

    while True:
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        cv2.putText(panel, "Who is this?",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        panel[THUMB_Y:THUMB_Y + thumb_h, THUMB_X:THUMB_X + thumb_w] = thumb_bgr

        box_y1, box_y2 = INPUT_Y, INPUT_Y + 36
        cv2.rectangle(panel, (10, box_y1), (PANEL_W - 10, box_y2), (50, 50, 50), -1)
        cv2.rectangle(panel, (10, box_y1), (PANEL_W - 10, box_y2), (0, 200, 255),  1)
        cv2.putText(panel, typed + "|",
                    (16, box_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(panel, "ENTER = confirm  |  ESC = cancel",
                    (10, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if   key == 13:
            break
        elif key == 27:
            typed = ""
            break
        elif key == 8:
            typed = typed[:-1]
        elif 32 <= key <= 126:
            typed += chr(key)

    cv2.destroyWindow(WIN)
    return typed.strip()


#  GUIDED 6-POSE CAPTURE

def register_guided(cap, uid: int) -> bool:
    """
    Walk the user through POSES, saving one face crop per pose.
    Returns True on success, False if the user pressed ESC.
    """
    folder = _person_folder(uid)
    os.makedirs(folder, exist_ok=True)
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

            cv2.rectangle(disp, (0, 0), (disp.shape[1], 80), (30, 30, 30), -1)
            cv2.putText(disp, f"Photo {i+1}/{len(POSES)}  -  {pose}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            cv2.putText(disp, instruction,
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(disp, "SPACE = capture  |  ESC = cancel",
                        (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)

            for (x, y, w, h) in faces:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (255, 200, 0), 3)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cv2.destroyWindow(WIN)
                shutil.rmtree(folder, ignore_errors=True)
                return False

            if key == 32:
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
                    cv2.imwrite(os.path.join(folder, f"{base + i}.jpg"), face_img)

                    flash = disp.copy()
                    cv2.rectangle(flash, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    cv2.putText(flash, "Captured!",
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow(WIN, flash)
                    cv2.waitKey(600)
                    captured = True
                else:
                    warn = disp.copy()
                    cv2.putText(warn, "No face detected!",
                                (10, disp.shape[0] - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    cv2.imshow(WIN, warn)
                    cv2.waitKey(800)

    cv2.destroyWindow(WIN)
    return True


#  BACKGROUND REGISTRATION THREAD

def registration_thread(cap, names: dict, recogniser_ref: list) -> None:
    """
    Daemon worker that owns the full unknown-face → register/ignore workflow.
    Listens on event_queue; replaces recogniser_ref[0] after each retrain.
    """
    WIN_PROMPT = "Unknown Face"

    while True:
        try:
            evt = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if evt.get("type") == "stop":
            break

        roi = evt["roi"]

        # Build the "Unknown Face" decision window
        h, w   = roi.shape[:2]
        big    = cv2.resize(roi, (max(w * 2, 200), max(h * 2, 200)))
        bh, bw = big.shape[:2]
        disp   = np.zeros((bh + 80, bw + 20, 3), dtype=np.uint8)
        disp[40:40 + bh, 10:10 + bw] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp, "Unknown face detected",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(disp, "R = Register  |  I = Ignore",
                    (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(WIN_PROMPT, disp)

        # Wait for R / I while keeping the camera preview alive
        decision = None
        while decision is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                decision = "register"
            elif key in (ord('i'), 27):
                decision = "ignore"

            f = get_frame()
            if f is not None:
                cv2.putText(f, "Waiting for registration decision...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow("Camera (registration pending)", f)

        cv2.destroyWindow(WIN_PROMPT)
        cv2.destroyWindow("Camera (registration pending)")

        if decision == "register":
            name_input = _ask_name_in_window(roi)

            if name_input:
                existing_uid = _name_exists(names, name_input)

                if existing_uid is not None:
                    uid = existing_uid
                    print(f"  [i] '{name_input}' already registered (ID {uid}) — adding photos.")
                else:
                    uid = _next_id(names)
                    print(f"  [i] New person — assigned ID {uid}.")

                if register_guided(cap, uid):
                    names[str(uid)] = name_input
                    _save_names(names)

                    new_rec = train_model(names)
                    if new_rec is not None:
                        recogniser_ref[0] = new_rec

                    print(f"  [OK] '{name_input}' registered with ID {uid}!")
                    ask_and_save_theme(uid, name_input)
            else:
                print("  [!] Empty name — registration cancelled.")

        registration_busy.clear()
