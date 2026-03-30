import cv2
import os
import json
import shutil
import time
import threading
import numpy as np

# Import shared constants and the cascade detector
from config import (
    DATA_DIR, NAMES_FILE, MODEL_FILE,
    IMG_SIZE, CONFIDENCE_MAX, ALERT_COOLDOWN, FACE_SIZE_MIN,
    face_cascade,
)

# Import everything the main loop needs from the registration subsystem
from registration import (
    set_frame,              # write latest frame for the reg. thread to read
    event_queue,            # queue to send unknown-face events
    registration_busy,      # Event flag: True while a prompt is open
    registration_thread,    # the background worker function
    train_model,            # used after delete to retrain / remove model
)


#  PERSISTENCE HELPERS  (read-only from this module's perspective)
#  registration.py owns all writes; face_recognition.py only reads names and the trained model.

def load_names() -> dict:
    """
    Load the ID -> name mapping from NAMES_FILE.
    Returns {} on first run (file does not exist yet). JSON keys are always strings; cast to int where needed.
    """
    if os.path.exists(NAMES_FILE):
        return json.load(open(NAMES_FILE, encoding="utf-8"))
    return {}


def _save_names(names: dict) -> None:
    """Persist the ID ->name mapping — called only by the DATABASE management helpers."""
    json.dump(names, open(NAMES_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def _person_folder(uid) -> str:
    """Return the image folder path for a given person ID, e.g. 'face_data/3'."""
    return os.path.join(DATA_DIR, str(uid))


def load_or_train_model(names: dict):
    """
    Load the saved LBPH model from disk if it exists, otherwise train fresh.
    Returns None if there is not enough data to train yet.
    """
    if os.path.exists(MODEL_FILE) and names:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(MODEL_FILE)
        return rec
    return train_model(names)

#  DATABASE MANAGEMENT HELPERS
#  Called from the management menu between camera sessions.

def list_people() -> dict:
    """
    Print a formatted table of all registered people with ID and photo count.
    Returns the names dict so the menu loop can chain calls if needed.
    """
    names = load_names()
    if not names:
        print("\n  (no people registered)")
        return names

    print(f"\n  {'ID':<6} {'Name':<30} Photos")
    print("  " + "-" * 44)
    for uid_str, name in names.items():
        folder = _person_folder(uid_str)
        count  = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        print(f"  {uid_str:<6} {name:<30} {count}")
    return names


def delete_person() -> None:
    """
    Interactively remove one person from the database.
    Asks for a name, confirms, deletes their image folder and JSON entry,
    then retrains the model (or removes it if the DB is now empty).
    """
    names = list_people()
    if not names:
        return

    name_input = input("\n  Name to delete (ENTER = cancel): ").strip().lower()
    if not name_input:
        print("  Cancelled.")
        return

    uid = next((k for k, v in names.items() if v.lower() == name_input), None)
    if not uid:
        print(f"  [!] '{name_input}' not found.")
        return

    if input(f"  Delete '{names[uid]}'? (y/n): ").strip().lower() != 'y':
        print("  Cancelled.")
        return

    deleted = names.pop(uid)
    _save_names(names)
    shutil.rmtree(_person_folder(uid), ignore_errors=True)

    # Retrain with remaining people, or remove the model if DB is now empty
    if names:
        train_model(names)
    elif os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    print(f"  [OK] '{deleted}' deleted.")


def delete_all() -> None:
    """
    Wipe the entire database (all images, names JSON, and trained model).
    Requires explicit y/n confirmation to prevent accidents.
    """
    if input("\n  Delete EVERYTHING? (y/n): ").strip().lower() != 'y':
        print("  Cancelled.")
        return

    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    for f in (NAMES_FILE, MODEL_FILE):
        if os.path.exists(f):
            os.remove(f)

    print("  [OK] Database wiped.")


#  MANAGEMENT MENU
#  Shown at startup and when the user presses M during live recognition.

def management_menu() -> bool:
    """
    Display the main text menu and handle the user's choice.

    Returns
    -------
    True  — user chose "Start camera".
    False — user chose "Quit".
    """
    options = {
        "1": "Start camera",
        "2": "List registered people",
        "3": "Delete a person",
        "4": "Delete everything",
        "0": "Quit",
    }
    while True:
        print("\n" + "=" * 46)
        print("   FACE RECOGNITION SYSTEM")
        print("=" * 46)
        for k, v in options.items():
            print(f"  {k}  ->  {v}")
        print("-" * 46)

        choice = input("  Option: ").strip()

        if   choice == "1": return True
        elif choice == "2": list_people()
        elif choice == "3": delete_person()
        elif choice == "4": delete_all()
        elif choice == "0": return False
        else: print("  [!] Invalid option.")



#  MAIN RECOGNITION LOOP

def main() -> None:
    # Show the menu first; exit immediately if the user chooses Quit.
    if not management_menu():
        return

    print("\nStarting camera...")
    cap = cv2.VideoCapture(0)   # 0 = default system webcam
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    names = load_names()

    # One-element list so the registration thread can swap in a newly trained
    # model and the main loop picks it up on the very next frame, with no
    # globals and no extra locking needed.
    recogniser_ref = [load_or_train_model(names)]

    # Cooldown tracking — prevents the unknown-face alert from firing every frame
    alert_active = False
    alert_since  = 0.0

    # Launch the registration subsystem as a background daemon thread.
    # Daemon = automatically killed if the main thread exits unexpectedly.
    t_reg = threading.Thread(
        target=registration_thread,
        args=(cap, names, recogniser_ref),
        daemon=True,
    )
    t_reg.start()

    print("System active.  Q = quit  |  M = menu\n")

    # ── Per-frame detection and recognition loop ────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        # Share the raw frame with the registration thread so it can keep
        # the "camera pending" preview refreshed during a prompt.
        set_frame(frame)

        # Convert to grayscale; equalise histogram to improve detection under
        # uneven lighting (backlight, dim rooms, etc.).
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Detect frontal faces.
        # scaleFactor=1.3  — pyramid downscale step between detection passes.
        # minNeighbors=5   — minimum overlapping rectangles to confirm a face
        #                    (higher = fewer false positives, may miss small faces).
        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(FACE_SIZE_MIN, FACE_SIZE_MIN),
        )

        # Snapshot the recogniser — the registration thread may replace it
        # between frames; reading once per frame keeps this iteration consistent.
        recogniser = recogniser_ref[0]

        for (x, y, w, h) in faces:
            # Crop from the un-equalised frame so prediction matches training
            # image quality; normalise size to match what the model was trained on.
            face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)

            if recogniser is not None:
                # LBPH predict() → (label_id, confidence_distance)
                # Distance 0 = identical; larger = less similar.
                label_id, confidence = recogniser.predict(face_roi)
                recognised = confidence < CONFIDENCE_MAX
            else:
                # No model yet — every face is unknown until someone registers
                recognised = False
                label_id   = -1
                confidence = 999.0

            if recognised:
                label        = names.get(str(label_id), "?")
                color        = (0, 200, 0)    # green
                alert_active = False          # reset so future unknowns alert again
            else:
                label = "UNKNOWN"
                color = (0, 0, 220)           # red

                # Fire the registration alert — but only when:
                #   1. No prompt is already open (registration_busy not set)
                #   2. Enough cooldown time has elapsed since the last alert
                if not registration_busy.is_set():
                    now = time.time()
                    if not alert_active or (now - alert_since) > ALERT_COOLDOWN:
                        alert_active = True
                        alert_since  = now
                        registration_busy.set()
                        event_queue.put({
                            "type": "unknown",
                            "roi":  gray[y:y+h, x:x+w].copy(),
                        })

            # ── Bounding box ──────────────────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

            # ── Label with filled background for contrast ─────────────────────
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # ── Match percentage for recognised faces ─────────────────────────
            # LBPH distance → 0-100% score:
            # distance 0  → 100% match  (perfect)
            # distance 80 → 0% match    (at threshold boundary)
            if recognised:
                cv2.putText(frame, f"{int(100 - confidence)}% match",
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, "Q = quit  |  M = menu",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('m'):
            # Pause the live feed, open the menu, then resume.
            cv2.destroyAllWindows()
            if not management_menu():
                break

            # Reload in case the user added/deleted people from the menu
            names = load_names()
            recogniser_ref[0] = load_or_train_model(names)
            alert_active = False
            print("\nCamera resumed.  Q = quit  |  M = menu\n")

    # ── Clean shutdown ────────────────────────────────────────────────────────
    event_queue.put({"type": "stop"})   # tell the registration thread to exit
    t_reg.join(timeout=2)              # wait for it to finish gracefully
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye.")


if __name__ == "__main__":
    main()
