import cv2
import os
import json
import shutil
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "psahorus-facerec")
mqtt_client.connect("localhost", 1883)
mqtt_client.loop_start()

from config import (
    DATA_DIR, NAMES_FILE, MODEL_FILE,
    IMG_SIZE, CONFIDENCE_MAX, ALERT_COOLDOWN, FACE_SIZE_MIN,
    face_cascade,
)
from registration import (
    set_frame,
    event_queue,
    registration_busy,
    registration_thread,
    train_model,
)
from theme_song import play_theme_for, list_music_requests, mark_request_done

# ── Camera driver ──────────────────────────────────────────────────────────
# Change CAMERA_BACKEND to "usb" (or "usb:1", "usb:2" …) to use a regular
# webcam instead of the RealSense.  All depth-dependent features (distance
# gate, anti-spoofing) are silently skipped when depth is unavailable.
#
# Examples:
#   CAMERA_BACKEND = "realsense"
#   CAMERA_BACKEND = "usb"
#   CAMERA_BACKEND = "usb:1"
CAMERA_BACKEND = "usb:1"

from camera import make_driver, get_face_depth

MAX_FACE_DEPTH      = 2.0    # metres — faces farther away are ignored
ANTISPOOFING_STD_MIN = 0.015  # faces with flat depth are flagged as spoofed


# ── Persistence helpers ────────────────────────────────────────────────────

def load_names() -> dict:
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_names(names: dict) -> None:
    with open(NAMES_FILE, "w", encoding="utf-8") as fh:
        json.dump(names, fh, ensure_ascii=False, indent=2)


def _person_folder(uid) -> str:
    return os.path.join(DATA_DIR, str(uid))


def load_or_train_model(names: dict):
    if os.path.exists(MODEL_FILE) and names:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(MODEL_FILE)
        return rec
    return train_model(names)


# ── Database management helpers ────────────────────────────────────────────

def list_people() -> dict:
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

    if names:
        train_model(names)
    elif os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    print(f"  [OK] '{deleted}' deleted.")


def delete_all() -> None:
    if input("\n  Delete EVERYTHING? (y/n): ").strip().lower() != 'y':
        print("  Cancelled.")
        return

    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    for f in (NAMES_FILE, MODEL_FILE):
        if os.path.exists(f):
            os.remove(f)

    print("  [OK] Database wiped.")


# ── Management menu ────────────────────────────────────────────────────────

def management_menu() -> bool:
    options = {
        "1": "Start camera",
        "2": "List registered people",
        "3": "Delete a person",
        "4": "Delete everything",
        "5": "Check Music requests",
        "6": "Mark music request as done",
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
        elif choice == "5": list_music_requests()
        elif choice == "6": mark_request_done()
        elif choice == "0": return False
        else: print("  [!] Invalid option.")


# ── Main recognition loop ──────────────────────────────────────────────────

def main() -> None:
    if not management_menu():
        return

    print(f"\nStarting camera ({CAMERA_BACKEND})…")
    try:
        cam = make_driver(CAMERA_BACKEND)
    except Exception as e:
        print(f"[ERROR] Cannot open camera: {e}")
        return

    names          = load_names()
    recogniser_ref = [load_or_train_model(names)]
    alert_active   = False
    alert_since    = 0.0

    t_reg = threading.Thread(
        target=registration_thread,
        args=(cam, names, recogniser_ref),
        daemon=True,
    )
    t_reg.start()

    print("System active.  Q = quit  |  M = menu\n")

    with cam:
        while True:
            color_bgr, depth_frame = cam.read_frame()

            if color_bgr is None:
                print("[ERROR] Failed to read from camera.")
                break

            frame = color_bgr
            set_frame(frame)

            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(FACE_SIZE_MIN, FACE_SIZE_MIN),
            )

            recogniser = recogniser_ref[0]

            for (x, y, w, h) in faces:

                # ── Depth checks (only when depth is available) ────────────
                if depth_frame is not None:
                    depth_m, depth_std = get_face_depth(depth_frame, x, y, w, h)

                    if depth_m is None:
                        continue  # no LiDAR return for this face

                    if depth_m > MAX_FACE_DEPTH:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
                        cv2.putText(frame, f"{depth_m:.1f}m — TOO FAR",
                                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
                        continue

                    if depth_std < ANTISPOOFING_STD_MIN:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 100, 255), 2)
                        cv2.putText(frame, "SPOOFING DETECTED",
                                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                        continue

                    depth_info = f"{depth_m:.2f}m"
                else:
                    depth_m    = None
                    depth_info = ""

                # ── Recognition ────────────────────────────────────────────
                face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)

                if recogniser is not None:
                    label_id, confidence = recogniser.predict(face_roi)
                    recognised = confidence < CONFIDENCE_MAX
                else:
                    recognised = False
                    label_id   = -1
                    confidence = 999.0

                if recognised:
                    label        = names.get(str(label_id))
                    color        = (0, 200, 0)
                    alert_active = False
                    play_theme_for(label_id)

                    fh, fw = frame.shape[:2]
                    mqtt_client.publish("psahorus/facerec/resultado", json.dumps({
                        "pessoa":    label,
                        "confianca": int(100 - confidence),
                        "acao":      "abrir",
                        "coord":     {"x": int(x + w/2 - fw/2), "y": int(fh/2 - (y + h/2))}
                    }))

                else:
                    label = "UNKNOWN"
                    color = (0, 0, 220)
                    fh, fw = frame.shape[:2]

                    mqtt_client.publish("psahorus/facerec/resultado", json.dumps({
                        "pessoa":    "UNKNOWN",
                        "confianca": 0,
                        "acao":      "negar",
                        "coord":     {"x": int(x + w/2 - fw/2), "y": int(fh/2 - (y + h/2))},
                    }))

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

                # ── Overlay ────────────────────────────────────────────────
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
                cv2.putText(frame, label, (x + 4, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                info = depth_info
                if recognised:
                    info += f"  {int(100 - confidence)}% match" if info else f"{int(100 - confidence)}% match"
                if info:
                    cv2.putText(frame, info, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, "Q = quit  |  M = menu",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('m'):
                cv2.destroyAllWindows()
                if not management_menu():
                    break
                names = load_names()
                recogniser_ref[0] = load_or_train_model(names)
                alert_active = False
                print("\nCamera resumed.  Q = quit  |  M = menu\n")

    # ── Clean shutdown ─────────────────────────────────────────────────────
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    event_queue.put({"type": "stop"})
    t_reg.join(timeout=2)
    cv2.destroyAllWindows()
    print("Goodbye.")


if __name__ == "__main__":
    main()
