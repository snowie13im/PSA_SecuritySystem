import cv2
import os
import json
import shutil
import time
import threading
import numpy as np
import pyrealsense2 as rs

from config import (
    DATA_DIR, NAMES_FILE, MODEL_FILE,
    IMG_SIZE, CONFIDENCE_MAX, ALERT_COOLDOWN, FACE_SIZE_MIN,
    face_cascade,
)
from registration import (
    set_frame, event_queue, registration_busy,
    registration_thread, train_model,
)

# ── Configuração RealSense ─────────────────────────────────────────────────

# Distância máxima para considerar um rosto (metros)
MAX_FACE_DEPTH = 2.0

# Desvio padrão mínimo do mapa de profundidade na ROI do rosto.
# Uma fotografia impressa tem profundidade quase constante (~0.005),
# um rosto real tem variação natural (~0.02+).
ANTISPOOFING_STD_MIN = 0.015


def _init_realsense():
    """
    Configura e arranca o pipeline RealSense para a L515.
    Activa o stream RGB (1280x720 @ 30fps) e o stream de profundidade (640x480 @ 30fps).
    Devolve (pipeline, align) onde align sincroniza os dois streams.
    """
    pipeline = rs.pipeline()
    config   = rs.config()

    # Stream RGB — frames de cor para deteção e reconhecimento
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Stream de profundidade — LiDAR da L515 (Z16 = 16-bit, milímetros)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    # align projeta o mapa de profundidade para o mesmo plano do RGB,
    # para que depth_frame[y][x] corresponda ao mesmo pixel do color_frame[y][x].
    align = rs.align(rs.stream.color)

    return pipeline, align


def _get_face_depth(depth_frame_aligned, x, y, w, h):
    """
    Calcula a profundidade mediana (metros) e o desvio padrão da ROI do rosto.
    Usa a mediana em vez da média para ignorar pixels sem retorno LiDAR (valor 0).

    Devolve (mediana_metros, std_metros).
    """
    # Converter depth_frame para array numpy (valores em milímetros)
    depth_image = np.asanyarray(depth_frame_aligned.get_data()).astype(np.float32)
    depth_image *= 0.00025   # depth scale da L515 (metros por unidade)

    roi = depth_image[y:y+h, x:x+w]

    # Ignorar pixels sem retorno (valor 0 = sem medição)
    valid = roi[roi > 0]
    if len(valid) == 0:
        return None, None

    return float(np.median(valid)), float(np.std(valid))


# ── Persistência ───────────────────────────────────────────────────────────

def load_names():
    if os.path.exists(NAMES_FILE):
        return json.load(open(NAMES_FILE, encoding="utf-8"))
    return {}

def _save_names(names):
    json.dump(names, open(NAMES_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

def _person_folder(uid):
    return os.path.join(DATA_DIR, str(uid))

def load_or_train_model(names):
    if os.path.exists(MODEL_FILE) and names:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(MODEL_FILE)
        return rec
    return train_model(names)


# ── Gestão da base de dados ────────────────────────────────────────────────

def list_people():
    names = load_names()
    if not names:
        print("\n  (nenhuma pessoa registada)")
        return names
    print(f"\n  {'ID':<6} {'Nome':<30} Fotos")
    print("  " + "-" * 44)
    for uid_str, name in names.items():
        folder = _person_folder(uid_str)
        count  = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        print(f"  {uid_str:<6} {name:<30} {count}")
    return names

def delete_person():
    names = list_people()
    if not names:
        return
    name_input = input("\n  Nome a apagar (ENTER = cancelar): ").strip().lower()
    if not name_input:
        print("  Cancelado.")
        return
    uid = next((k for k, v in names.items() if v.lower() == name_input), None)
    if not uid:
        print(f"  [!] '{name_input}' não encontrado.")
        return
    if input(f"  Apagar '{names[uid]}'? (s/n): ").strip().lower() != 's':
        print("  Cancelado.")
        return
    deleted = names.pop(uid)
    _save_names(names)
    shutil.rmtree(_person_folder(uid), ignore_errors=True)
    if names:
        train_model(names)
    elif os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    print(f"  [OK] '{deleted}' apagado.")

def delete_all():
    if input("\n  Apagar TUDO? (s/n): ").strip().lower() != 's':
        print("  Cancelado.")
        return
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in (NAMES_FILE, MODEL_FILE):
        if os.path.exists(f):
            os.remove(f)
    print("  [OK] Base de dados apagada.")


# ── Menu ───────────────────────────────────────────────────────────────────

def management_menu():
    options = {
        "1": "Iniciar câmara",
        "2": "Listar pessoas",
        "3": "Apagar pessoa",
        "4": "Apagar tudo",
        "0": "Sair",
    }
    while True:
        print("\n" + "=" * 46)
        print("   SISTEMA DE RECONHECIMENTO FACIAL  [L515]")
        print("=" * 46)
        for k, v in options.items():
            print(f"  {k}  ->  {v}")
        print("-" * 46)
        choice = input("  Opção: ").strip()
        if   choice == "1": return True
        elif choice == "2": list_people()
        elif choice == "3": delete_person()
        elif choice == "4": delete_all()
        elif choice == "0": return False
        else: print("  [!] Opção inválida.")


# ── Loop principal ─────────────────────────────────────────────────────────

def main():
    if not management_menu():
        return

    print("\nA iniciar câmara RealSense L515...")
    try:
        pipeline, align = _init_realsense()
    except Exception as e:
        print(f"[ERRO] Não foi possível iniciar a L515: {e}")
        return

    # Criar um objecto cap-like para a thread de registo.
    # A thread usa cap.read() — aqui criamos um wrapper simples que
    # lê do pipeline RealSense e devolve apenas o frame RGB, tal como
    # cv2.VideoCapture.read() faria.
    class RealSenseCap:
        def read(self_inner):
            try:
                frames        = pipeline.wait_for_frames(timeout_ms=5000)
                aligned       = align.process(frames)
                color_frame   = aligned.get_color_frame()
                if not color_frame:
                    return False, None
                return True, np.asanyarray(color_frame.get_data())
            except Exception:
                return False, None

    cap = RealSenseCap()

    names          = load_names()
    recogniser_ref = [load_or_train_model(names)]
    alert_active   = False
    alert_since    = 0.0

    t_reg = threading.Thread(
        target=registration_thread,
        args=(cap, names, recogniser_ref),
        daemon=True,
    )
    t_reg.start()

    print("Sistema ativo.  Q = sair  |  M = menu\n")

    while True:
        # Ler frame RGB + profundidade alinhada
        try:
            frames        = pipeline.wait_for_frames(timeout_ms=5000)
            aligned       = align.process(frames)
            color_frame   = aligned.get_color_frame()
            depth_frame   = aligned.get_depth_frame()
        except Exception:
            print("[ERRO] Falha na leitura da câmara.")
            break

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        set_frame(frame)

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.3, minNeighbors=5,
            minSize=(FACE_SIZE_MIN, FACE_SIZE_MIN),
        )

        recogniser = recogniser_ref[0]

        for (x, y, w, h) in faces:

            # ── Filtro de profundidade ──────────────────────────────────────
            depth_m, depth_std = _get_face_depth(depth_frame, x, y, w, h)

            if depth_m is None:
                # Sem retorno LiDAR — ignorar este rosto
                continue

            if depth_m > MAX_FACE_DEPTH:
                # Rosto demasiado longe
                cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
                cv2.putText(frame, f"{depth_m:.1f}m — longe demais",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
                continue

            # ── Anti-spoofing por profundidade ──────────────────────────────
            # Uma fotografia impressa tem profundidade quase constante (std baixo).
            # Um rosto real tem variação natural na superfície (std alto).
            if depth_std < ANTISPOOFING_STD_MIN:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 100, 255), 2)
                cv2.putText(frame, "SPOOFING DETETADO",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                continue

            # ── Reconhecimento LBPH (igual ao original) ─────────────────────
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
            else:
                label = "unknown"
                color = (0, 0, 220)
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

            # ── Desenho ─────────────────────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Profundidade + match % abaixo da caixa
            info = f"{depth_m:.2f}m"
            if recognised:
                info += f"  {int(100 - confidence)}% match"
            cv2.putText(frame, info, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, "Q = sair  |  M = menu",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Reconhecimento Facial [L515]", frame)

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
            print("\nCâmara retomada.  Q = sair  |  M = menu\n")

    # Shutdown
    event_queue.put({"type": "stop"})
    t_reg.join(timeout=2)
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Até logo.")


if __name__ == "__main__":
    main()