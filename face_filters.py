"""
face_filters.py
===============
Filtros de rosto em tempo real usando MediaPipe Face Mesh (468 landmarks).

Controlos:
  F        → próximo filtro
  S        → filtro anterior  
  1–7      → saltar para filtro direto
  Q / ESC  → sair

Filtros disponíveis:
  1. Sem filtro       — câmara limpa
  2. Óculos           — overlay de óculos alinhado com os olhos
  3. Chapéu           — overlay de chapéu no topo da cabeça
  4. Nariz de palhaço — círculo vermelho no nariz
  5. Maquilhagem      — blush nas maçãs do rosto + batom
  6. Distorção        — olhos e boca esticados
  7. Pixelizar rosto  — rosto pixelizado (efeito censura)

Para adicionares as tuas próprias imagens PNG (com canal alpha):
  - Óculos  → coloca em  overlays/glasses.png
  - Chapéu  → coloca em  overlays/hat.png
  Sem esses ficheiros, o código desenha versões simples em OpenCV.
"""

from __future__ import annotations

import os
import sys
import math
import time

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe setup ──────────────────────────────────────────────────────────

mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 468-landmark model; refine_landmarks adds iris points (468–477)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Landmark index constants ──────────────────────────────────────────────────
# Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Eyes
L_EYE_OUTER   = 33
L_EYE_INNER   = 133
R_EYE_OUTER   = 362
R_EYE_INNER   = 263
L_EYE_TOP     = 159
L_EYE_BOT     = 145
R_EYE_TOP     = 386
R_EYE_BOT     = 374

# Iris centres (refine_landmarks required)
L_IRIS        = 468
R_IRIS        = 473

# Nose
NOSE_TIP      = 4
NOSE_BRIDGE   = 6

# Mouth
MOUTH_LEFT    = 61
MOUTH_RIGHT   = 291
MOUTH_TOP     = 13
MOUTH_BOT     = 14

# Forehead / chin approximations
FOREHEAD      = 10
CHIN          = 152

# Cheeks (for blush)
L_CHEEK       = 234
R_CHEEK       = 454


# ── Overlay image loader ──────────────────────────────────────────────────────

def _load_overlay(path: str) -> np.ndarray | None:
    """Load a PNG with alpha channel; return None if missing."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        return None
    return img


def _place_overlay(frame: np.ndarray, overlay: np.ndarray,
                   cx: int, cy: int, w: int, h: int,
                   angle: float = 0.0) -> np.ndarray:
    """
    Paste *overlay* (BGRA) onto *frame* centred at (cx, cy) with size (w, h).
    Supports rotation via *angle* (degrees).
    """
    if w <= 0 or h <= 0:
        return frame

    resized = cv2.resize(overlay, (abs(w), abs(h)))

    if angle != 0.0:
        M       = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
        resized = cv2.warpAffine(resized, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Clip to frame boundaries
    fx1 = max(x1, 0);  fy1 = max(y1, 0)
    fx2 = min(x2, frame.shape[1]);  fy2 = min(y2, frame.shape[0])
    ox1 = fx1 - x1;    oy1 = fy1 - y1
    ox2 = ox1 + (fx2 - fx1);       oy2 = oy1 + (fy2 - fy1)

    if fx2 <= fx1 or fy2 <= fy1:
        return frame

    roi   = frame[fy1:fy2, fx1:fx2]
    patch = resized[oy1:oy2, ox1:ox2]

    bgr   = patch[:, :, :3]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0

    frame[fy1:fy2, fx1:fx2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    return frame


# ── Landmark helpers ──────────────────────────────────────────────────────────

def _lm(landmarks, idx: int, W: int, H: int) -> tuple[int, int]:
    """Return pixel (x, y) for a landmark index."""
    p = landmarks[idx]
    return int(p.x * W), int(p.y * H)


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a: tuple, b: tuple) -> float:
    """Angle of the line from a to b, in degrees."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# ── Overlay assets (loaded once) ──────────────────────────────────────────────

os.makedirs("overlays", exist_ok=True)
_OVL_GLASSES = _load_overlay("overlays/glasses.png")
_OVL_HAT     = _load_overlay("overlays/hat.png")
_OVL_BEARD    = _load_overlay("overlays/beard.png")


# ── Fallback drawing functions (used when PNG assets are absent) ──────────────

def _draw_glasses_cv(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Draw simple rectangular glasses with OpenCV shapes."""
    lo = _lm(lm_list, L_EYE_OUTER, W, H)
    li = _lm(lm_list, L_EYE_INNER, W, H)
    ri = _lm(lm_list, R_EYE_INNER, W, H)
    ro = _lm(lm_list, R_EYE_OUTER, W, H)

    pad  = int(_dist(lo, li) * 0.25)
    lt   = _lm(lm_list, L_EYE_TOP, W, H)[1] - pad
    lb   = _lm(lm_list, L_EYE_BOT, W, H)[1] + pad
    rt   = _lm(lm_list, R_EYE_TOP, W, H)[1] - pad
    rb   = _lm(lm_list, R_EYE_BOT, W, H)[1] + pad

    col  = (20, 20, 20)
    thk  = 2

    # Left lens
    cv2.rectangle(frame, (lo[0] - pad, lt), (li[0] + pad, lb), col, thk)
    # Right lens
    cv2.rectangle(frame, (ri[0] - pad, rt), (ro[0] + pad, rb), col, thk)
    # Bridge
    cv2.line(frame, (li[0] + pad, (lt + lb) // 2),
                    (ri[0] - pad, (rt + rb) // 2), col, thk)
    # Arms (temples)
    cv2.line(frame, (lo[0] - pad, (lt + lb) // 2),
                    (lo[0] - pad * 5, (lt + lb) // 2), col, thk)
    cv2.line(frame, (ro[0] + pad, (rt + rb) // 2),
                    (ro[0] + pad * 5, (rt + rb) // 2), col, thk)
    return frame


def _draw_hat_cv(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Draw a simple top hat above the forehead."""
    fh    = _lm(lm_list, FOREHEAD,   W, H)
    lo    = _lm(lm_list, L_EYE_OUTER, W, H)
    ro    = _lm(lm_list, R_EYE_OUTER, W, H)

    face_w = int(_dist(lo, ro) * 1.6)
    brim_h = max(int(face_w * 0.08), 6)
    top_h  = int(face_w * 0.55)
    top_w  = int(face_w * 0.75)

    cx     = fh[0]
    brim_y = fh[1] - int(face_w * 0.05)

    col    = (30, 30, 30)

    # Brim
    cv2.rectangle(frame,
                  (cx - face_w // 2, brim_y - brim_h),
                  (cx + face_w // 2, brim_y),
                  col, -1)
    # Top
    cv2.rectangle(frame,
                  (cx - top_w // 2, brim_y - brim_h - top_h),
                  (cx + top_w // 2, brim_y - brim_h),
                  col, -1)
    # Hat band
    band_col = (0, 60, 180)
    cv2.rectangle(frame,
                  (cx - top_w // 2, brim_y - brim_h - int(top_h * 0.2)),
                  (cx + top_w // 2, brim_y - brim_h),
                  band_col, -1)
    return frame


# ── FILTROS ───────────────────────────────────────────────────────────────────

def filter_none(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Sem filtro — câmara limpa."""
    return frame


def filter_glasses(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Óculos — PNG com alpha ou fallback OpenCV."""
    if _OVL_GLASSES is not None:
        lo    = _lm(lm_list, L_EYE_OUTER, W, H)
        ro    = _lm(lm_list, R_EYE_OUTER, W, H)
        w     = int(_dist(lo, ro) * 1.5)
        h     = w // 3
        cx    = (lo[0] + ro[0]) // 2
        cy    = (lo[1] + ro[1]) // 2
        angle = _angle_deg(lo, ro)
        _place_overlay(frame, _OVL_GLASSES, cx, cy, w, h, angle)
    else:
        _draw_glasses_cv(frame, lm_list, W, H)
    return frame


def filter_hat(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Chapéu — PNG com alpha ou fallback OpenCV."""
    if _OVL_HAT is not None:
        fh    = _lm(lm_list, FOREHEAD,    W, H)
        lo    = _lm(lm_list, L_EYE_OUTER, W, H)
        ro    = _lm(lm_list, R_EYE_OUTER, W, H)
        w     = int(_dist(lo, ro) * 1.8)
        h     = int(w * 1.1)
        cx    = fh[0]
        cy    = fh[1] - h // 2
        angle = _angle_deg(lo, ro)
        _place_overlay(frame, _OVL_HAT, cx, cy, w, h, angle)
    else:
        _draw_hat_cv(frame, lm_list, W, H)
    return frame


def filter_clown_nose(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Nariz vermelho de palhaço."""
    nose = _lm(lm_list, NOSE_TIP, W, H)
    lo   = _lm(lm_list, L_EYE_OUTER, W, H)
    ro   = _lm(lm_list, R_EYE_OUTER, W, H)
    r    = max(int(_dist(lo, ro) * 0.12), 8)

    overlay = frame.copy()
    cv2.circle(overlay, nose, r, (0, 0, 220), -1)
    cv2.circle(overlay, (nose[0] - r // 4, nose[1] - r // 4),
               r // 4, (80, 80, 255), -1)   # specular highlight
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def filter_makeup(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Blush nas maçãs do rosto + batom vermelho."""
    H_img, W_img = frame.shape[:2]
    overlay = frame.copy()

    lo   = _lm(lm_list, L_EYE_OUTER, W, H)
    ro   = _lm(lm_list, R_EYE_OUTER, W, H)
    face_w = _dist(lo, ro)

    # Blush — elipse rosa em cada bochecha
    lc = _lm(lm_list, L_CHEEK, W, H)
    rc = _lm(lm_list, R_CHEEK, W, H)
    br = (int(face_w * 0.22), int(face_w * 0.13))   # (rx, ry)

    cv2.ellipse(overlay, lc, br, 0, 0, 360, (100, 100, 255), -1)
    cv2.ellipse(overlay, rc, br, 0, 0, 360, (100, 100, 255), -1)

    # Batom — polígono a cobrir os lábios
    lip_pts = [MOUTH_LEFT, 40, 37, MOUTH_TOP, 267, 270, MOUTH_RIGHT,
               321, 314, MOUTH_BOT, 84, 91]
    pts = np.array([_lm(lm_list, i, W, H) for i in lip_pts], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (30, 30, 200))

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    return frame


def filter_distort(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """
    Distorção de rosto — olhos e boca esticados com remap de malha.
    Usa cv2.remap com um campo de deslocamento calculado a partir dos landmarks.
    """
    H_img, W_img = frame.shape[:2]

    # Pontos de controlo: origem → destino
    lo   = _lm(lm_list, L_EYE_OUTER, W, H)
    ro   = _lm(lm_list, R_EYE_OUTER, W, H)
    face_w = _dist(lo, ro)
    strength = face_w * 0.18

    # Centros de distorção: olho esquerdo, olho direito, boca
    centres = [
        _lm(lm_list, L_IRIS if len(lm_list) > L_IRIS else L_EYE_INNER, W, H),
        _lm(lm_list, R_IRIS if len(lm_list) > R_IRIS else R_EYE_INNER, W, H),
        ((_lm(lm_list, MOUTH_LEFT,  W, H)[0] + _lm(lm_list, MOUTH_RIGHT, W, H)[0]) // 2,
         (_lm(lm_list, MOUTH_TOP,   W, H)[1] + _lm(lm_list, MOUTH_BOT,   W, H)[1]) // 2),
    ]
    radii = [face_w * 0.28, face_w * 0.28, face_w * 0.22]

    # Build identity map
    map_x = np.tile(np.arange(W_img, dtype=np.float32), (H_img, 1))
    map_y = np.repeat(np.arange(H_img, dtype=np.float32)[:, None], W_img, axis=1)

    for (cx, cy), radius in zip(centres, radii):
        dx  = map_x - cx
        dy  = map_y - cy
        dist_sq = dx * dx + dy * dy
        mask = dist_sq < (radius ** 2)
        # Bulge: pixels inside the radius are pulled toward the centre
        factor = np.where(mask, 1.0 - (1.0 - dist_sq / (radius ** 2)) * 0.55, 1.0)
        map_x  = np.where(mask, cx + dx * factor, map_x)
        map_y  = np.where(mask, cy + dy * factor, map_y)

    distorted = cv2.remap(frame, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    return distorted


def filter_pixelate(frame: np.ndarray, lm_list, W: int, H: int) -> np.ndarray:
    """Pixeliza a região do rosto (efeito censura)."""
    lo  = _lm(lm_list, L_EYE_OUTER, W, H)
    ro  = _lm(lm_list, R_EYE_OUTER, W, H)
    fh  = _lm(lm_list, FOREHEAD,    W, H)
    ch  = _lm(lm_list, CHIN,        W, H)

    face_w = int(_dist(lo, ro) * 1.4)
    face_h = int(_dist(fh, ch) * 1.2)
    cx     = (lo[0] + ro[0]) // 2
    cy     = (fh[1] + ch[1]) // 2

    x1 = max(cx - face_w // 2, 0)
    y1 = max(cy - face_h // 2, 0)
    x2 = min(cx + face_w // 2, W)
    y2 = min(cy + face_h // 2, H)

    if x2 <= x1 or y2 <= y1:
        return frame

    block  = max(face_w // 14, 4)
    roi    = frame[y1:y2, x1:x2]
    small  = cv2.resize(roi, (max((x2 - x1) // block, 1),
                               max((y2 - y1) // block, 1)),
                        interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1),
                           interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated
    return frame


# ── Filter registry ───────────────────────────────────────────────────────────

FILTERS = [
    ("Sem filtro",        filter_none),
    ("Oculos",            filter_glasses),
    ("Chapeu",            filter_hat),
    ("Nariz de Palhaço",  filter_clown_nose),
    ("Maquilhagem",       filter_makeup),
    ("Distorcao",         filter_distort),
    ("Pixelizar",         filter_pixelate),
]


# ── HUD helpers ───────────────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, filter_idx: int, fps: float) -> None:
    H, W = frame.shape[:2]
    name = FILTERS[filter_idx][0]

    # Bottom bar
    cv2.rectangle(frame, (0, H - 38), (W, H), (20, 20, 20), -1)
    cv2.putText(frame, f"Filtro {filter_idx + 1}/{len(FILTERS)}: {name}",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame, f"F/S = trocar  |  1-{len(FILTERS)} = direto  |  Q = sair",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # FPS top-right
    cv2.putText(frame, f"{fps:.0f} fps",
                (W - 70, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Filter name top-left
    cv2.putText(frame, f"  F/S = trocar  1-{len(FILTERS)} = direto  Q = sair",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # Big filter label at top
    cv2.rectangle(frame, (0, 0), (W, 36), (20, 20, 20), -1)
    cv2.putText(frame, f"[{filter_idx + 1}] {name}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, f"{fps:.0f} fps",
                (W - 72, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("A abrir câmara...")

    cap = None
    for idx in range(4):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                cap = c
                print(f"  [OK] Câmara no índice {idx}.")
                break
        if cap:
            break

    if cap is None:
        print("[ERRO] Não foi possível abrir a câmara.")
        sys.exit(1)

    filter_idx  = 0
    prev_time   = time.time()

    print(f"\nControlos:")
    print(f"  F / S  → próximo / filtro anterior")
    print(f"  1-{len(FILTERS)}    → saltar para filtro")
    print(f"  Q / ESC → sair\n")

    WIN = "Face Filters"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha na leitura da câmara.")
            break

        H, W = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                lm_list = face_lms.landmark

                try:
                    frame = FILTERS[filter_idx][1](frame, lm_list, W, H)
                except Exception as e:
                    # Nunca deixar um erro de filtro matar o loop
                    cv2.putText(frame, f"Erro no filtro: {e}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)

        # FPS
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        _draw_hud(frame, filter_idx, fps)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            filter_idx = (filter_idx + 1) % len(FILTERS)
            print(f"  → Filtro: {FILTERS[filter_idx][0]}")
        elif key == ord('s'):
            filter_idx = (filter_idx - 1) % len(FILTERS)
            print(f"  → Filtro: {FILTERS[filter_idx][0]}")
        elif ord('1') <= key <= ord('0') + len(FILTERS):
            filter_idx = key - ord('1')
            print(f"  → Filtro: {FILTERS[filter_idx][0]}")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Adeus!")


if __name__ == "__main__":
    main()
