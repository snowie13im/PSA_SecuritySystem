from __future__ import annotations

"""
theme_song.py
=============
Módulo de theme songs por pessoa.
"""

import os
import json
import threading
import time

# ── Configuração ────────────────────────────────────────────────────────────

AUDIO_DIR      = "audios"
THEME_FILE     = "theme_songs.json"
REQUESTS_FILE  = "music_requests.json"
SUPPORTED_EXT  = (".mp3", ".wav", ".ogg", ".flac")
PLAY_COOLDOWN  = 15

# ── Detecção opcional do pygame ─────────────────────────────────────────────

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception as _e:
    _PYGAME_OK = False
    print(f"[theme_song] pygame não disponível ({_e}). "
          "As theme songs não serão reproduzidas.")

# ── Estado de reprodução ────────────────────────────────────────────────────

_play_lock       = threading.Lock()
_last_played_uid: "int | None" = None
_last_play_time:  float        = 0.0

# ── FIX: In-memory theme cache — avoids re-reading the JSON file every frame.
# play_theme_for() is called on every recognised face at ~30 fps; without
# this cache that means 30 disk reads per second per recognised person.
_themes_cache:      "dict | None" = None
_themes_cache_mtime: float        = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE PERSISTÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def _load_themes() -> dict:
    """
    Return the uid→audio-path mapping.
    Re-reads from disk only when the file has been modified since the last
    read; otherwise returns the in-memory cache (O(1), no I/O).
    """
    global _themes_cache, _themes_cache_mtime

    if not os.path.exists(THEME_FILE):
        return {}

    try:
        mtime = os.path.getmtime(THEME_FILE)
    except OSError:
        return _themes_cache or {}

    # Cache is still fresh — skip the disk read entirely
    if _themes_cache is not None and mtime == _themes_cache_mtime:
        return _themes_cache

    # File changed (or first load) — reload and update cache
    try:
        with open(THEME_FILE, encoding="utf-8") as fh:
            _themes_cache       = json.load(fh)
            _themes_cache_mtime = mtime
    except Exception:
        _themes_cache = _themes_cache or {}

    return _themes_cache


def _save_themes(themes: dict) -> None:
    # FIX: use 'with' so the file handle is always closed, even on error
    with open(THEME_FILE, "w", encoding="utf-8") as fh:
        json.dump(themes, fh, ensure_ascii=False, indent=2)

    # Invalidate cache so the next read picks up the new file
    global _themes_cache, _themes_cache_mtime
    _themes_cache       = themes.copy()
    _themes_cache_mtime = os.path.getmtime(THEME_FILE)


def _load_requests() -> list:
    if not os.path.exists(REQUESTS_FILE):
        return []
    try:
        # FIX: use 'with' for proper file-handle cleanup
        with open(REQUESTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_requests(requests: list) -> None:
    with open(REQUESTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(requests, fh, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  LISTAGEM DE ÁUDIOS DISPONÍVEIS
# ══════════════════════════════════════════════════════════════════════════════

def list_audio_files() -> list[str]:
    """Return sorted list of audio file paths in AUDIO_DIR."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    files= [
        os.path.join(AUDIO_DIR, f)
        for f in sorted(os.listdir(AUDIO_DIR))
        if f.lower().endswith(SUPPORTED_EXT)
    ]
    return files


# ══════════════════════════════════════════════════════════════════════════════
#  JANELA OPENVC DE SELECÇÃO DE MÚSICA
# ══════════════════════════════════════════════════════════════════════════════

def _ask_theme_in_window(person_name: str) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    WIN = "Theme Song"
    W, H   = 480, 380
    BG     = (20,  20,  20)
    CYAN   = (0,  200, 255)
    WHITE  = (255, 255, 255)
    GREY   = (160, 160, 160)
    GREEN  = (0,  220,   0)
    YELLOW = (0,  220, 220)

    audio_files = list_audio_files()

    # Drain any keys that were buffered during the registration flow
    # (e.g. the SPACE from the last photo capture) so they don't
    # accidentally dismiss this window before the user sees it.
    for _ in range(5):
        cv2.waitKey(1)

    # Phase 1: does the user want a theme song?
    while True:
        panel = np.full((H, W, 3), BG, dtype=np.uint8)
        cv2.putText(panel, f"Theme song para {person_name}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        cv2.line(panel, (20, 48), (W - 20, 48), GREY, 1)
        cv2.putText(panel, "Queres associar uma musica?",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)
        cv2.putText(panel, "Y = Sim    N = Nao    ESC = cancelar",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)
        cv2.imshow(WIN, panel)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('n'), 27):   # só N ou ESC cancelam — ENTER já não cancela
            cv2.destroyWindow(WIN)
            return None
        if key == ord('y'):
            break
        if key == 13:               # ENTER sem escolha → sem música
            cv2.destroyWindow(WIN)
            return None

    # Phase 2: select a song or make a request
    if not audio_files:
        return _ask_music_request_in_window(WIN, person_name, W, H,
                                            BG, CYAN, WHITE, GREY, YELLOW)

    typed       = ""
    selected    = None
    MAX_VISIBLE = min(len(audio_files), 8)

    while selected is None:
        panel = np.full((H, W, 3), BG, dtype=np.uint8)

        cv2.putText(panel, "Escolhe uma musica (numero + ENTER):",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)

        for idx, path in enumerate(audio_files[:MAX_VISIBLE]):
            label = f"{idx + 1}. {os.path.basename(path)}"
            if len(label) > 52:
                label = label[:49] + "..."
            color = GREEN if str(idx + 1) == typed else WHITE
            cv2.putText(panel, label,
                        (20, 60 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        if len(audio_files) > MAX_VISIBLE:
            cv2.putText(panel,
                        f"  (+{len(audio_files) - MAX_VISIBLE} mais nao mostradas)",
                        (20, 60 + MAX_VISIBLE * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)

        input_y = H - 100
        cv2.line(panel, (20, input_y - 10), (W - 20, input_y - 10), GREY, 1)
        cv2.putText(panel, "Numero: " + typed + "|",
                    (20, input_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.putText(panel, "R = pedir musica ao admin  |  ESC = cancelar",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(WIN)
            return None
        elif key == ord('r'):
            return _ask_music_request_in_window(WIN, person_name, W, H,
                                                BG, CYAN, WHITE, GREY, YELLOW)
        elif key == 8:
            typed = typed[:-1]
        elif 48 <= key <= 57:
            typed += chr(key)
        elif key == 13:
            try:
                choice = int(typed)
                if 1 <= choice <= len(audio_files):
                    selected = audio_files[choice - 1]
                else:
                    typed = ""
            except ValueError:
                typed = ""

    cv2.destroyWindow(WIN)
    return selected


def _ask_music_request_in_window(WIN, person_name, W, H,
                                  BG, CYAN, WHITE, GREY, YELLOW) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    typed = ""
    while True:
        panel = np.full((H, W, 3), BG, dtype=np.uint8)

        cv2.putText(panel, "Pedido de musica para o admin",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)
        cv2.line(panel, (20, 50), (W - 20, 50), GREY, 1)
        cv2.putText(panel, "Escreve o nome da musica / artista:",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        cv2.rectangle(panel, (16, 110), (W - 16, 150), (50, 50, 50), -1)
        cv2.rectangle(panel, (16, 110), (W - 16, 150), YELLOW, 1)

        display_text = typed[-45:] if len(typed) > 45 else typed
        cv2.putText(panel, display_text + "|",
                    (22, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.putText(panel, "ENTER = enviar pedido  |  ESC = cancelar",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(WIN)
            return None
        elif key == 8:
            typed = typed[:-1]
        elif key == 13:
            cv2.destroyWindow(WIN)
            return f"REQUEST:{typed.strip()}" if typed.strip() else None
        elif 32 <= key <= 126:
            typed += chr(key)


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def ask_and_save_theme(uid: int, person_name: str) -> None:
    result = _ask_theme_in_window(person_name)

    if result is None:
        print("  [theme] Sem theme song.")
        return

    if result.startswith("REQUEST:"):
        request_text = result[len("REQUEST:"):]
        requests = _load_requests()
        requests.append({
            "uid":     uid,
            "name":    person_name,
            "request": request_text,
            "done":    False,
        })
        _save_requests(requests)
        print(f"  [theme] Pedido guardado para o admin: '{request_text}'")
        return

    themes = _load_themes()
    themes[str(uid)] = result
    _save_themes(themes)
    print(f"  [theme] Theme song de '{person_name}' → {result}")


def play_theme_for(uid: int) -> None:
    """
    Play the theme for *uid* in a background thread, respecting PLAY_COOLDOWN.
    Uses the in-memory cache so there is no disk I/O on the hot recognition path.
    """
    if not _PYGAME_OK:
        return

    themes = _load_themes()          # O(1) cache hit in the common case
    path   = themes.get(str(uid))
    if not path or not os.path.exists(path):
        return

    with _play_lock:
        global _last_played_uid, _last_play_time
        now = time.time()
        if (now - _last_play_time) < PLAY_COOLDOWN:
            return
        _last_played_uid = uid
        _last_play_time  = now

    threading.Thread(target=_play_audio, args=(path,), daemon=True).start()


def _play_audio(path: str) -> None:
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[theme_song] Erro ao reproduzir '{path}': {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  GESTÃO DE PEDIDOS — MENU DO ADMIN
# ══════════════════════════════════════════════════════════════════════════════

def list_music_requests() -> None:
    requests = _load_requests()
    pending  = [r for r in requests if not r.get("done")]

    if not pending:
        print("\n  (sem pedidos de música pendentes)")
        return

    print(f"\n  {'#':<4} {'Nome':<25} Pedido")
    print("  " + "-" * 60)
    for i, r in enumerate(pending):
        print(f"  {i:<4} {r['name']:<25} {r['request']}")


def mark_request_done() -> None:
    requests = _load_requests()
    pending  = [(i, r) for i, r in enumerate(requests) if not r.get("done")]

    if not pending:
        print("\n  (sem pedidos pendentes)")
        return

    list_music_requests()
    raw = input("\n  Numero do pedido a marcar como resolvido (ENTER = cancelar): ").strip()
    if not raw:
        print("  Cancelado.")
        return

    try:
        choice = int(raw)
        orig_idx, _ = pending[choice]
        requests[orig_idx]["done"] = True
        _save_requests(requests)
        print("  [OK] Pedido marcado como resolvido.")
    except (ValueError, IndexError):
        print("  [!] Número inválido.")
