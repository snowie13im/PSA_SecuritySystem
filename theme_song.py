from __future__ import annotations

"""
theme_song.py
=============
Módulo de theme songs por pessoa

Responsabilidades
-----------------
- Listar as músicas disponíveis na pasta AUDIO_DIR.
- Perguntar ao utilizador (dentro de uma janela OpenCV) se quer associar
  uma música ao seu registo.
- Guardar / carregar o mapeamento  uid → ficheiro de áudio  num JSON.
- Tocar a tema da pessoa quando ela é reconhecida, sem bloquear o loop
  principal (thread dedicada) e com protecção contra crashes por
  biblioteca em falta ou ficheiro corrompido.
- Registar pedidos de música feitos pelos utilizadores para o admin ver.

Dependências externas opcionais
--------------------------------
pygame   — reprodução de áudio (instalável com  pip install pygame)
Se pygame não estiver instalado o módulo funciona silenciosamente —
todas as funções de reprodução ficam no-ops e é impresso um aviso
único no arranque.

Estrutura de ficheiros
----------------------
audios/                   ← pasta com os ficheiros de áudio (.mp3 / .wav / .ogg)
theme_songs.json          ← { "0": "audios/eye_of_the_tiger.mp3", "3": "..." }
music_requests.json       ← [ {"uid": 2, "name": "Alice", "request": "..."}, ... ]
"""

import os
import json
import threading
import time

# ── Configuração ────────────────────────────────────────────────────────────

AUDIO_DIR          = "audios"                # pasta com ficheiros de áudio
THEME_FILE         = "theme_songs.json"      # mapa uid → caminho do áudio
REQUESTS_FILE      = "music_requests.json"   # pedidos pendentes para o admin
SUPPORTED_EXT      = (".mp3", ".wav", ".ogg", ".flac")

# Segundos mínimos entre dois plays da mesma (ou diferente) música.
# Evita que a música recomece a cada frame enquanto a pessoa está no ecrã.
PLAY_COOLDOWN      = 15

# ── Detecção opcional do pygame ─────────────────────────────────────────────

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception as _e:
    _PYGAME_OK = False
    print(f"[theme_song] pygame não disponível ({_e}). "
          "As theme songs não serão reproduzidas.")

# ── Estado de reprodução (partilhado entre threads) ─────────────────────────

_play_lock      = threading.Lock()
_last_played_uid: "int | None" = None   # uid da última pessoa cujo tema tocou
_last_play_time: float       = 0.0    # timestamp do último play


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE PERSISTÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def _load_themes() -> dict:
    """Carrega o mapeamento uid (str) → caminho do áudio."""
    if os.path.exists(THEME_FILE):
        try:
            return json.load(open(THEME_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_themes(themes: dict) -> None:
    """Persiste o mapeamento de temas."""
    json.dump(themes, open(THEME_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def _load_requests() -> list:
    """Carrega a lista de pedidos de música pendentes."""
    if os.path.exists(REQUESTS_FILE):
        try:
            return json.load(open(REQUESTS_FILE, encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_requests(requests: list) -> None:
    """Persiste os pedidos de música."""
    json.dump(requests, open(REQUESTS_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  LISTAGEM DE ÁUDIOS DISPONÍVEIS
# ══════════════════════════════════════════════════════════════════════════════

def list_audio_files() -> list[str]:
    """
    Devolve a lista de caminhos completos dos ficheiros de áudio em AUDIO_DIR.
    Cria a pasta se não existir.
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in sorted(os.listdir(AUDIO_DIR))
        if f.lower().endswith(SUPPORTED_EXT)
    ]
    return files


# ══════════════════════════════════════════════════════════════════════════════
#  JANELA OPENVC DE SELECÇÃO DE MÚSICA
# ══════════════════════════════════════════════════════════════════════════════

def _ask_theme_in_window(person_name: str) -> str | None:
    """
    Mostra uma janela OpenCV para o utilizador escolher uma theme song.

    Fluxo
    -----
    1. Pergunta Y / N se quer música.
    2. Se Y → mostra lista numerada de músicas disponíveis.
       - Digita o número e confirma com ENTER.
       - Ou pressiona R para fazer um pedido de música ao admin.
    3. Se N → devolve None (sem música).
    4. Se a pasta estiver vazia, vai directamente para o pedido.

    Devolve
    -------
    Caminho do ficheiro escolhido, a string especial ``"REQUEST"``
    se o utilizador quer fazer um pedido, ou ``None`` se não quer música.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None  # sem cv2, não mostramos janela

    WIN = "Theme Song"
    W, H = 480, 380
    BG       = (20,  20,  20)
    CYAN     = (0,  200, 255)
    WHITE    = (255, 255, 255)
    GREY     = (160, 160, 160)
    GREEN    = (0,  220,   0)
    YELLOW   = (0,  220, 220)

    audio_files = list_audio_files()

    # ── Fase 1: queres música? ────────────────────────────────────────────────
    while True:
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        panel[:] = BG

        cv2.putText(panel, f"Theme song para {person_name}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        cv2.line(panel, (20, 48), (W - 20, 48), GREY, 1)
        cv2.putText(panel, "Queres associar uma musica?",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)
        cv2.putText(panel, "Y = Sim    N = Nao (ENTER / ESC)",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('n'), 27):   # N ou ESC → sem música
            cv2.destroyWindow(WIN)
            return None
        if key == ord('y'):
            break
        if key == 13:               # ENTER sem escolha → sem música
            cv2.destroyWindow(WIN)
            return None

    # ── Fase 2: selecção da música (ou pedido) ────────────────────────────────
    if not audio_files:
        # Sem ficheiros disponíveis — ir directamente para pedido
        return _ask_music_request_in_window(WIN, person_name, W, H,
                                            BG, CYAN, WHITE, GREY, YELLOW)

    typed    = ""
    selected = None
    # Quantas músicas cabem visualmente (máx 8 linhas)
    MAX_VISIBLE = min(len(audio_files), 8)

    while selected is None:
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        panel[:] = BG

        cv2.putText(panel, "Escolhe uma musica (numero + ENTER):",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)

        for idx, path in enumerate(audio_files[:MAX_VISIBLE]):
            label  = f"{idx + 1}. {os.path.basename(path)}"
            # trunca para caber na janela
            if len(label) > 52:
                label = label[:49] + "..."
            y_pos  = 60 + idx * 28
            color  = GREEN if str(idx + 1) == typed else WHITE
            cv2.putText(panel, label,
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        if len(audio_files) > MAX_VISIBLE:
            cv2.putText(panel,
                        f"  (+{len(audio_files) - MAX_VISIBLE} mais nao mostradas)",
                        (20, 60 + MAX_VISIBLE * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)

        # Caixa de input
        input_y = H - 100
        cv2.line(panel, (20, input_y - 10), (W - 20, input_y - 10), GREY, 1)
        cv2.putText(panel, "Numero: " + typed + "|",
                    (20, input_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.putText(panel, "R = pedir musica ao admin  |  ESC = cancelar",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:           # ESC → sem música
            cv2.destroyWindow(WIN)
            return None

        elif key == ord('r'):   # pedido ao admin
            return _ask_music_request_in_window(WIN, person_name, W, H,
                                                BG, CYAN, WHITE, GREY, YELLOW)

        elif key == 8:          # BACKSPACE
            typed = typed[:-1]

        elif 48 <= key <= 57:   # dígito 0-9
            typed += chr(key)

        elif key == 13:         # ENTER — validar
            try:
                choice = int(typed)
                if 1 <= choice <= len(audio_files):
                    selected = audio_files[choice - 1]
                else:
                    typed = ""  # número fora de alcance — limpar
            except ValueError:
                typed = ""

    cv2.destroyWindow(WIN)
    return selected


def _ask_music_request_in_window(WIN, person_name, W, H,
                                  BG, CYAN, WHITE, GREY, YELLOW) -> str | None:
    """
    Sub-janela para o utilizador escrever o pedido de música.
    Devolve ``"REQUEST:<texto>"`` ou None se cancelado.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    typed = ""
    while True:
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        panel[:] = BG

        cv2.putText(panel, "Pedido de musica para o admin",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)
        cv2.line(panel, (20, 50), (W - 20, 50), GREY, 1)
        cv2.putText(panel, "Escreve o nome da musica / artista:",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        # caixa de texto
        cv2.rectangle(panel, (16, 110), (W - 16, 150), (50, 50, 50), -1)
        cv2.rectangle(panel, (16, 110), (W - 16, 150), YELLOW, 1)

        # wrap simples — mostra só os últimos 45 caracteres para não sair da caixa
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
            if typed.strip():
                return f"REQUEST:{typed.strip()}"
            return None
        elif 32 <= key <= 126:
            typed += chr(key)


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA — CHAMADA PELO registration.py
# ══════════════════════════════════════════════════════════════════════════════

def ask_and_save_theme(uid: int, person_name: str) -> None:
    """
    Pergunta ao utilizador se quer uma theme song e guarda a escolha.
    Chamado no final do registo bem-sucedido, ainda dentro da
    registration_thread (antes de limpar registration_busy).

    Trata também dos pedidos: guarda-os em REQUESTS_FILE para o admin ver.
    """
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

    # Ficheiro de áudio válido escolhido
    themes = _load_themes()
    themes[str(uid)] = result
    _save_themes(themes)
    print(f"  [theme] Theme song de '{person_name}' → {result}")


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUÇÃO — CHAMADA PELO face_recognition.py
# ══════════════════════════════════════════════════════════════════════════════

def play_theme_for(uid: int) -> None:
    """
    Toca a theme song do utilizador com o *uid* dado, numa thread separada,
    se e só se:
      - pygame estiver disponível,
      - existir uma música associada a esse uid,
      - o cooldown desde o último play tiver expirado.

    Seguro para chamar em qualquer thread — usa lock interno.
    Nunca lança excepção para o caller.
    """
    if not _PYGAME_OK:
        return

    themes = _load_themes()
    path   = themes.get(str(uid))
    if not path or not os.path.exists(path):
        return

    with _play_lock:
        global _last_played_uid, _last_play_time
        now = time.time()

        # Cooldown global: não tocar se já está a tocar ou se passou pouco tempo
        if (now - _last_play_time) < PLAY_COOLDOWN:
            return

        _last_played_uid = uid
        _last_play_time  = now

    # Lança numa thread daemon para não bloquear o loop principal
    threading.Thread(target=_play_audio, args=(path,), daemon=True).start()


def _play_audio(path: str) -> None:
    """Worker que realmente toca o ficheiro. Corre numa thread separada."""
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        # Aguarda o fim da reprodução sem bloquear outras threads
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[theme_song] Erro ao reproduzir '{path}': {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  GESTÃO DE PEDIDOS — MENU DO ADMIN
# ══════════════════════════════════════════════════════════════════════════════

def list_music_requests() -> None:
    """
    Imprime todos os pedidos de música pendentes.
    Chamado a partir do management_menu em face_recognition.py.
    """
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
    """
    Permite ao admin marcar um pedido como resolvido (interactivo, no terminal).
    """
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
