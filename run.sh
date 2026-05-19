#!/usr/bin/env bash
# Neurosonancy launcher — valida ambiente, baixa o que faltar, sobe a app.
#
# Verificações em ordem:
#   1. ffmpeg/ffprobe (concat + duração)
#   2. venv/ principal (com textual instalado)
#   3. venv_coqui/ (essencial pro Leitor de Textos)
#   4. models/coqui/.../model.pth (baixa via scripts/download_models.sh se ausente)
#   5. venv_chatterbox/ (warning se ausente; Chatterbox engine fica indisponível)
#
# Se alguma verificação crítica falhar, orienta o usuário a rodar ./install.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECK_ONLY=0
APP_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --check|--dry-run) CHECK_ONLY=1 ;;
        --help|-h)
            echo "Uso: $0 [--check] [args...]"
            echo "  --check   só valida o ambiente, não inicia o app"
            echo "  args...   passados direto pro main.py"
            exit 0
            ;;
        *) APP_ARGS+=("$arg") ;;
    esac
done

VENV_MAIN="$SCRIPT_DIR/venv"
VENV_COQUI="$SCRIPT_DIR/venv_coqui"
VENV_CHATTERBOX="$SCRIPT_DIR/venv_chatterbox"
PYTHON_MAIN="$VENV_MAIN/bin/python"
PYTHON_COQUI="$VENV_COQUI/bin/python"
MODELS_DIR="$SCRIPT_DIR/models/coqui/tts_models--multilingual--multi-dataset--xtts_v2"
MAIN_PY="$SCRIPT_DIR/main.py"

color() { printf "\033[%sm%s\033[0m" "$1" "$2"; }
ok()   { echo "  $(color "32" "OK")    $1"; }
warn() { echo "  $(color "33" "AVISO") $1"; }
err()  { echo "  $(color "31" "ERRO")  $1" >&2; }
step() { echo ""; echo "$(color "36;1" "[run]") $1"; }

abort_install() {
    err "$1"
    echo ""
    err "Instalação incompleta. Rode primeiro:"
    err "    cd $SCRIPT_DIR && ./install.sh"
    exit 1
}

step "verificando ambiente"

# 1. ffmpeg / ffprobe
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    err "ffmpeg/ffprobe não encontrados no PATH."
    err "Instale com: sudo apt install ffmpeg"
    exit 1
fi
ok "ffmpeg + ffprobe"

# 2. venv principal
if [ ! -x "$PYTHON_MAIN" ]; then
    abort_install "venv principal ausente em $VENV_MAIN/bin/python."
fi
if ! "$PYTHON_MAIN" -c "import textual" 2>/dev/null; then
    abort_install "venv principal sem textual instalado."
fi
ok "venv/ (principal)"

# 3. venv_coqui (necessário pro Leitor)
if [ ! -x "$PYTHON_COQUI" ]; then
    abort_install "venv_coqui ausente em $VENV_COQUI/bin/python."
fi
if ! "$PYTHON_COQUI" -c "from TTS.api import TTS" 2>/dev/null; then
    abort_install "venv_coqui sem coqui-tts. Reinstale com ./install.sh."
fi
ok "venv_coqui/ (Coqui XTTS engine)"

# 4. modelo XTTS v2
if [ ! -f "$MODELS_DIR/model.pth" ]; then
    warn "Modelo XTTS v2 ausente em models/coqui/."
    step "baixando modelo (~1.8 GB, pode levar alguns minutos)"
    if ! bash "$SCRIPT_DIR/scripts/download_models.sh"; then
        err "Falha ao baixar modelo. O Leitor de Textos não funcionará."
        err "Rode manualmente: bash scripts/download_models.sh"
        exit 1
    fi
fi
# Re-valida tamanho
MODEL_SIZE=$(stat -c '%s' "$MODELS_DIR/model.pth" 2>/dev/null || echo 0)
if [ "$MODEL_SIZE" -lt 1000000000 ]; then
    err "Modelo XTTS truncado ($MODEL_SIZE bytes; esperado ~1.8 GB)."
    err "Apague $MODELS_DIR e rode: make download-models"
    exit 1
fi
ok "models/coqui/xtts_v2/model.pth (~1.8 GB)"

# 5. venv_chatterbox (não-crítico)
if [ -x "$VENV_CHATTERBOX/bin/python" ]; then
    if "$VENV_CHATTERBOX/bin/python" -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS" 2>/dev/null; then
        ok "venv_chatterbox/ (Chatterbox engine)"
    else
        warn "venv_chatterbox existe mas chatterbox-tts não está utilizável."
        warn "Engine Chatterbox no Leitor ficará indisponível."
    fi
else
    warn "venv_chatterbox ausente. Engine Chatterbox no Leitor ficará indisponível."
    warn "Pra habilitar, rode ./install.sh."
fi

# 6. ~/Desktop (padrão de output do Leitor; tolerante)
if [ -d "$HOME/Desktop" ]; then
    mkdir -p "$HOME/Desktop/whispers" 2>/dev/null || true
    ok "~/Desktop/whispers (saída padrão do Leitor)"
else
    warn "~/Desktop ausente. Defina outra Saída na aba Leitor antes de gerar."
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    step "ambiente validado (--check); app não foi iniciada."
    exit 0
fi

step "iniciando NEUROSONANCY"
echo ""

exec "$PYTHON_MAIN" "$MAIN_PY" "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
