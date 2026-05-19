#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_MAIN="$SCRIPT_DIR/venv"
VENV_CHATTERBOX="$SCRIPT_DIR/venv_chatterbox"
VENV_COQUI="$SCRIPT_DIR/venv_coqui"
PYTHON_CMD="python3"
DESKTOP_FILE="$HOME/.local/share/applications/neurosonancy.desktop"
ICON_PATH="$SCRIPT_DIR/assets/icon.png"

echo "============================================"
echo "    NEUROSONANCY // INSTALLER v4.1         "
echo "    Voice Cloning + Leitor (offline-first)"
echo "============================================"
echo ""

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "[ERRO] Python3 nao encontrado. Instale Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python version: $PYTHON_VERSION"

if ! command -v ffmpeg &> /dev/null || ! command -v ffprobe &> /dev/null; then
    echo "[ERRO] ffmpeg/ffprobe necessarios (concat de audio + leitura de duracao)."
    echo "       Instale com: sudo apt install ffmpeg"
    exit 1
fi

if ! command -v kitty &> /dev/null; then
    echo "[AVISO] 'kitty' nao encontrado (scripts/launch.sh usa kitty por padrao)."
    echo "        Voce ainda pode rodar 'venv/bin/python main.py' direto em qualquer terminal."
    echo "        Para instalar: sudo apt install kitty"
fi

check_venv() {
    local venv_path=$1
    local test_import=$2

    if [ ! -d "$venv_path" ]; then
        return 1
    fi

    if [ ! -f "$venv_path/bin/python" ]; then
        return 1
    fi

    if [ -n "$test_import" ]; then
        if ! "$venv_path/bin/python" -c "$test_import" &>/dev/null; then
            return 1
        fi
    fi

    return 0
}

install_venv() {
    local venv_path=$1
    local venv_name=$2
    local requirements=$3

    echo "  -> Criando ambiente virtual..."
    $PYTHON_CMD -m venv "$venv_path"

    echo "  -> Instalando dependencias base..."
    "$venv_path/bin/pip" install --upgrade pip wheel setuptools --quiet

    if [ -f "$requirements" ]; then
        echo "  -> Instalando requirements..."
        "$venv_path/bin/pip" install -r "$requirements" --quiet
    fi

    echo "  -> $venv_name instalado!"
}

echo ""
echo "[FASE 1/4] Ambiente Principal"
echo "============================================"
if check_venv "$VENV_MAIN" "import textual; import elevenlabs"; then
    echo "[OK] venv/ ja existe. Atualizando dependencias..."
else
    echo "[INFO] Instalando venv (principal)..."
    [ -d "$VENV_MAIN" ] && rm -rf "$VENV_MAIN"
    python3 -m venv "$VENV_MAIN"
fi
echo "  -> Instalando requirements.txt..."
"$VENV_MAIN/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo "  -> venv principal instalado!"

echo ""
echo "[FASE 2/4] Ambiente Chatterbox TTS (Multilingual)"
echo "============================================"
if check_venv "$VENV_CHATTERBOX" "from chatterbox.mtl_tts import ChatterboxMultilingualTTS"; then
    echo "[OK] venv_chatterbox/ ja existe. Atualizando dependencias..."
else
    echo "[INFO] Instalando venv_chatterbox..."
    [ -d "$VENV_CHATTERBOX" ] && rm -rf "$VENV_CHATTERBOX"
    install_venv "$VENV_CHATTERBOX" "venv_chatterbox" ""
fi
echo "  -> Instalando Chatterbox TTS Multilingual + PyTorch + peft..."
"$VENV_CHATTERBOX/bin/pip" install chatterbox-tts torch torchaudio peft --quiet
echo "  -> Chatterbox TTS Multilingual instalado!"

echo ""
echo "[FASE 3/4] Ambiente Coqui TTS"
echo "============================================"
if check_venv "$VENV_COQUI" "from TTS.api import TTS"; then
    echo "[OK] venv_coqui/ ja existe. Atualizando dependencias..."
else
    echo "[INFO] Instalando venv_coqui..."
    [ -d "$VENV_COQUI" ] && rm -rf "$VENV_COQUI"
    install_venv "$VENV_COQUI" "venv_coqui" ""
fi
echo "  -> Instalando Coqui TTS + PyTorch + extras..."
"$VENV_COQUI/bin/pip" install coqui-tts torch torchaudio pydub torchcodec --quiet
echo "  -> Coqui TTS instalado!"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data_input"
mkdir -p "$SCRIPT_DIR/data_output"
mkdir -p "$SCRIPT_DIR/models/coqui"

echo ""
echo "[FASE 4/5] Modelos TTS Locais (~1.8 GB)"
echo "============================================"
if [ -f "$SCRIPT_DIR/models/coqui/tts_models--multilingual--multi-dataset--xtts_v2/model.pth" ]; then
    echo "[OK] XTTS v2 ja presente em models/coqui/."
else
    echo "[INFO] Baixando XTTS v2 para models/coqui/ (offline-first)..."
    if ! bash "$SCRIPT_DIR/scripts/download_models.sh"; then
        echo "[ERRO] Falha ao baixar modelos. O Leitor de Textos nao funcionara."
        echo "       Rode manualmente: bash scripts/download_models.sh"
        # nao aborta a instalacao — usuario pode preferir baixar depois
    fi
fi

echo ""
echo "[FASE 5/5] Registrando Aplicativo"
echo "============================================"

mkdir -p "$HOME/.local/share/applications"
chmod +x "$SCRIPT_DIR/scripts/launch.sh"

cp "$SCRIPT_DIR/neurosonancy.desktop" "$DESKTOP_FILE"
chmod +x "$DESKTOP_FILE"

if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
fi

echo "[OK] Aplicativo registrado no menu!"
echo "  -> $DESKTOP_FILE"
echo "  -> Launcher: $SCRIPT_DIR/scripts/launch.sh"

echo ""
echo "============================================"
echo "        INSTALACAO CONCLUIDA               "
echo "============================================"
echo ""
echo "Ambientes:"
echo "  - venv/            : Interface principal (ElevenLabs apenas para clonagem)"
echo "  - venv_chatterbox/ : Chatterbox TTS (offline)"
echo "  - venv_coqui/      : Coqui XTTS v2 (offline; usado pelo Leitor de Textos)"
echo ""
echo "Modelos:"
echo "  - models/coqui/    : XTTS v2 base (~1.8 GB; baixe via 'make download-models' se faltar)"
echo ""
echo "Para executar:"
echo "  1. Busque 'Neurosonancy' no menu de aplicativos"
echo "  2. Ou execute: cd $SCRIPT_DIR && ./run.sh        (valida + lanca inline)"
echo "  3. Ou: make run                                  (mesmo que ./run.sh)"
echo "  4. Validar ambiente sem subir o app: ./run.sh --check"
echo "  5. Atalhos da TUI: [1] Media | [2] Trainer | [3] Clone | [4] Leitor"
echo ""
