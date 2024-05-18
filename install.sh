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
echo "    NEUROSONANCY // INSTALLER v4.0         "
echo "    Voice Cloning & Training Toolkit       "
echo "============================================"
echo ""

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "[ERRO] Python3 nao encontrado. Instale Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python version: $PYTHON_VERSION"

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
    echo "[OK] venv/ ja instalado e funcional. Pulando..."
else
    echo "[INFO] Instalando venv (principal)..."
    [ -d "$VENV_MAIN" ] && rm -rf "$VENV_MAIN"
    install_venv "$VENV_MAIN" "venv (principal)" "$SCRIPT_DIR/requirements.txt"
fi

echo ""
echo "[FASE 2/4] Ambiente Chatterbox TTS"
echo "============================================"
if check_venv "$VENV_CHATTERBOX" "from chatterbox.tts import ChatterboxTTS"; then
    echo "[OK] venv_chatterbox/ ja instalado e funcional. Pulando..."
else
    echo "[INFO] Instalando venv_chatterbox..."
    [ -d "$VENV_CHATTERBOX" ] && rm -rf "$VENV_CHATTERBOX"
    install_venv "$VENV_CHATTERBOX" "venv_chatterbox" ""
    echo "  -> Instalando Chatterbox TTS + PyTorch..."
    "$VENV_CHATTERBOX/bin/pip" install chatterbox-tts torch torchaudio --quiet
    echo "  -> Chatterbox TTS instalado!"
fi

echo ""
echo "[FASE 3/4] Ambiente Coqui TTS"
echo "============================================"
if check_venv "$VENV_COQUI" "from TTS.api import TTS"; then
    echo "[OK] venv_coqui/ ja instalado e funcional. Pulando..."
else
    echo "[INFO] Instalando venv_coqui..."
    [ -d "$VENV_COQUI" ] && rm -rf "$VENV_COQUI"
    install_venv "$VENV_COQUI" "venv_coqui" ""
    echo "  -> Instalando Coqui TTS + PyTorch..."
    "$VENV_COQUI/bin/pip" install coqui-tts torch torchaudio pydub torchcodec --quiet
    echo "  -> Coqui TTS instalado!"
fi

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data_input"
mkdir -p "$SCRIPT_DIR/data_output"

echo ""
echo "[FASE 4/4] Registrando Aplicativo"
echo "============================================"

mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Neurosonancy
GenericName=Voice Cloning Toolkit
Comment=Voice Cloning and Training Toolkit
Exec=bash -c 'cd "$SCRIPT_DIR" && source venv/bin/activate && python3 main.py'
Icon=$ICON_PATH
Terminal=true
Categories=AudioVideo;Audio;Utility;
Keywords=voice;clone;tts;audio;training;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
fi

echo "[OK] Aplicativo registrado no menu!"
echo "  -> $DESKTOP_FILE"

echo ""
echo "============================================"
echo "        INSTALACAO CONCLUIDA               "
echo "============================================"
echo ""
echo "Ambientes:"
echo "  - venv/            : Interface principal + ElevenLabs"
echo "  - venv_chatterbox/ : Treinamento Chatterbox TTS"
echo "  - venv_coqui/      : Treinamento Coqui XTTS"
echo ""
echo "Para executar:"
echo "  1. Busque 'Neurosonancy' no menu de aplicativos"
echo "  2. Ou execute: cd $SCRIPT_DIR && source venv/bin/activate && python3 main.py"
echo ""
