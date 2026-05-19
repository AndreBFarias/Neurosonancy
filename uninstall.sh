#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_MAIN="$SCRIPT_DIR/venv"
VENV_CHATTERBOX="$SCRIPT_DIR/venv_chatterbox"
VENV_COQUI="$SCRIPT_DIR/venv_coqui"
DESKTOP_FILE="$HOME/.local/share/applications/neurosonancy.desktop"

echo "============================================"
echo "    NEUROSONANCY // UNINSTALLER            "
echo "============================================"
echo ""

remove_venv() {
    local venv_path=$1
    local venv_name=$2

    if [ -d "$venv_path" ]; then
        echo "[INFO] Removendo $venv_name..."
        rm -rf "$venv_path"
        echo "  -> Removido!"
    else
        echo "[INFO] $venv_name nao encontrado."
    fi
}

remove_venv "$VENV_MAIN" "venv (principal)"
remove_venv "$VENV_CHATTERBOX" "venv_chatterbox"
remove_venv "$VENV_COQUI" "venv_coqui"

echo ""
echo "[INFO] Removendo cache Python..."
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "[OPCIONAL] Dados pesados (preservados por padrao)"
echo "============================================"

prompt_remove() {
    local path=$1
    local label=$2
    if [ ! -d "$path" ]; then
        return 0
    fi
    local size
    size=$(du -sh "$path" 2>/dev/null | cut -f1)
    echo ""
    echo "[?] Remover $label? ($path, $size)"
    read -p "    digite 'sim' para confirmar [padrao: nao]: " confirm
    if [ "$confirm" = "sim" ]; then
        rm -rf "$path"
        echo "  -> Removido."
    else
        echo "  -> Preservado."
    fi
}

prompt_remove "$SCRIPT_DIR/models" "modelos TTS locais (XTTS v2 ~1.8GB)"
prompt_remove "$SCRIPT_DIR/data_output" "datasets gerados (clone_voice, lore_inteiro, etc)"

echo ""
echo "[INFO] Removendo entrada do menu..."
if [ -f "$DESKTOP_FILE" ]; then
    rm -f "$DESKTOP_FILE"
    echo "  -> Removido: $DESKTOP_FILE"
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi
else
    echo "  -> Arquivo .desktop nao encontrado."
fi

echo ""
echo "============================================"
echo "        DESINSTALACAO CONCLUIDA            "
echo "============================================"
echo ""
echo "Ambientes virtuais e entrada do menu removidos."
echo "Arquivos fonte preservados."
echo "Modelos e datasets preservados (a menos que voce confirmou remocao acima)."
echo ""
echo "Para remover TUDO definitivamente, delete a pasta manualmente:"
echo "  rm -rf $SCRIPT_DIR"
echo ""
