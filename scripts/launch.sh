#!/usr/bin/env bash
# Launcher gráfico do Neurosonancy: abre um terminal novo e roda run.sh dentro.
# Usado pelo .desktop entry. run.sh faz as validações e chama main.py.

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR" || exit 1

RUN_SH="$DIR/run.sh"

if [ ! -x "$RUN_SH" ]; then
    echo "[neurosonancy] $RUN_SH ausente ou sem permissão de execução." >&2
    echo "[neurosonancy] rode: chmod +x $RUN_SH" >&2
    exit 1
fi

if command -v kitty >/dev/null 2>&1; then
    exec kitty \
        --title "NEUROSONANCY" \
        --app-id io.neurosonancy.app \
        --override background_opacity=0.95 \
        bash "$RUN_SH" "$@"
fi

if command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal \
        --title="NEUROSONANCY" \
        --geometry=180x50 \
        -- bash "$RUN_SH" "$@"
fi

if command -v xterm >/dev/null 2>&1; then
    exec xterm -title "NEUROSONANCY" -geometry 180x50 -e bash "$RUN_SH" "$@"
fi

# Sem terminal gráfico disponível: roda inline (TTY atual).
echo "[neurosonancy] nenhum terminal gráfico encontrado; rodando no TTY atual."
exec bash "$RUN_SH" "$@"
