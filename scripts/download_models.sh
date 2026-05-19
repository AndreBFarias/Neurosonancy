#!/usr/bin/env bash
# Download dos modelos TTS locais para <root>/models/.
# Idempotente: skipa modelos já presentes.
#
# Uso:
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh --force   # baixa de novo mesmo se existir

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT}/models"
COQUI_DIR="${MODELS_DIR}/coqui"
CHATTERBOX_DIR="${MODELS_DIR}/chatterbox"
XTTS_DIR="${COQUI_DIR}/tts_models--multilingual--multi-dataset--xtts_v2"
VENV_COQUI_PY="${ROOT}/venv_coqui/bin/python"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

mkdir -p "${COQUI_DIR}"
mkdir -p "${CHATTERBOX_DIR}"

if [[ ${FORCE} -eq 0 && -f "${XTTS_DIR}/model.pth" ]]; then
    SIZE=$(du -sh "${XTTS_DIR}" | cut -f1)
    echo "[xtts_v2] já presente em ${XTTS_DIR} (${SIZE}); pulando."
else
    if [[ ! -x "${VENV_COQUI_PY}" ]]; then
        echo "[xtts_v2] ERRO: venv_coqui não encontrado em ${VENV_COQUI_PY}" >&2
        echo "         Crie o venv e instale coqui-tts antes de rodar este script." >&2
        exit 1
    fi
    echo "[xtts_v2] baixando para ${XTTS_DIR}..."
    TTS_HOME="${COQUI_DIR}" "${VENV_COQUI_PY}" - <<'PY'
import os
from TTS.api import TTS

home = os.environ.get("TTS_HOME", "")
print(f"[xtts_v2] TTS_HOME={home}")
TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("[xtts_v2] download concluido")
PY
    # O Coqui salva direto em $TTS_HOME/tts_models--multilingual--multi-dataset--xtts_v2/
fi

# Validação pós-download (cobre Coqui mudar convenção de path numa versão futura)
if [[ ! -f "${XTTS_DIR}/model.pth" ]]; then
    echo "[xtts_v2] ERRO: model.pth nao encontrado em ${XTTS_DIR} apos download." >&2
    echo "          Conteudo de ${COQUI_DIR}:" >&2
    ls -la "${COQUI_DIR}" >&2
    exit 1
fi
SIZE_BYTES=$(stat -c '%s' "${XTTS_DIR}/model.pth")
if [[ ${SIZE_BYTES} -lt 1000000000 ]]; then  # < 1GB indica download truncado
    echo "[xtts_v2] ERRO: model.pth com tamanho suspeito (${SIZE_BYTES} bytes; esperado ~1.8GB)." >&2
    echo "          Apague ${XTTS_DIR} e rode novamente." >&2
    exit 1
fi

# Chatterbox: pasta pre-criada. O modelo é baixado on-demand pelo daemon
# com HF_HOME=models/chatterbox/ — primeira chamada baixa, demais usam o cache.

echo "OK — modelos prontos em ${MODELS_DIR}"
