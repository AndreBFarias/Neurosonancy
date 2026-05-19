"""Constantes do TTS daemon (multi-engine: coqui, chatterbox)."""
from __future__ import annotations

from pathlib import Path

# Engines suportados. Cada um roda em seu próprio venv + socket.
ENGINES = ("coqui", "chatterbox")
DEFAULT_ENGINE = "coqui"


def socket_path_for(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"engine desconhecido: {engine}")
    return f"/tmp/neurosonancy_tts_{engine}.sock"


def pid_file_for(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"engine desconhecido: {engine}")
    return f"/tmp/neurosonancy_tts_{engine}.pid"


def venv_python_for(engine: str, root_dir: Path) -> Path:
    """Retorna o python do venv responsável por carregar a lib do engine."""
    if engine == "coqui":
        return root_dir / "venv_coqui" / "bin" / "python"
    if engine == "chatterbox":
        return root_dir / "venv_chatterbox" / "bin" / "python"
    raise ValueError(f"engine desconhecido: {engine}")


# Timeouts (segundos)
CONNECT_TIMEOUT = 5.0
HEALTH_TIMEOUT = 3.0
STARTUP_TIMEOUT = 120.0
STARTUP_POLL_INTERVAL = 1.0

# Geração: timeout dinâmico baseado em len(text). Chatterbox em CPU pode levar
# até ~0.3 s/caractere; Coqui XTTS em CUDA é muito mais rápido.
# A janela mínima cobre cold-start; a máxima evita travamento eterno.
GENERATE_TIMEOUT_MIN = 120.0
GENERATE_TIMEOUT_PER_CHAR = 0.3
GENERATE_TIMEOUT_MAX = 1800.0

# Retrocompat: alguns callers passam GENERATE_TIMEOUT direto.
GENERATE_TIMEOUT = GENERATE_TIMEOUT_MAX


def timeout_for(text_len: int) -> float:
    """Calcula timeout adequado pro tamanho do texto."""
    return max(
        GENERATE_TIMEOUT_MIN,
        min(GENERATE_TIMEOUT_MAX, GENERATE_TIMEOUT_PER_CHAR * max(0, text_len)),
    )

# Protocolo
MSG_DELIM = b"\n"
RECV_CHUNK = 4096
