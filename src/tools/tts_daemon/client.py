"""Cliente do TTS daemon multi-engine (coqui, chatterbox)."""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

from .constants import (
    DEFAULT_ENGINE,
    ENGINES,
    GENERATE_TIMEOUT,
    HEALTH_TIMEOUT,
    MSG_DELIM,
    RECV_CHUNK,
    STARTUP_POLL_INTERVAL,
    STARTUP_TIMEOUT,
    pid_file_for,
    socket_path_for,
    timeout_for,
    venv_python_for,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


class TTSDaemonError(RuntimeError):
    """Erro de comunicação com o daemon."""


def is_running(engine: str = DEFAULT_ENGINE) -> bool:
    """Verifica rapidamente se o daemon do engine responde."""
    if engine not in ENGINES:
        raise TTSDaemonError(f"engine desconhecido: {engine}")
    if not Path(socket_path_for(engine)).exists():
        return False
    try:
        response = _request(engine, {"command": "health"}, timeout=HEALTH_TIMEOUT)
    except TTSDaemonError:
        return False
    return response.get("status") == "ok" and response.get("model_loaded")


def ensure_running(
    engine: str = DEFAULT_ENGINE,
    startup_timeout: float = STARTUP_TIMEOUT,
) -> bool:
    """Inicia o daemon do engine (background) caso não esteja rodando.
    Retorna True quando o daemon responde a `health`."""
    if engine not in ENGINES:
        raise TTSDaemonError(f"engine desconhecido: {engine}")
    if is_running(engine):
        return True

    venv_py = venv_python_for(engine, ROOT_DIR)
    if not venv_py.is_file():
        raise TTSDaemonError(
            f"venv para engine={engine} não encontrado em {venv_py}; "
            "rode bash install.sh primeiro."
        )

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = open(
        log_dir / f"tts_daemon_{engine}_spawn.log", "a", encoding="utf-8"
    )
    try:
        subprocess.Popen(
            [str(venv_py), "-m", "src.tools.tts_daemon.daemon", "--engine", engine],
            cwd=str(ROOT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except OSError as exc:
        log_file.close()
        raise TTSDaemonError(f"Falha ao spawnar daemon {engine}: {exc}") from exc

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if is_running(engine):
            return True
        time.sleep(STARTUP_POLL_INTERVAL)

    raise TTSDaemonError(
        f"Daemon {engine} não respondeu em {startup_timeout:.0f}s. "
        f"Veja {log_dir / f'tts_daemon_{engine}.log'}."
    )


def synthesize(
    text: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float = 1.0,
    language: str = "pt",
    engine: str = DEFAULT_ENGINE,
    timeout: float | None = None,
) -> Path:
    """Pede ao daemon do engine que sintetize `text` para `output_wav`.

    Se `timeout` for None (default), calcula dinamicamente baseado em len(text)
    via `timeout_for()` — chunks longos no Chatterbox CPU recebem mais tempo.
    """
    if engine not in ENGINES:
        raise TTSDaemonError(f"engine desconhecido: {engine}")
    if timeout is None:
        timeout = timeout_for(len(text))
    payload = {
        "command": "generate",
        "text": text,
        "reference_wav": str(Path(reference_wav).resolve()),
        "output_wav": str(Path(output_wav).resolve()),
        "speed": float(speed),
        "language": language,
    }
    response = _request(engine, payload, timeout=timeout)
    if response.get("status") != "ok":
        raise TTSDaemonError(response.get("message", "erro desconhecido"))
    if not Path(output_wav).is_file() or Path(output_wav).stat().st_size < 1024:
        raise TTSDaemonError(f"Saída vazia/ausente: {output_wav}")
    return Path(output_wav)


def shutdown(engine: str = DEFAULT_ENGINE) -> bool:
    """Pede shutdown limpo. Retorna True se o daemon respondeu OK."""
    try:
        response = _request(engine, {"command": "shutdown"}, timeout=HEALTH_TIMEOUT)
    except TTSDaemonError:
        return False
    return response.get("status") == "ok"


def health(engine: str = DEFAULT_ENGINE) -> dict | None:
    """Retorna dict completo do health-check (engine, device, model_loaded)
    ou None se o daemon estiver indisponível.
    """
    if engine not in ENGINES:
        raise TTSDaemonError(f"engine desconhecido: {engine}")
    if not Path(socket_path_for(engine)).exists():
        return None
    try:
        response = _request(engine, {"command": "health"}, timeout=HEALTH_TIMEOUT)
    except TTSDaemonError:
        return None
    if response.get("status") != "ok":
        return None
    return response


def status_all() -> dict[str, bool]:
    """Retorna {engine: running} pra todos os engines."""
    return {engine: is_running(engine) for engine in ENGINES}


def _request(engine: str, payload: dict, timeout: float) -> dict:
    sock_path = socket_path_for(engine)
    if not os.path.exists(sock_path):
        raise TTSDaemonError(f"socket ausente: {sock_path}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(sock_path)
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout) as exc:
            raise TTSDaemonError(f"falha ao conectar em {sock_path}: {exc}") from exc

        data = (json.dumps(payload).encode("utf-8") + MSG_DELIM)
        try:
            sock.sendall(data)
        except OSError as exc:
            raise TTSDaemonError(f"falha ao enviar request: {exc}") from exc

        buf = b""
        bytes_received = 0
        while True:
            try:
                chunk = sock.recv(RECV_CHUNK)
            except socket.timeout as exc:
                raise TTSDaemonError(
                    f"timeout ({timeout:.0f}s) aguardando resposta do daemon {engine} "
                    f"(recebido {bytes_received} bytes parciais até agora)"
                ) from exc
            if not chunk:
                break
            buf += chunk
            bytes_received = len(buf)
            if MSG_DELIM in buf:
                break

        if not buf:
            raise TTSDaemonError(
                f"daemon {engine} fechou socket sem enviar resposta. "
                f"Causas comuns: daemon crashou, sendall() falhou após exception interna, "
                f"ou timeout do cliente. Veja logs/tts_daemon_{engine}.log."
            )
        raw = buf.split(MSG_DELIM, 1)[0]
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TTSDaemonError(
                f"resposta do daemon {engine} é JSON inválido (recebido {bytes_received} bytes): {exc}"
            ) from exc
    finally:
        try:
            sock.close()
        except OSError:
            pass
