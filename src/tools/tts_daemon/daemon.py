"""TTS daemon socket-based, multi-engine (Coqui XTTS v2 ou Chatterbox MTL).

Cada engine roda em um daemon próprio, no seu próprio venv:
- engine=coqui      → venv_coqui   → socket /tmp/neurosonancy_tts_coqui.sock
- engine=chatterbox → venv_chatterbox → socket /tmp/neurosonancy_tts_chatterbox.sock

Iniciar:
    venv_coqui/bin/python      -m src.tools.tts_daemon.daemon --engine coqui
    venv_chatterbox/bin/python -m src.tools.tts_daemon.daemon --engine chatterbox

Parar:
    venv_coqui/bin/python      -m src.tools.tts_daemon.daemon --engine coqui --stop
    venv_chatterbox/bin/python -m src.tools.tts_daemon.daemon --engine chatterbox --stop

Protocolo (JSON por linha):
    Request:  {"command": "generate", "text": "...", "reference_wav": "...",
               "output_wav": "...", "speed": 1.0, "language": "pt"}
    Response: {"status": "ok", "elapsed_ms": 1234} | {"status": "error", "message": "..."}

    Request:  {"command": "health"}
    Response: {"status": "ok", "engine": "coqui|chatterbox", "device": "cuda|cpu", "model_loaded": true}

    Request:  {"command": "shutdown"}
    Response: {"status": "ok"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from .constants import (
    DEFAULT_ENGINE,
    ENGINES,
    MSG_DELIM,
    RECV_CHUNK,
    pid_file_for,
    socket_path_for,
)

# Detecta ROOT a partir deste arquivo: <root>/src/tools/tts_daemon/daemon.py
ROOT_DIR = Path(__file__).resolve().parents[3]
COQUI_MODEL_DIR = (
    ROOT_DIR / "models" / "coqui" / "tts_models--multilingual--multi-dataset--xtts_v2"
)
CHATTERBOX_CACHE_DIR = ROOT_DIR / "models" / "chatterbox"

logger = logging.getLogger("tts_daemon")


def _setup_logging(engine: str) -> None:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    handler = logging.FileHandler(
        log_dir / f"tts_daemon_{engine}.log", encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(stdout)


class TTSDaemon:
    """Servidor singleton de TTS para UM engine específico."""

    def __init__(self, engine: str) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine inválido: {engine}")
        self.engine = engine
        self.model = None
        self.device = "cpu"
        self.server_socket: socket.socket | None = None
        self._running = False
        self._lock = threading.Lock()
        # Defesa offline-first em todos os engines
        CHATTERBOX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(CHATTERBOX_CACHE_DIR))
        os.environ.setdefault("HF_HUB_CACHE", str(CHATTERBOX_CACHE_DIR / "hub"))
        if engine == "coqui":
            os.environ.setdefault("TTS_HOME", str(COQUI_MODEL_DIR.parent))
            os.environ.setdefault("COQUI_TOS_AGREED", "1")

    def load_model(self) -> bool:
        try:
            import torch  # noqa: WPS433
        except ImportError:
            logger.error("torch ausente; venv mal instalado.")
            return False

        if self.engine == "coqui":
            return self._load_coqui(torch)
        if self.engine == "chatterbox":
            return self._load_chatterbox(torch)
        return False

    def _load_coqui(self, torch) -> bool:
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = patched_load

        try:
            from TTS.api import TTS  # noqa: WPS433
        except ImportError:
            logger.error("coqui-tts ausente no venv_coqui.")
            return False

        if not (COQUI_MODEL_DIR / "model.pth").is_file():
            logger.error(
                "Modelo XTTS não encontrado em %s. Rode scripts/download_models.sh.",
                COQUI_MODEL_DIR,
            )
            return False

        logger.info("Carregando XTTS v2 de %s...", COQUI_MODEL_DIR)
        t0 = time.time()
        self.model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        )
        force_cpu = os.environ.get("NEUROSONANCY_FORCE_CPU", "0") == "1"
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("Falha ao mover modelo para %s: %s", self.device, exc)
            self.device = "cpu"
        logger.info("XTTS carregado em %.1fs (device=%s).", time.time() - t0, self.device)
        return True

    def _load_chatterbox(self, torch) -> bool:
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError:
            try:
                from chatterbox.tts_multilingual import ChatterboxMultilingualTTS
            except ImportError:
                logger.error("chatterbox-tts ausente no venv_chatterbox.")
                return False

        logger.info(
            "Carregando Chatterbox Multilingual (HF cache: %s)...",
            CHATTERBOX_CACHE_DIR,
        )
        t0 = time.time()

        # 1ª tentativa: CUDA (se disponível e não forçado pra CPU via env)
        force_cpu = os.environ.get("NEUROSONANCY_FORCE_CPU", "0") == "1"
        prefer_cuda = torch.cuda.is_available() and not force_cpu
        if prefer_cuda:
            try:
                self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
                self.device = "cuda"
                logger.info(
                    "Chatterbox carregado em %.1fs (device=cuda).",
                    time.time() - t0,
                )
                return True
            except torch.cuda.OutOfMemoryError as exc:
                logger.warning(
                    "CUDA OOM ao carregar Chatterbox (%s). "
                    "Caindo para CPU. Para forçar GPU, pare o outro daemon TTS antes.",
                    exc,
                )
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "CUDA OOM ao carregar Chatterbox (%s). Caindo para CPU.",
                        exc,
                    )
                    torch.cuda.empty_cache()
                else:
                    logger.exception("Falha ao carregar Chatterbox na CUDA")
                    return False

        # 2ª tentativa: CPU
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        except Exception as exc:
            logger.exception("Falha ao carregar Chatterbox (CPU)")
            logger.error("Detalhes: %s", exc)
            return False
        self.device = "cpu"
        logger.info(
            "Chatterbox carregado em %.1fs (device=cpu).",
            time.time() - t0,
        )
        return True

    def generate(
        self,
        text: str,
        reference_wav: str,
        output_wav: str,
        speed: float,
        language: str,
    ) -> dict:
        if self.model is None:
            return {"status": "error", "message": "Modelo não carregado"}
        text = (text or "").strip()
        if not text:
            return {"status": "error", "message": "Texto vazio"}
        if not Path(reference_wav).is_file():
            return {"status": "error", "message": f"reference_wav inexistente: {reference_wav}"}

        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        with self._lock:
            try:
                if self.engine == "coqui":
                    self.model.tts_to_file(
                        text=text,
                        file_path=output_wav,
                        speaker_wav=reference_wav,
                        language=language,
                        speed=speed,
                        split_sentences=False,
                    )
                elif self.engine == "chatterbox":
                    self._chatterbox_generate(
                        text=text,
                        reference_wav=reference_wav,
                        output_wav=output_wav,
                        language=language,
                    )
                else:
                    return {"status": "error", "message": f"engine não implementado: {self.engine}"}
            except Exception as exc:
                logger.exception("Falha na síntese")
                return {"status": "error", "message": str(exc)}

        elapsed_ms = int((time.time() - t0) * 1000)
        if not Path(output_wav).is_file() or Path(output_wav).stat().st_size < 1024:
            return {"status": "error", "message": f"Saída vazia/ausente: {output_wav}"}
        return {"status": "ok", "elapsed_ms": elapsed_ms}

    def _chatterbox_generate(
        self,
        text: str,
        reference_wav: str,
        output_wav: str,
        language: str,
    ) -> None:
        """Sintetiza via Chatterbox e salva em `output_wav` como wav 24kHz."""
        import torch  # noqa: WPS433
        import torchaudio  # noqa: WPS433

        # API mais recente do Chatterbox Multilingual:
        # model.generate(text, audio_prompt_path, language_id, ...)
        try:
            wav = self.model.generate(
                text=text,
                audio_prompt_path=reference_wav,
                language_id=language,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM durante generate (device={self.device}, "
                f"text_len={len(text)}). GPU sem memória pra inferência. "
                f"Mitigações: 1) feche outros daemons/apps GPU; 2) reduza chunk_size; "
                f"3) export NEUROSONANCY_FORCE_CPU=1 antes de subir o daemon. "
                f"Detalhes: {exc}"
            ) from exc
        if wav is None:
            raise RuntimeError("Chatterbox.generate retornou None")

        audio_cpu = wav.cpu()
        if audio_cpu.numel() == 0:
            raise RuntimeError("Chatterbox produziu tensor vazio (0 samples)")
        if audio_cpu.dim() == 3:
            audio_cpu = audio_cpu.squeeze(0)
        if audio_cpu.dim() == 1:
            audio_cpu = audio_cpu.unsqueeze(0)
        if audio_cpu.dim() != 2 or audio_cpu.shape[1] < 1024:
            raise RuntimeError(
                f"Chatterbox produziu shape inválido: {tuple(audio_cpu.shape)} "
                f"(esperado 2D com pelo menos 1024 samples)"
            )

        # Chatterbox sample rate: usa `self.model.sr` se disponível, senão 24000.
        sr = getattr(self.model, "sr", 24000)
        torchaudio.save(output_wav, audio_cpu, sr)

    def handle_client(self, client_sock: socket.socket) -> None:
        request_start = time.time()
        response: dict | None = None
        bytes_in = 0
        try:
            data = b""
            while True:
                chunk = client_sock.recv(RECV_CHUNK)
                if not chunk:
                    break
                data += chunk
                bytes_in = len(data)
                if MSG_DELIM in data:
                    break
            if not data:
                logger.debug("handle_client: cliente fechou sem enviar nada")
                return
            raw = data.split(MSG_DELIM, 1)[0]
            try:
                request = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                response = {
                    "status": "error",
                    "message": f"JSON inválido: {exc}",
                }
            else:
                cmd = request.get("command", "generate")
                logger.info(
                    "request cmd=%s text_len=%d bytes_in=%d",
                    cmd,
                    len(request.get("text", "")) if cmd == "generate" else 0,
                    bytes_in,
                )
                try:
                    response = self._dispatch(request)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Erro inesperado em _dispatch")
                    response = {
                        "status": "error",
                        "message": f"erro interno: {exc.__class__.__name__}: {exc}",
                    }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro inesperado em handle_client (parsing)")
            response = {
                "status": "error",
                "message": f"erro parsing: {exc.__class__.__name__}: {exc}",
            }

        # Tenta sempre enviar a resposta, mesmo que o socket possa estar quebrado.
        if response is not None:
            payload = json.dumps(response).encode("utf-8") + MSG_DELIM
            try:
                client_sock.sendall(payload)
                elapsed = time.time() - request_start
                logger.info(
                    "response status=%s bytes_out=%d elapsed=%.2fs",
                    response.get("status"),
                    len(payload),
                    elapsed,
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                # Cliente fechou antes do daemon terminar (timeout do client, etc).
                # O daemon não morre — apenas registra. Quem chamou recebe "resposta vazia"
                # do recv() e vai logar essa causa do lado dele.
                elapsed = time.time() - request_start
                logger.warning(
                    "cliente fechou socket antes da resposta (status=%s, elapsed=%.2fs): %s",
                    response.get("status"),
                    elapsed,
                    exc,
                )

        try:
            client_sock.close()
        except OSError:
            pass

    def _dispatch(self, request: dict) -> dict:
        command = request.get("command", "generate")
        if command == "generate":
            return self.generate(
                text=request.get("text", ""),
                reference_wav=request.get("reference_wav", ""),
                output_wav=request.get("output_wav", ""),
                speed=float(request.get("speed", 1.0)),
                language=request.get("language", "pt"),
            )
        if command == "health":
            return {
                "status": "ok",
                "engine": self.engine,
                "device": self.device,
                "model_loaded": self.model is not None,
            }
        if command == "shutdown":
            logger.info("Shutdown solicitado via cliente.")
            self._running = False
            return {"status": "ok"}
        return {"status": "error", "message": f"Comando desconhecido: {command}"}

    def serve(self) -> None:
        sock_path = socket_path_for(self.engine)
        pid_file = pid_file_for(self.engine)

        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(sock_path)
        self.server_socket.listen(8)
        self.server_socket.settimeout(1.0)
        os.chmod(sock_path, 0o666)  # nosec: socket local

        Path(pid_file).write_text(str(os.getpid()), encoding="utf-8")
        self._running = True
        logger.info(
            "Daemon %s escutando em %s (PID %d)",
            self.engine,
            sock_path,
            os.getpid(),
        )

        try:
            while self._running:
                try:
                    client_sock, _ = self.server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if not self._running:
                        break
                    raise
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_sock,),
                    daemon=True,
                )
                thread.start()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        logger.info("Encerrando daemon %s...", self.engine)
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
        for path in (socket_path_for(self.engine), pid_file_for(self.engine)):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Falha ao remover %s: %s", path, exc)


def _send_stop(engine: str) -> int:
    pid_file = pid_file_for(engine)
    if not Path(pid_file).is_file():
        print(f"Daemon {engine} não está rodando (sem PID file).")
        return 0
    try:
        pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        print(f"PID inválido: {exc}", file=sys.stderr)
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"SIGTERM enviado para daemon {engine} (PID {pid}).")
        return 0
    except ProcessLookupError:
        print(f"PID {pid} não existe (órfão); limpando arquivos.")
        for path in (socket_path_for(engine), pid_file):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        return 0
    except OSError as exc:
        print(f"Falha ao matar PID {pid}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Neurosonancy TTS daemon")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=DEFAULT_ENGINE,
        help="qual engine o daemon vai servir (default: %(default)s)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="encerra o daemon do engine indicado",
    )
    args = parser.parse_args()

    if args.stop:
        return _send_stop(args.engine)

    _setup_logging(args.engine)
    daemon = TTSDaemon(args.engine)

    def _on_signal(_sig, _frm):
        logger.info("Sinal recebido — encerrando.")
        daemon._running = False  # noqa: SLF001

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if not daemon.load_model():
        logger.error("Falha ao carregar modelo; abortando.")
        return 1

    daemon.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
