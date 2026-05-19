"""Helper para sintetizar áudio via Coqui XTTS v2 local.

Carrega o modelo de `<root>/models/coqui/tts_models--multilingual--multi-dataset--xtts_v2/`
e a reference.wav fornecida pelo chamador. Funciona em duas frentes:

- `synthesize()`        — uma chamada, um wav, um chunk
- `synthesize_chunked()` — texto longo dividido em chunks de sentenças

A execução real acontece em subprocesso dentro do `venv_coqui` (que tem
coqui-tts + torch instalados). O texto é passado via stdin JSON, NUNCA
interpolado em string — evita injection e quebras com aspas/quebras de linha.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = (
    ROOT_DIR / "models" / "coqui" / "tts_models--multilingual--multi-dataset--xtts_v2"
)
WRAPPER_PATH = Path(__file__).with_name("coqui_runner_wrapper.py")
VENV_COQUI_PY = ROOT_DIR / "venv_coqui" / "bin" / "python"

# Daemon socket-based é preferido (modelo carrega 1x e fica residente).
# Pode ser desligado via NEUROSONANCY_TTS_DAEMON=0 — útil em CI ou debug.
USE_DAEMON = os.environ.get("NEUROSONANCY_TTS_DAEMON", "1") != "0"

_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?…])\s+")


class CoquiRunnerError(RuntimeError):
    """Erro proveniente do wrapper Coqui."""


@dataclass(frozen=True)
class ChunkResult:
    index: int
    text: str
    wav_path: Path


def _check_environment(engine: str = "coqui") -> None:
    # Pré-checagens só fazem sentido pro fallback subprocess (coqui).
    # Pra chatterbox, a validação real acontece quando o daemon spawna.
    if engine != "coqui":
        return
    if not VENV_COQUI_PY.is_file():
        raise CoquiRunnerError(
            f"venv_coqui não encontrado em {VENV_COQUI_PY}. "
            "Crie o venv e instale coqui-tts."
        )
    if not WRAPPER_PATH.is_file():
        raise CoquiRunnerError(
            f"Wrapper ausente em {WRAPPER_PATH} (esperado no mesmo diretório)."
        )
    if not (MODEL_DIR / "model.pth").is_file():
        raise CoquiRunnerError(
            f"Modelo XTTS não encontrado em {MODEL_DIR}. "
            "Rode `bash scripts/download_models.sh`."
        )


def synthesize(
    text: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float = 1.0,
    timeout: int = 600,
    language: str = "pt",
    engine: str = "coqui",
) -> Path:
    """Sintetiza um chunk único de texto, gerando `output_wav` (formato .wav).

    Caminho preferido: daemon socket-based do engine (modelo residente em RAM/VRAM).
    Fallback (apenas Coqui): subprocess `venv_coqui` cold-load a cada chamada.
    Chatterbox sem daemon falha imediatamente (sem fallback implementado).
    """
    _check_environment(engine=engine)
    text = text.strip()
    if not text:
        raise CoquiRunnerError("texto vazio")

    reference_wav = Path(reference_wav).resolve()
    if not reference_wav.is_file():
        raise CoquiRunnerError(f"reference_wav inexistente: {reference_wav}")

    output_wav = Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if USE_DAEMON:
        try:
            from src.tools.tts_daemon import client as daemon_client

            daemon_client.ensure_running(engine=engine)
            return daemon_client.synthesize(
                text=text,
                reference_wav=reference_wav,
                output_wav=output_wav,
                speed=speed,
                language=language,
                engine=engine,
                timeout=timeout,
            )
        except Exception as exc:
            if engine != "coqui":
                # Chatterbox não tem fallback subprocess; falha direto.
                raise CoquiRunnerError(
                    f"Daemon {engine} falhou e não há fallback subprocess: {exc}"
                ) from exc
            logger.warning(
                "Daemon TTS coqui indisponível (%s); usando fallback subprocess.",
                exc,
            )

    # Fallback só pra coqui
    return _synthesize_subprocess(
        text=text,
        reference_wav=reference_wav,
        output_wav=output_wav,
        speed=speed,
        timeout=timeout,
        language=language,
    )


def _synthesize_subprocess(
    text: str,
    reference_wav: Path,
    output_wav: Path,
    speed: float,
    timeout: int,
    language: str,
) -> Path:
    payload = {
        "text": text,
        "reference_wav": str(reference_wav),
        "output_wav": str(output_wav),
        "speed": float(speed),
        "model_dir": str(MODEL_DIR),
        "language": language,
    }

    proc = subprocess.run(
        [str(VENV_COQUI_PY), str(WRAPPER_PATH)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )

    if proc.returncode != 0 or "SUCCESS" not in proc.stdout:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise CoquiRunnerError(
            "Wrapper falhou (rc={}):\n  stdout: {}\n  stderr: {}".format(
                proc.returncode,
                proc.stdout.strip(),
                "\n         ".join(stderr_tail) if stderr_tail else "(vazio)",
            )
        )

    if not output_wav.is_file() or output_wav.stat().st_size < 1024:
        raise CoquiRunnerError(
            f"Saída inesperadamente vazia: {output_wav}"
        )

    return output_wav


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Quebra `text` em chunks <= max_chars, agrupando sentenças completas."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
            # Sentença sozinha pode estourar o limite — aceita, XTTS lida.
        else:
            current.append(sentence)
            current_len += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


def synthesize_chunked(
    text: str,
    reference_wav: Path,
    output_dir: Path,
    speed: float = 1.0,
    max_chars: int = 250,
    timeout_per_chunk: int = 600,
    language: str = "pt",
    engine: str = "coqui",
    on_chunk_start: Callable[[int, int, str], None] | None = None,
    on_chunk_done: Callable[[int, int, Path], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ChunkResult]:
    """Sintetiza um texto longo em múltiplos chunks .wav numerados.

    Cada chunk vira `output_dir/chunk_NN.wav`. Retorna lista de `ChunkResult`.

    Callbacks:
        on_chunk_start(idx, total, chunk_text)
        on_chunk_done(idx, total, wav_path)
        should_stop()  -> bool; se True entre chunks, interrompe.
    """
    _check_environment(engine=engine)
    chunks = _split_into_chunks(text, max_chars)
    if not chunks:
        raise CoquiRunnerError("Texto não produziu chunks (vazio ou inválido)")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ChunkResult] = []
    total = len(chunks)
    for idx, chunk_text in enumerate(chunks, start=1):
        if should_stop and should_stop():
            break
        if on_chunk_start:
            on_chunk_start(idx, total, chunk_text)

        wav_path = output_dir / f"chunk_{idx:03d}.wav"
        synthesize(
            text=chunk_text,
            reference_wav=reference_wav,
            output_wav=wav_path,
            speed=speed,
            timeout=timeout_per_chunk,
            language=language,
            engine=engine,
        )
        result = ChunkResult(index=idx, text=chunk_text, wav_path=wav_path)
        results.append(result)
        if on_chunk_done:
            on_chunk_done(idx, total, wav_path)

    return results


def iter_chunks(text: str, max_chars: int = 250) -> Iterable[str]:
    """Helper público para preview (estima quantos chunks o texto vai gerar)."""
    return _split_into_chunks(text, max_chars)
