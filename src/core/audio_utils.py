"""Utilitários de áudio compartilhados (concat WAV → MP3, duração)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


class AudioUtilsError(RuntimeError):
    """Erro de operação de áudio (ffmpeg/ffprobe)."""


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise AudioUtilsError(f"binário '{name}' não encontrado no PATH")
    return binary


def concat_wavs_to_mp3(
    parts: Sequence[Path],
    output_mp3: Path,
    bitrate: str = "192k",
) -> Path:
    """Concatena vários WAV em um único MP3 via `ffmpeg -f concat`.

    Todos os WAVs precisam ter o mesmo sample rate / canais — caso típico
    do XTTS v2 (24 kHz mono). Output é re-encodado em MP3 com `libmp3lame`
    no bitrate especificado.
    """
    ffmpeg = _require_binary("ffmpeg")
    if not parts:
        raise AudioUtilsError("lista de parts vazia")

    resolved: list[Path] = []
    for part in parts:
        p = Path(part).resolve()
        if not p.is_file():
            raise AudioUtilsError(f"parte inexistente: {p}")
        resolved.append(p)

    output_mp3 = Path(output_mp3).resolve()
    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as fh:
        list_file = Path(fh.name)
        for p in resolved:
            # ffmpeg concat exige aspas simples; precisamos escapar
            # ocorrência interna duplicando-as.
            safe = str(p).replace("'", r"'\''")
            fh.write(f"file '{safe}'\n")

    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                str(output_mp3),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AudioUtilsError(
                f"ffmpeg falhou (rc={proc.returncode}): {proc.stderr.strip()}"
            )
    finally:
        list_file.unlink(missing_ok=True)

    if not output_mp3.is_file() or output_mp3.stat().st_size < 512:
        raise AudioUtilsError(f"saída ausente ou vazia: {output_mp3}")

    return output_mp3


def get_audio_duration(path: Path) -> float:
    """Retorna duração em segundos via `ffprobe`."""
    ffprobe = _require_binary("ffprobe")
    p = Path(path).resolve()
    if not p.is_file():
        raise AudioUtilsError(f"arquivo inexistente: {p}")

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(p),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AudioUtilsError(
            f"ffprobe falhou (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    text = (proc.stdout or "").strip()
    try:
        return float(text)
    except ValueError as exc:
        raise AudioUtilsError(f"duração inválida: '{text}'") from exc
