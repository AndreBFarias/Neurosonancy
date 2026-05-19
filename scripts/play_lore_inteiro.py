#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Narração da Éris para LORE_INTEIRO.md.

Lê os bullets do arquivo na raiz, agrupa em chunks <= 4500 chars,
gera mp3 via ElevenLabs (turbo_v2_5), concatena via ffmpeg e toca via ffplay.

Uso:
    python scripts/play_lore_inteiro.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

LORE_FILE = ROOT / "LORE_INTEIRO.md"
OUTPUT_DIR = ROOT / "data_output" / "lore_inteiro"

VOICE_ID = "FXrW2eDeVmtaCkzUWW7X"
MODEL_ID = "eleven_turbo_v2_5"
VOICE_SETTINGS = {
    "stability": 0.40,
    "similarity_boost": 0.82,
    "style": 0.70,
    "use_speaker_boost": True,
    "speed": 1.15,
}
MAX_CHARS_PER_CHUNK = 4500
API_TIMEOUT = 180


def parse_bullets(md_path: Path) -> List[str]:
    bullets: List[str] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def chunk_bullets(bullets: List[str], limit: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for bullet in bullets:
        projected = current_len + len(bullet) + (1 if current else 0)
        if projected > limit and current:
            chunks.append(" ".join(current))
            current = [bullet]
            current_len = len(bullet)
        else:
            current.append(bullet)
            current_len = projected
    if current:
        chunks.append(" ".join(current))
    return chunks


def synth(text: str, out_path: Path, api_key: str) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    params = {"output_format": "mp3_44100_128"}
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }
    response = requests.post(
        url, params=params, headers=headers, json=payload, timeout=API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs {response.status_code}: {response.text[:300]}"
        )
    out_path.write_bytes(response.content)


def concat_mp3s(parts: List[Path], out_path: Path) -> None:
    list_file = out_path.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in parts),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy", str(out_path),
            ],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def play(path: Path) -> None:
    if shutil.which("ffplay") is None:
        print(f"[lore] ffplay ausente. Áudio salvo em: {path}")
        return
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
    )


def main() -> int:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[lore] ELEVENLABS_API_KEY ausente em .env", file=sys.stderr)
        return 1
    if not LORE_FILE.exists():
        print(f"[lore] arquivo não encontrado: {LORE_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bullets = parse_bullets(LORE_FILE)
    if not bullets:
        print("[lore] nenhum bullet '- ' encontrado em LORE_INTEIRO.md", file=sys.stderr)
        return 1

    chunks = chunk_bullets(bullets, MAX_CHARS_PER_CHUNK)
    total_chars = sum(len(c) for c in chunks)
    print(
        f"[lore] {len(bullets)} bullets -> {len(chunks)} chunk(s) | {total_chars} chars totais"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts: List[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        part_path = OUTPUT_DIR / f"part_{timestamp}_{index:02d}.mp3"
        print(
            f"[lore] chunk {index}/{len(chunks)} ({len(chunk)} chars) -> {part_path.name}"
        )
        synth(chunk, part_path, api_key)
        parts.append(part_path)
        if index < len(chunks):
            time.sleep(0.4)

    final = OUTPUT_DIR / f"lore_inteiro_{timestamp}.mp3"
    if len(parts) == 1:
        parts[0].rename(final)
    else:
        concat_mp3s(parts, final)
        for part in parts:
            part.unlink(missing_ok=True)

    print(f"[lore] pronto: {final}")
    play(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
