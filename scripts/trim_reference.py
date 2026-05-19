#!/usr/bin/env python3
"""Recorta `unified_reference.wav` em uma janela contínua de 15–30 s
(formato ideal pros engines XTTS / Chatterbox).

Saída por entidade:
    data_output/clone_voice/<Entity>_<ts>/top_10_selection/
        unified_reference.wav     (original, mantido como fallback)
        coqui_reference.wav       (recortado; preferido pelo Coqui)
        chatterbox_reference.wav  (recortado; preferido pelo Chatterbox)

No MVP os dois arquivos resultantes contêm o mesmo conteúdo; manter os dois
permite no futuro otimizar engines separadamente.

Uso:
    python scripts/trim_reference.py                       # todas as entidades
    python scripts/trim_reference.py --entity eris         # uma só
    python scripts/trim_reference.py --target 22 --force   # 22s, regrava
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Permite rodar `python scripts/trim_reference.py` com o venv principal.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

ENTITY_PROFILES = ROOT_DIR / "data_input" / "entity_profiles.json"
DATASETS_ROOT = ROOT_DIR / "data_output" / "clone_voice"

DEFAULT_MIN_SECONDS = 15.0
DEFAULT_MAX_SECONDS = 30.0
DEFAULT_TARGET_SECONDS = 22.0

# Parâmetros do VAD (pydub.silence). Calibrados pra fala humana em
# referências de 22050 Hz vindas de TTS sintético (ElevenLabs) com dinâmica
# achatada. Valores mais permissivos do que pra fala real, pra não fragmentar.
SILENCE_THRESH_DB = -40  # abaixo de -40 dBFS é silêncio (mais permissivo)
MIN_SILENCE_MS = 700  # só pausas >700ms separam segmentos (respiros ficam)
GAP_MERGE_MS = 1800  # une segmentos com gap <1.8s (frases vizinhas)

FADE_MS = 50

logger = logging.getLogger("trim_reference")


def _latest_dataset_for(entity_display_name: str) -> Path | None:
    if not DATASETS_ROOT.is_dir():
        return None
    pattern = f"{entity_display_name}_*"
    candidates = sorted(
        (d for d in DATASETS_ROOT.glob(pattern) if d.is_dir()),
        key=lambda d: (d / "top_10_selection").stat().st_mtime
        if (d / "top_10_selection").is_dir()
        else 0,
        reverse=True,
    )
    for c in candidates:
        if (c / "top_10_selection").is_dir():
            return c / "top_10_selection"
    return None


def _merge_segments(
    segments: list[tuple[int, int]], gap_ms: int
) -> list[tuple[int, int]]:
    """Une segmentos consecutivos cuja distância é menor que `gap_ms`."""
    if not segments:
        return []
    merged: list[list[int]] = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= gap_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _pick_window(
    merged: list[tuple[int, int]],
    target_ms: int,
    min_ms: int,
    max_ms: int,
) -> tuple[int, int]:
    """Escolhe a melhor janela contígua de fala.

    Estratégia:
      1. Se algum segmento já cair entre [min, max], usa o mais próximo de
         `target_ms` em duração.
      2. Senão, escolhe o maior segmento e recorta até `target_ms`.
    """
    if not merged:
        raise RuntimeError(
            "VAD não encontrou fala — silêncio total ou parâmetros muito agressivos."
        )

    valid = [(s, e) for s, e in merged if min_ms <= (e - s) <= max_ms]
    if valid:
        # Mais próximo do alvo
        best = min(valid, key=lambda se: abs((se[1] - se[0]) - target_ms))
        return best

    # Sem janela ideal — pega o maior segmento e corta
    longest = max(merged, key=lambda se: se[1] - se[0])
    start = longest[0]
    end = min(start + target_ms, longest[1])
    if end - start < min_ms:
        # Segmento mais longo é menor que o mínimo desejado — usa ele inteiro
        end = longest[1]
    return (start, end)


def trim_for(
    entity_key: str,
    entity_display: str,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    force: bool,
) -> tuple[str, dict]:
    """Recorta a reference de uma entidade. Retorna (status, info_dict)."""
    dataset = _latest_dataset_for(entity_display)
    if dataset is None:
        return ("skip", {"reason": f"sem dataset clone_voice para {entity_display}"})

    src = dataset / "unified_reference.wav"
    if not src.is_file():
        return ("skip", {"reason": f"{src} ausente"})

    coqui_out = dataset / "coqui_reference.wav"
    chatterbox_out = dataset / "chatterbox_reference.wav"

    # Idempotente: skipa se ambos existem e têm tamanho razoável
    if (
        not force
        and coqui_out.is_file()
        and chatterbox_out.is_file()
        and coqui_out.stat().st_size > 100_000
        and chatterbox_out.stat().st_size > 100_000
    ):
        return ("ok", {
            "reason": "já existe (use --force para regerar)",
            "coqui": str(coqui_out),
            "chatterbox": str(chatterbox_out),
        })

    audio = AudioSegment.from_wav(str(src))
    src_duration_s = len(audio) / 1000.0

    segments = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )
    merged = _merge_segments(segments, gap_ms=GAP_MERGE_MS)

    target_ms = int(target_seconds * 1000)
    min_ms = int(min_seconds * 1000)
    max_ms = int(max_seconds * 1000)

    start_ms, end_ms = _pick_window(merged, target_ms, min_ms, max_ms)
    trimmed = audio[start_ms:end_ms].fade_in(FADE_MS).fade_out(FADE_MS)

    trimmed.export(str(coqui_out), format="wav")
    trimmed.export(str(chatterbox_out), format="wav")

    return ("ok", {
        "src_duration_s": round(src_duration_s, 2),
        "trim_duration_s": round((end_ms - start_ms) / 1000.0, 2),
        "trim_window_ms": (start_ms, end_ms),
        "segments_found": len(segments),
        "segments_merged": len(merged),
        "coqui": str(coqui_out),
        "chatterbox": str(chatterbox_out),
    })


def _load_entities() -> list[tuple[str, str]]:
    if not ENTITY_PROFILES.is_file():
        raise FileNotFoundError(f"{ENTITY_PROFILES} não encontrado")
    with ENTITY_PROFILES.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [(key, entry.get("name", key)) for key, entry in raw.items()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recorta unified_reference.wav em 15–30 s pros engines TTS.",
    )
    parser.add_argument(
        "--entity",
        default="todas",
        help="chave da entidade (eris, mars...) ou 'todas' (default)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET_SECONDS,
        help=f"duração-alvo do recorte em segundos (default {DEFAULT_TARGET_SECONDS})",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        dest="min_seconds",
        help=f"duração mínima aceitável (default {DEFAULT_MIN_SECONDS})",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        dest="max_seconds",
        help=f"duração máxima aceitável (default {DEFAULT_MAX_SECONDS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regrava mesmo se coqui/chatterbox_reference.wav já existem",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    entities = _load_entities()
    if args.entity != "todas":
        entities = [e for e in entities if e[0] == args.entity]
        if not entities:
            print(f"Entidade '{args.entity}' não encontrada.", file=sys.stderr)
            return 1

    print(f"{'Entidade':14s} {'orig':>8s} {'corte':>8s}  status")
    print("-" * 70)
    failures = 0
    for key, display in entities:
        status, info = trim_for(
            entity_key=key,
            entity_display=display,
            target_seconds=args.target,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            force=args.force,
        )
        if status == "ok":
            src = info.get("src_duration_s", "—")
            trim = info.get("trim_duration_s", "—")
            extra = info.get("reason", "")
            print(
                f"{display:14s} {str(src):>8s} {str(trim):>8s}  OK {extra}".rstrip()
            )
        elif status == "skip":
            print(f"{display:14s} {'—':>8s} {'—':>8s}  SKIP {info['reason']}")
        else:
            failures += 1
            print(f"{display:14s} {'—':>8s} {'—':>8s}  ERRO {info}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
