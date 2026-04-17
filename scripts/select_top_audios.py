#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import logging
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.modules.clone_voice.core.audio_quality import (
    AudioQualityAnalyzer,
    analyze_and_select_best,
)

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normaliza nome para comparacao: lowercase, sem underscores/espacos."""
    return name.lower().replace("_", "").replace(" ", "")


def find_dataset(entity_name: str, base_dir: Path) -> Path:
    """Busca o dataset mais recente de uma entidade pelo nome."""
    normalized = _normalize_name(entity_name)
    candidates = sorted(
        [
            d
            for d in base_dir.iterdir()
            if d.is_dir()
            and _normalize_name(d.name.split("_")[0] if "_20" in d.name else d.name).startswith(normalized)
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum dataset encontrado para '{entity_name}' em {base_dir}"
        )
    return candidates[0]


def run_selection(
    dataset_path: Path, n_best: int = 10, diverse: bool = True
) -> None:
    """Executa analise, selecao e exportacao dos melhores audios."""
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        logger.error("Dataset nao encontrado: %s", dataset_path)
        sys.exit(1)

    output_path = dataset_path / "top_10_selection"
    wavs_output = output_path / "wavs"

    logger.info("Dataset: %s", dataset_path.name)
    logger.info("Saida: %s", output_path)

    result, selection = analyze_and_select_best(
        dataset_path=dataset_path,
        n_best=n_best,
        diverse=diverse,
    )

    logger.info(
        "Arquivos analisados: %d/%d", result.analyzed_files, result.total_files
    )
    logger.info("Duracao total: %.1fs", result.total_duration_seconds)
    logger.info("Score medio: %.1f", result.avg_quality_score)

    if not selection:
        logger.error("Nenhum audio selecionado")
        sys.exit(1)

    analyzer = AudioQualityAnalyzer()
    analyzer.metrics = list(result.metrics)
    analyzer._calculate_quality_scores()

    exports = analyzer.export_selection(selection, output_path)

    wavs_output.mkdir(parents=True, exist_ok=True)
    for m in selection:
        src = m.file_path
        dst = wavs_output / src.name
        shutil.copy2(str(src), str(dst))
        logger.info(
            "  Copiado: %s (score=%.1f, dur=%.1fs)",
            src.name,
            m.quality_score,
            m.duration_seconds,
        )

    unified = analyzer.create_unified_reference(selection, output_path)
    if unified:
        logger.info("Audio unificado: %s", unified.name)

    total_sel_duration = sum(m.duration_seconds for m in selection)
    avg_sel_score = sum(m.quality_score for m in selection) / len(selection)

    logger.info("")
    logger.info("=== Resumo da Selecao ===")
    logger.info("Selecionados: %d audios", len(selection))
    logger.info("Duracao total: %.1fs", total_sel_duration)
    logger.info("Score medio: %.1f", avg_sel_score)
    logger.info("Arquivos exportados:")
    for name, path in exports.items():
        logger.info("  %s: %s", name, path.name)
    if unified:
        logger.info("  unified_reference: %s", unified.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona os melhores audios de um dataset de voz"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        type=str,
        help="Caminho direto para o diretorio do dataset",
    )
    group.add_argument(
        "--entity",
        type=str,
        help="Nome da entidade (busca em data_output/clone_voice/)",
    )

    parser.add_argument(
        "--n-best",
        type=int,
        default=10,
        help="Numero de audios a selecionar (default: 10)",
    )
    parser.add_argument(
        "--no-diverse",
        action="store_true",
        help="Desativa selecao diversificada (usa apenas score)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_output/clone_voice",
        help="Diretorio base dos datasets",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Ativa logging detalhado",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
    else:
        base_dir = (ROOT_DIR / args.output_dir).resolve()
        dataset_path = find_dataset(args.entity, base_dir)
        logger.info("Dataset encontrado: %s", dataset_path.name)

    run_selection(
        dataset_path=dataset_path,
        n_best=args.n_best,
        diverse=not args.no_diverse,
    )


if __name__ == "__main__":
    main()

# "A qualidade nunca e um acidente; ela e sempre o resultado de um esforco inteligente." -- John Ruskin
