#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empacota datasets de voz em tarballs para distribuição via GitHub Releases.

Cria tarballs individuais por entidade e um bundle com todas,
calcula SHA256 de cada arquivo e gera RELEASE_NOTES.md.

Uso:
    python scripts/package_release.py
    python scripts/package_release.py --version v2.0.0
    python scripts/package_release.py --output-dir releases/beta/
"""

import argparse
import hashlib
import json
import logging
import sys
import tarfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).parent.parent.resolve()

ENTITY_MAP: Dict[str, str] = {
    "Luna Gothic": "luna",
    "Mars": "mars",
    "Juno": "juno",
    "Eris": "eris",
    "Somn": "somn",
    "Lars": "lars",
}

REQUIRED_FILES: List[str] = [
    "metadata.json",
    "metadata.csv",
    "chatterbox_data.jsonl",
    "coqui_manifest.json",
]

REQUIRED_DIRS: List[str] = [
    "wavs",
    "top_10_selection",
]

DATA_OUTPUT_DIR = ROOT_DIR / "data_output" / "clone_voice"

logger = logging.getLogger("package_release")


def _setup_logging(verbose: bool = False) -> None:
    """Configura logging com RotatingFileHandler e saída no console."""
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    file_handler = RotatingFileHandler(
        log_dir / "package_release.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s",
    ))
    console_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def _normalize_name(name: str) -> str:
    """Normaliza nome para comparação: lowercase, sem underscores/espaços."""
    return name.lower().replace("_", "").replace(" ", "")


def find_entity_datasets(base_dir: Path) -> Dict[str, Path]:
    """
    Encontra o dataset mais recente de cada entidade.

    Retorna dicionário {nome_entidade: caminho_dataset}.
    Se múltiplos datasets existem para a mesma entidade, usa o mais recente
    (ordenação lexicográfica do timestamp no nome do diretório).
    """
    if not base_dir.exists():
        logger.error("Diretório de datasets não encontrado: %s", base_dir)
        return {}

    found: Dict[str, Path] = {}

    for entity_name in ENTITY_MAP:
        normalized = _normalize_name(entity_name)
        candidates = sorted(
            [
                d
                for d in base_dir.iterdir()
                if d.is_dir()
                and _normalize_name(
                    d.name.rsplit("_20", 1)[0] if "_20" in d.name else d.name
                ) == normalized
            ],
            key=lambda p: p.name,
            reverse=True,
        )

        if candidates:
            found[entity_name] = candidates[0]
            if len(candidates) > 1:
                logger.info(
                    "Múltiplos datasets para '%s' -- usando mais recente: %s",
                    entity_name,
                    candidates[0].name,
                )
        else:
            logger.warning("Nenhum dataset encontrado para '%s'", entity_name)

    return found


def validate_dataset(dataset_path: Path) -> Tuple[bool, List[str]]:
    """
    Valida que o dataset contém todos os arquivos obrigatórios.

    Retorna (válido, lista_de_erros).
    """
    errors: List[str] = []

    for required_file in REQUIRED_FILES:
        if not (dataset_path / required_file).exists():
            errors.append(f"Arquivo ausente: {required_file}")

    for required_dir in REQUIRED_DIRS:
        dir_path = dataset_path / required_dir
        if not dir_path.exists():
            errors.append(f"Diretório ausente: {required_dir}")
        elif not any(dir_path.iterdir()):
            errors.append(f"Diretório vazio: {required_dir}")

    return len(errors) == 0, errors


def load_metadata(dataset_path: Path) -> Optional[dict]:
    """Carrega metadata.json do dataset."""
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Erro ao ler metadata de %s: %s", dataset_path.name, e)
        return None


def load_selection_info(dataset_path: Path) -> Optional[dict]:
    """Carrega selection_info.json do top_10_selection."""
    info_path = dataset_path / "top_10_selection" / "selection_info.json"
    if not info_path.exists():
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Erro ao ler selection_info de %s: %s", dataset_path.name, e)
        return None


def create_entity_tarball(
    entity_name: str,
    dataset_path: Path,
    output_dir: Path,
    version: str,
) -> Optional[Path]:
    """
    Cria tarball individual para uma entidade.

    Estrutura interna:
        {slug}-voices-{version}/
            wavs/
            top_10_selection/
            metadata.json
            metadata.csv
            chatterbox_data.jsonl
            coqui_manifest.json
    """
    slug = ENTITY_MAP[entity_name]
    tarball_name = f"{slug}-voices-{version}.tar.gz"
    tarball_path = output_dir / tarball_name
    archive_prefix = f"{slug}-voices-{version}"

    logger.info("Empacotando '%s' -> %s", entity_name, tarball_name)

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            for required_file in REQUIRED_FILES:
                file_path = dataset_path / required_file
                if file_path.exists():
                    arcname = f"{archive_prefix}/{required_file}"
                    tar.add(str(file_path), arcname=arcname)

            wavs_dir = dataset_path / "wavs"
            if wavs_dir.exists():
                for audio_file in sorted(wavs_dir.iterdir()):
                    if audio_file.is_file():
                        arcname = f"{archive_prefix}/wavs/{audio_file.name}"
                        tar.add(str(audio_file), arcname=arcname)

            top10_dir = dataset_path / "top_10_selection"
            if top10_dir.exists():
                _add_directory_recursive(tar, top10_dir, f"{archive_prefix}/top_10_selection")

        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        logger.info("  Criado: %s (%.1f MB)", tarball_name, size_mb)
        return tarball_path

    except OSError as e:
        logger.error("Erro ao criar tarball para '%s': %s", entity_name, e)
        return None


def _add_directory_recursive(tar: tarfile.TarFile, dir_path: Path, arcname_prefix: str) -> None:
    """Adiciona conteúdo de um diretório recursivamente ao tarball."""
    for item in sorted(dir_path.iterdir()):
        arcname = f"{arcname_prefix}/{item.name}"
        if item.is_file():
            tar.add(str(item), arcname=arcname)
        elif item.is_dir():
            _add_directory_recursive(tar, item, arcname)


def create_bundle_tarball(
    entity_tarballs: Dict[str, Path],
    datasets: Dict[str, Path],
    output_dir: Path,
    version: str,
) -> Optional[Path]:
    """
    Cria tarball bundle contendo todos os diretórios de entidades.

    Estrutura interna:
        all-voices-{version}/
            luna-voices-{version}/
                ...
            mars-voices-{version}/
                ...
    """
    tarball_name = f"all-voices-{version}.tar.gz"
    tarball_path = output_dir / tarball_name
    bundle_prefix = f"all-voices-{version}"

    logger.info("Criando bundle: %s", tarball_name)

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            for entity_name, dataset_path in sorted(datasets.items()):
                slug = ENTITY_MAP[entity_name]
                entity_prefix = f"{bundle_prefix}/{slug}-voices-{version}"

                for required_file in REQUIRED_FILES:
                    file_path = dataset_path / required_file
                    if file_path.exists():
                        arcname = f"{entity_prefix}/{required_file}"
                        tar.add(str(file_path), arcname=arcname)

                wavs_dir = dataset_path / "wavs"
                if wavs_dir.exists():
                    for audio_file in sorted(wavs_dir.iterdir()):
                        if audio_file.is_file():
                            arcname = f"{entity_prefix}/wavs/{audio_file.name}"
                            tar.add(str(audio_file), arcname=arcname)

                top10_dir = dataset_path / "top_10_selection"
                if top10_dir.exists():
                    _add_directory_recursive(tar, top10_dir, f"{entity_prefix}/top_10_selection")

        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        logger.info("  Bundle criado: %s (%.1f MB)", tarball_name, size_mb)
        return tarball_path

    except OSError as e:
        logger.error("Erro ao criar bundle: %s", e)
        return None


def calculate_checksums(tarball_paths: List[Path]) -> Dict[str, str]:
    """Calcula SHA256 de cada tarball."""
    checksums: Dict[str, str] = {}

    for tarball_path in tarball_paths:
        sha256 = hashlib.sha256()
        try:
            with open(tarball_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            checksums[tarball_path.name] = sha256.hexdigest()
            logger.debug("SHA256 %s: %s", tarball_path.name, checksums[tarball_path.name])
        except OSError as e:
            logger.error("Erro ao calcular SHA256 de %s: %s", tarball_path.name, e)

    return checksums


def write_checksums_file(checksums: Dict[str, str], output_dir: Path) -> Path:
    """Escreve CHECKSUMS.txt com SHA256 de cada tarball."""
    checksums_path = output_dir / "CHECKSUMS.txt"
    lines: List[str] = []

    for filename in sorted(checksums.keys()):
        lines.append(f"{checksums[filename]}  {filename}")

    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("CHECKSUMS.txt gerado com %d entradas", len(checksums))
    return checksums_path


def generate_release_notes(
    datasets: Dict[str, Path],
    checksums: Dict[str, str],
    version: str,
    output_dir: Path,
) -> Path:
    """
    Gera RELEASE_NOTES.md com informações dos datasets.

    Conteúdo:
    - Resumo da release
    - Tabela de entidades com contagem de frases e scores
    - Parâmetros de geração (modelo, voice settings)
    - Instruções de uso dos áudios de referência
    - Checksums SHA256
    """
    notes_path = output_dir / "RELEASE_NOTES.md"
    sections: List[str] = []

    sections.append(f"# Neurosonancy Voice Dataset {version}\n")
    sections.append(f"Data de publicação: {datetime.now().strftime('%Y-%m-%d')}\n")

    sections.append("## Entidades\n")
    sections.append("| Entidade | Frases | Geradas | Score Médio (Top 10) | Duração Total |")
    sections.append("|----------|--------|---------|----------------------|---------------|")

    entity_details: List[str] = []
    total_phrases = 0
    total_generated = 0

    for entity_name in sorted(datasets.keys()):
        dataset_path = datasets[entity_name]
        metadata = load_metadata(dataset_path)
        selection_info = load_selection_info(dataset_path)

        if metadata is None:
            continue

        stats = metadata.get("stats", {})
        requested = stats.get("total_requested", 0)
        generated = stats.get("total_generated", 0)
        duration_ms = stats.get("total_duration_ms", 0)
        duration_str = f"{duration_ms / 1000:.1f}s"

        total_phrases += requested
        total_generated += generated

        avg_score = "N/A"
        if selection_info:
            score_val = selection_info.get("avg_quality_score", 0)
            avg_score = f"{score_val:.1f}"

        slug = ENTITY_MAP[entity_name]
        sections.append(
            f"| {entity_name} ({slug}) | {requested} | {generated} | {avg_score} | {duration_str} |"
        )

        detail_lines: List[str] = []
        detail_lines.append(f"### {entity_name}\n")
        detail_lines.append(f"- **Modelo**: `{metadata.get('model_id', 'N/A')}`")
        detail_lines.append(f"- **Voice ID**: `{metadata.get('voice_id', 'N/A')}`")
        detail_lines.append(f"- **Frases geradas**: {generated}/{requested}")
        detail_lines.append(f"- **Duração total**: {duration_str}")

        cat_settings = metadata.get("category_voice_settings", {})
        if cat_settings:
            detail_lines.append(f"- **Categorias**: {len(cat_settings)}")
            for cat_name, settings in cat_settings.items():
                stability = settings.get("stability", 0)
                similarity = settings.get("similarity_boost", 0)
                style = settings.get("style", 0)
                detail_lines.append(
                    f"  - `{cat_name}`: stability={stability:.2f}, "
                    f"similarity={similarity:.2f}, style={style:.2f}"
                )

        if selection_info:
            sel_total = selection_info.get("total_selected", 0)
            sel_duration = selection_info.get("total_duration_seconds", 0)
            sel_score = selection_info.get("avg_quality_score", 0)
            detail_lines.append(
                f"- **Top {sel_total} selecionados**: "
                f"duração={sel_duration:.1f}s, score médio={sel_score:.1f}"
            )

        entity_details.append("\n".join(detail_lines))

    sections.append("")
    sections.append(f"**Total**: {total_generated}/{total_phrases} frases geradas\n")

    sections.append("## Detalhes por Entidade\n")
    sections.extend(entity_details)

    sections.append("\n## Estrutura dos Pacotes\n")
    sections.append("Cada tarball individual contém:\n")
    sections.append("```")
    sections.append("{entidade}-voices-" + version + "/")
    sections.append("  wavs/                        # Todos os áudios gerados")
    sections.append("  top_10_selection/             # Top 10 selecionados + unified_reference.wav")
    sections.append("  metadata.json                # Scores, parâmetros de geração, timestamps")
    sections.append("  metadata.csv                 # Formato LJSpeech")
    sections.append("  chatterbox_data.jsonl         # Formato Chatterbox fine-tuning")
    sections.append("  coqui_manifest.json           # Formato Coqui TTS")
    sections.append("```\n")

    sections.append("## Uso dos Áudios de Referência\n")
    sections.append(
        "O diretório `top_10_selection/` contém os 10 melhores áudios selecionados "
        "por qualidade e diversidade, além do arquivo `unified_reference.wav` que "
        "concatena todos os selecionados em um único áudio.\n"
    )
    sections.append("Para fine-tuning com Chatterbox:\n")
    sections.append("```bash")
    sections.append(f"tar xzf luna-voices-{version}.tar.gz")
    sections.append(f"cd luna-voices-{version}/top_10_selection/")
    sections.append("# Use unified_reference.wav como referência de voz")
    sections.append("# Use chatterbox_data.jsonl para o treinamento")
    sections.append("```\n")
    sections.append("Para fine-tuning com Coqui TTS:\n")
    sections.append("```bash")
    sections.append(f"tar xzf luna-voices-{version}.tar.gz")
    sections.append(f"cd luna-voices-{version}/")
    sections.append("# Use coqui_manifest.json como manifesto do dataset")
    sections.append("# Os áudios estão em wavs/")
    sections.append("```\n")

    sections.append("## Verificação de Integridade\n")
    sections.append("```bash")
    sections.append("sha256sum -c CHECKSUMS.txt")
    sections.append("```\n")

    sections.append("### Checksums SHA256\n")
    sections.append("```")
    for filename in sorted(checksums.keys()):
        sections.append(f"{checksums[filename]}  {filename}")
    sections.append("```\n")

    notes_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("RELEASE_NOTES.md gerado")
    return notes_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Empacota datasets de voz em tarballs para GitHub Release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s
  %(prog)s --version v2.0.0
  %(prog)s --output-dir releases/beta/ --verbose
        """,
    )

    parser.add_argument(
        "--version",
        default="v1.0.0",
        help="Versão da release (padrão: v1.0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório de saída (padrão: releases/{version}/)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Diretório base dos datasets (padrão: data_output/clone_voice/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ativa logs detalhados (DEBUG)",
    )

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    version: str = args.version
    data_dir = Path(args.data_dir) if args.data_dir else DATA_OUTPUT_DIR
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT_DIR / output_dir
    else:
        output_dir = ROOT_DIR / "releases" / version

    logger.info("Neurosonancy Voice Packager")
    logger.info("Versão: %s", version)
    logger.info("Dados: %s", data_dir)
    logger.info("Saída: %s", output_dir)

    datasets = find_entity_datasets(data_dir)
    if not datasets:
        logger.error("Nenhum dataset encontrado em %s", data_dir)
        return 1

    logger.info("Datasets encontrados: %d/%d entidades", len(datasets), len(ENTITY_MAP))

    valid_datasets: Dict[str, Path] = {}
    has_failures = False

    for entity_name, dataset_path in datasets.items():
        is_valid, errors = validate_dataset(dataset_path)
        if is_valid:
            valid_datasets[entity_name] = dataset_path
            logger.info("  [OK] %s (%s)", entity_name, dataset_path.name)
        else:
            has_failures = True
            logger.error("  [FALHA] %s (%s)", entity_name, dataset_path.name)
            for err in errors:
                logger.error("    - %s", err)

    if not valid_datasets:
        logger.error("Nenhum dataset válido para empacotar")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    entity_tarballs: Dict[str, Path] = {}
    all_tarballs: List[Path] = []

    for entity_name, dataset_path in sorted(valid_datasets.items()):
        tarball_path = create_entity_tarball(
            entity_name=entity_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            version=version,
        )
        if tarball_path:
            entity_tarballs[entity_name] = tarball_path
            all_tarballs.append(tarball_path)
        else:
            has_failures = True

    if entity_tarballs:
        bundle_path = create_bundle_tarball(
            entity_tarballs=entity_tarballs,
            datasets=valid_datasets,
            output_dir=output_dir,
            version=version,
        )
        if bundle_path:
            all_tarballs.append(bundle_path)
        else:
            has_failures = True

    if not all_tarballs:
        logger.error("Nenhum tarball criado")
        return 1

    checksums = calculate_checksums(all_tarballs)
    write_checksums_file(checksums, output_dir)

    generate_release_notes(
        datasets=valid_datasets,
        checksums=checksums,
        version=version,
        output_dir=output_dir,
    )

    logger.info("")
    logger.info("=== Resumo ===")
    logger.info("Tarballs individuais: %d", len(entity_tarballs))
    logger.info("Bundle: %s", "sim" if len(all_tarballs) > len(entity_tarballs) else "não")
    logger.info("Saída: %s", output_dir)

    for tarball_path in sorted(all_tarballs, key=lambda p: p.name):
        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        logger.info("  %s (%.1f MB)", tarball_path.name, size_mb)

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())

# "O segredo da liberdade reside na coragem." -- Pericles
