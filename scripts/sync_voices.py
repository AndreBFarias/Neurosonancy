#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sincroniza arquivos unified_reference.wav dos datasets do Neurosonancy
para os diretórios de voz do projeto Luna.
"""

import sys
import os
import re
import hashlib
import shutil
import logging
import argparse
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

handler = RotatingFileHandler(
    LOGS_DIR / "sync_voices.log", maxBytes=2_000_000, backupCount=3
)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("sync_voices")

ENTITY_MAP: dict[str, str] = {
    "luna gothic": "luna",
    "mars": "mars",
    "juno": "juno",
    "eris": "eris",
    "somn": "somn",
    "lars": "lars",
}

TIMESTAMP_PATTERN = re.compile(r"_\d{8}_\d{6}$")

VOICE_ENGINES = ("coqui", "chatterbox")


def _sha256(filepath: Path) -> str:
    """Calcula hash SHA256 de um arquivo."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_entity_key(dirname: str) -> str:
    """Extrai a chave da entidade a partir do nome do diretório do dataset.

    Remove o sufixo de timestamp (_YYYYMMDD_HHMMSS) e retorna em lowercase.
    """
    clean = TIMESTAMP_PATTERN.sub("", dirname)
    return clean.lower().strip()


def _resolve_luna_name(entity_key: str) -> str:
    """Resolve o nome da entidade para o padrão do Luna.

    Usa o mapeamento conhecido; para entidades desconhecidas,
    retorna a primeira palavra em lowercase.
    """
    if entity_key in ENTITY_MAP:
        return ENTITY_MAP[entity_key]

    first_word = entity_key.split()[0] if " " in entity_key else entity_key
    logger.warning(
        "Entidade '%s' não encontrada no mapeamento, usando '%s'",
        entity_key,
        first_word,
    )
    return first_word


def discover_entities(base_dir: Path) -> dict[str, Path]:
    """Descobre entidades disponíveis nos datasets.

    Retorna dict mapeando luna_name -> caminho do unified_reference.wav.
    Para entidades duplicadas, mantém o dataset mais recente (ordenação pelo nome).
    """
    if not base_dir.is_dir():
        logger.error("Diretório de datasets não encontrado: %s", base_dir)
        return {}

    entities: dict[str, Path] = {}
    candidates: dict[str, list[Path]] = {}

    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue

        ref_file = entry / "top_10_selection" / "unified_reference.wav"
        if not ref_file.is_file():
            logger.warning(
                "unified_reference.wav ausente em %s, ignorando", entry.name
            )
            continue

        entity_key = _extract_entity_key(entry.name)
        luna_name = _resolve_luna_name(entity_key)

        if luna_name not in candidates:
            candidates[luna_name] = []
        candidates[luna_name].append(ref_file)

    for luna_name, paths in candidates.items():
        paths.sort(key=lambda p: p.parent.parent.name, reverse=True)
        entities[luna_name] = paths[0]
        if len(paths) > 1:
            logger.warning(
                "Múltiplos datasets para '%s', usando o mais recente: %s",
                luna_name,
                paths[0].parent.parent.name,
            )

    return entities


def sync_entity(
    luna_name: str,
    source: Path,
    luna_path: Path,
) -> dict[str, str]:
    """Sincroniza o arquivo de referência de uma entidade para o Luna.

    Retorna dict com status por engine.
    """
    source_hash = _sha256(source)
    results: dict[str, str] = {}

    for engine in VOICE_ENGINES:
        dest_dir = (
            luna_path
            / "src"
            / "assets"
            / "panteao"
            / "entities"
            / luna_name
            / "voice"
            / engine
        )
        dest_file = dest_dir / "reference.wav"

        if dest_file.is_file():
            dest_hash = _sha256(dest_file)
            if dest_hash == source_hash:
                logger.info(
                    "[%s/%s] sem alterações (SHA256 confere)", luna_name, engine
                )
                results[engine] = "sem alterações"
                continue

            logger.info(
                "[%s/%s] arquivo existente difere, atualizando", luna_name, engine
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_file)

        verify_hash = _sha256(dest_file)
        if verify_hash != source_hash:
            logger.error(
                "[%s/%s] falha na verificação de integridade: "
                "esperado=%s obtido=%s",
                luna_name,
                engine,
                source_hash,
                verify_hash,
            )
            results[engine] = "FALHA"
            continue

        logger.info(
            "[%s/%s] copiado com sucesso (SHA256: %s)",
            luna_name,
            engine,
            source_hash[:16],
        )
        results[engine] = "copiado"

    return results


def print_summary(summary: dict[str, dict[str, str]]) -> None:
    """Exibe tabela de resumo da sincronização."""
    col_entity = 12
    col_engine = 12
    col_status = 18

    header = (
        f"{'Entidade':<{col_entity}} "
        f"{'Engine':<{col_engine}} "
        f"{'Status':<{col_status}}"
    )
    separator = "-" * len(header)

    lines = [
        "",
        "Resultado da sincronização",
        separator,
        header,
        separator,
    ]

    for entity, engines in sorted(summary.items()):
        for engine, status in sorted(engines.items()):
            lines.append(
                f"{entity:<{col_entity}} "
                f"{engine:<{col_engine}} "
                f"{status:<{col_status}}"
            )

    lines.append(separator)
    report = "\n".join(lines)
    logger.info(report)
    sys.stdout.write(report + "\n")


def resolve_luna_path(cli_arg: Optional[str]) -> Path:
    """Resolve o caminho do projeto Luna por prioridade:
    1. Argumento CLI --luna-path
    2. Variável de ambiente LUNA_PROJECT_PATH
    3. Fallback para ../Luna relativo à raiz do Neurosonancy
    """
    if cli_arg:
        return Path(cli_arg).resolve()

    env_val = os.environ.get("LUNA_PROJECT_PATH")
    if env_val:
        return Path(env_val).resolve()

    return (ROOT_DIR / ".." / "Luna").resolve()


def main() -> int:
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Sincroniza referências de voz do Neurosonancy para o Luna"
    )
    parser.add_argument(
        "--luna-path",
        type=str,
        default=None,
        help=(
            "Caminho do projeto Luna. "
            "Fallback: $LUNA_PROJECT_PATH ou ../Luna"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista o que seria feito, sem copiar arquivos",
    )
    args = parser.parse_args()

    luna_path = resolve_luna_path(args.luna_path)
    data_dir = ROOT_DIR / "data_output" / "clone_voice"

    logger.info("Iniciando sincronização de vozes")
    logger.info("Diretório de datasets: %s", data_dir)
    logger.info("Diretório Luna: %s", luna_path)

    entities = discover_entities(data_dir)
    if not entities:
        logger.error("Nenhuma entidade com unified_reference.wav encontrada")
        sys.stdout.write("Nenhuma entidade encontrada para sincronizar.\n")
        return 1

    logger.info("Entidades descobertas: %s", list(entities.keys()))

    if args.dry_run:
        sys.stdout.write("Modo dry-run ativado. Entidades encontradas:\n")
        for name, path in sorted(entities.items()):
            sys.stdout.write(f"  {name} <- {path}\n")
        return 0

    summary: dict[str, dict[str, str]] = {}
    has_failure = False

    for luna_name, source_path in sorted(entities.items()):
        logger.info(
            "Processando entidade '%s' a partir de %s",
            luna_name,
            source_path.parent.parent.name,
        )
        results = sync_entity(luna_name, source_path, luna_path)
        summary[luna_name] = results

        if "FALHA" in results.values():
            has_failure = True

    print_summary(summary)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())


# "O impedimento para a ação avança a ação. O que está no caminho torna-se o caminho."
# -- Marco Aurélio
