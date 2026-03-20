#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validador de arquivos reference.wav sincronizados no Luna.

Verifica integridade, formato e correspondência SHA256 dos arquivos
de referência de voz de cada entidade no projeto Luna.

Uso:
    python scripts/validate_voices.py
    python scripts/validate_voices.py --luna-path /caminho/para/Luna
    python scripts/validate_voices.py --source-path data_output/clone_voice
    python scripts/validate_voices.py -v
"""

import argparse
import hashlib
import logging
import os
import sys
import wave
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("validate_voices")

ENGINES = ("coqui", "chatterbox")

ENTITY_SOURCE_MAP: Dict[str, str] = {
    "luna": "Luna Gothic_*",
    "mars": "Mars_*",
    "juno": "Juno_*",
    "eris": "Eris_*",
    "somn": "Somn_*",
    "lars": "Lars_*",
}


@dataclass
class CheckResult:
    entity: str
    engine: str
    check: str
    status: str
    detail: str = ""


@dataclass
class ValidationSummary:
    results: List[CheckResult] = field(default_factory=list)
    total_errors: int = 0
    total_warnings: int = 0
    total_ok: int = 0

    def add(self, result: CheckResult) -> None:
        self.results.append(result)
        if result.status == "ERROR":
            self.total_errors += 1
        elif result.status == "WARNING":
            self.total_warnings += 1
        else:
            self.total_ok += 1


def _setup_logging(verbose: bool = False) -> None:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    file_handler = RotatingFileHandler(
        log_dir / "validate_voices.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s",
    ))
    console.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console)


def _sha256(filepath: Path) -> str:
    """Calcula SHA256 de um arquivo."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_source_reference(entity_name: str, source_dir: Path) -> Optional[Path]:
    """Localiza o unified_reference.wav correspondente na fonte Neurosonancy."""
    pattern = ENTITY_SOURCE_MAP.get(entity_name.lower())
    if not pattern:
        logger.debug("Sem mapeamento de fonte para entidade '%s'", entity_name)
        return None

    prefix = pattern.rstrip("*").rstrip("_")
    candidates = sorted(
        [
            d for d in source_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ],
        key=lambda p: p.name,
        reverse=True,
    )

    if not candidates:
        logger.debug("Nenhum dataset encontrado para '%s' em %s", entity_name, source_dir)
        return None

    ref = candidates[0] / "top_10_selection" / "unified_reference.wav"
    if ref.exists():
        return ref

    logger.debug("unified_reference.wav ausente em %s", candidates[0])
    return None


def _is_voice_entity(entity_dir: Path) -> bool:
    """Verifica se o diretório da entidade possui subdiretórios de voz."""
    for engine in ENGINES:
        voice_dir = entity_dir / "voice" / engine
        if voice_dir.exists():
            return True
    return False


def _check_exists(filepath: Path, entity: str, engine: str) -> CheckResult:
    if filepath.exists():
        return CheckResult(entity, engine, "Arquivo existe", "OK")
    return CheckResult(entity, engine, "Arquivo existe", "ERROR", str(filepath))


def _check_size(filepath: Path, entity: str, engine: str) -> CheckResult:
    if not filepath.exists():
        return CheckResult(entity, engine, "Tamanho > 0", "ERROR", "Arquivo inexistente")
    size = filepath.stat().st_size
    if size > 0:
        return CheckResult(entity, engine, "Tamanho > 0", "OK", f"{size} bytes")
    return CheckResult(entity, engine, "Tamanho > 0", "ERROR", "Arquivo vazio (0 bytes)")


def _check_wav_header(filepath: Path, entity: str, engine: str) -> CheckResult:
    if not filepath.exists():
        return CheckResult(entity, engine, "Formato WAV", "ERROR", "Arquivo inexistente")
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
        if header == b"RIFF":
            return CheckResult(entity, engine, "Formato WAV", "OK")
        return CheckResult(
            entity, engine, "Formato WAV", "ERROR",
            f"Header inválido: {header!r} (esperado: b'RIFF')",
        )
    except OSError as e:
        return CheckResult(entity, engine, "Formato WAV", "ERROR", str(e))


def _check_duration(filepath: Path, entity: str, engine: str, min_seconds: float = 3.0) -> CheckResult:
    if not filepath.exists():
        return CheckResult(entity, engine, "Duração > 3s", "ERROR", "Arquivo inexistente")
    try:
        with wave.open(str(filepath), "rb") as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate()
            if framerate == 0:
                return CheckResult(entity, engine, "Duração > 3s", "WARNING", "Framerate = 0")
            duration = frames / framerate
        if duration >= min_seconds:
            return CheckResult(entity, engine, "Duração > 3s", "OK", f"{duration:.1f}s")
        return CheckResult(
            entity, engine, "Duração > 3s", "WARNING",
            f"{duration:.1f}s (mínimo recomendado: {min_seconds}s)",
        )
    except wave.Error as e:
        return CheckResult(entity, engine, "Duração > 3s", "WARNING", f"Erro ao ler WAV: {e}")
    except OSError as e:
        return CheckResult(entity, engine, "Duração > 3s", "WARNING", f"Erro de I/O: {e}")


def _check_sha256(
    filepath: Path, source_path: Optional[Path], entity: str, engine: str,
) -> CheckResult:
    if source_path is None:
        return CheckResult(entity, engine, "SHA256", "WARNING", "Fonte não encontrada para comparação")
    if not filepath.exists():
        return CheckResult(entity, engine, "SHA256", "ERROR", "Arquivo inexistente")

    target_hash = _sha256(filepath)
    source_hash = _sha256(source_path)

    if target_hash == source_hash:
        return CheckResult(entity, engine, "SHA256", "OK", target_hash[:16])
    return CheckResult(
        entity, engine, "SHA256", "WARNING",
        f"Hashes divergem (destino: {target_hash[:16]}... fonte: {source_hash[:16]}...)",
    )


def validate_entity(
    entity_name: str,
    entity_dir: Path,
    source_dir: Optional[Path],
    summary: ValidationSummary,
) -> None:
    """Executa todas as validações para uma entidade."""
    source_ref = _find_source_reference(entity_name, source_dir) if source_dir else None

    for engine in ENGINES:
        ref_path = entity_dir / "voice" / engine / "reference.wav"
        voice_dir = entity_dir / "voice" / engine

        if not voice_dir.exists():
            logger.debug("Diretório %s não existe, pulando", voice_dir)
            continue

        logger.info("  Validando %s/%s ...", entity_name, engine)

        summary.add(_check_exists(ref_path, entity_name, engine))
        summary.add(_check_size(ref_path, entity_name, engine))
        summary.add(_check_wav_header(ref_path, entity_name, engine))
        summary.add(_check_duration(ref_path, entity_name, engine))
        summary.add(_check_sha256(ref_path, source_ref, entity_name, engine))


def _print_summary_table(summary: ValidationSummary) -> None:
    """Exibe tabela de resultados formatada."""
    logger.info("")
    logger.info("=" * 78)
    logger.info("  RESULTADO DA VALIDAÇÃO")
    logger.info("=" * 78)

    header = f"{'Entidade':<14} {'Engine':<12} {'Verificação':<20} {'Status':<8} {'Detalhe'}"
    logger.info(header)
    logger.info("-" * 78)

    for r in summary.results:
        detail_display = r.detail[:30] if r.detail else ""
        logger.info(
            "%-14s %-12s %-20s %-8s %s",
            r.entity, r.engine, r.check, r.status, detail_display,
        )

    logger.info("-" * 78)
    logger.info(
        "  Total: %d OK | %d WARNING | %d ERROR",
        summary.total_ok, summary.total_warnings, summary.total_errors,
    )
    logger.info("=" * 78)


def _resolve_luna_path(cli_arg: Optional[str]) -> Path:
    """Resolve o caminho do projeto Luna via argumento, variável de ambiente ou fallback."""
    if cli_arg:
        return Path(cli_arg).resolve()

    env_val = os.getenv("LUNA_PROJECT_PATH")
    if env_val:
        return Path(env_val).resolve()

    return (ROOT_DIR / ".." / "Luna").resolve()


def _resolve_source_path(cli_arg: Optional[str]) -> Optional[Path]:
    """Resolve o caminho dos datasets fonte no Neurosonancy."""
    if cli_arg:
        p = Path(cli_arg)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p.resolve()

    default = ROOT_DIR / "data_output" / "clone_voice"
    if default.exists():
        return default.resolve()

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida arquivos reference.wav das entidades no projeto Luna.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s
  %(prog)s --luna-path /caminho/para/Luna
  %(prog)s --source-path data_output/clone_voice
  %(prog)s -v
        """,
    )

    parser.add_argument(
        "--luna-path",
        type=str,
        default=None,
        help="Caminho do projeto Luna (padrão: LUNA_PROJECT_PATH ou ../Luna)",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Caminho dos datasets fonte para comparação SHA256 (padrão: data_output/clone_voice/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ativa logs detalhados (DEBUG)",
    )

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    luna_path = _resolve_luna_path(args.luna_path)
    entities_dir = luna_path / "src" / "assets" / "panteao" / "entities"

    if not entities_dir.exists():
        logger.error("Diretório de entidades não encontrado: %s", entities_dir)
        return 1

    source_path = _resolve_source_path(args.source_path)
    if source_path:
        logger.info("Fonte para comparação SHA256: %s", source_path)
    else:
        logger.warning("Nenhuma fonte disponível para comparação SHA256")

    all_voice_dirs = sorted([
        d for d in entities_dir.iterdir()
        if d.is_dir() and _is_voice_entity(d)
    ])

    entity_dirs = []
    for d in all_voice_dirs:
        if d.name.lower() in ENTITY_SOURCE_MAP:
            entity_dirs.append(d)
        else:
            logger.debug(
                "Entidade '%s' não gerenciada por este pipeline, pulando", d.name,
            )

    if not entity_dirs:
        logger.warning("Nenhuma entidade com diretório de voz encontrada em %s", entities_dir)
        return 0

    logger.info("Entidades com voz encontradas: %d", len(entity_dirs))

    summary = ValidationSummary()

    for entity_dir in entity_dirs:
        entity_name = entity_dir.name
        logger.info("Entidade: %s", entity_name)
        validate_entity(entity_name, entity_dir, source_path, summary)

    _print_summary_table(summary)

    if summary.total_errors > 0:
        logger.error("Validação finalizada com %d erro(s)", summary.total_errors)
        return 1

    logger.info("Validação concluída sem erros")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# "Aquele que não examina a própria vida não é digno de vivê-la." -- Sócrates
