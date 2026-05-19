"""Catálogo de vozes disponíveis para o Leitor de Textos.

Combina as entidades de `data_input/entity_profiles.json` com as `unified_reference.wav`
geradas em `data_output/clone_voice/<Entity>_<ts>/top_10_selection/`.

Cada entidade vira um `VoiceProfile` com display name, descrição, voice_settings padrão
e o caminho absoluto da reference.wav mais recente encontrada.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[4]
ENTITY_PROFILES = ROOT_DIR / "data_input" / "entity_profiles.json"
DATASETS_ROOT = ROOT_DIR / "data_output" / "clone_voice"

# Convenções de nome dentro de `top_10_selection/`. Procuramos arquivos
# específicos por engine; caso ausente, caímos no unified.
REFERENCE_DEFAULT = "unified_reference.wav"
REFERENCE_BY_ENGINE = {
    "coqui": ("coqui_reference.wav", REFERENCE_DEFAULT),
    "chatterbox": ("chatterbox_reference.wav", REFERENCE_DEFAULT),
}


class VoiceCatalogError(RuntimeError):
    """Erro ao montar catálogo de vozes."""


@dataclass(frozen=True)
class VoiceProfile:
    key: str
    display_name: str
    description: str
    reference_wav: Path  # Default (coqui). Mantido por compatibilidade.
    default_speed: float = 1.0
    # Mapeia engine → reference específica. Sempre contém pelo menos "coqui".
    # Pode incluir "chatterbox_reference.wav" se existir no dataset.
    references: dict[str, Path] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen=True não impede atribuir via object.__setattr__ no __post_init__
        if self.references is None:
            object.__setattr__(self, "references", {"coqui": self.reference_wav})

    @property
    def has_reference(self) -> bool:
        return self.reference_wav.is_file()

    def reference_for(self, engine: str) -> Path:
        """Retorna a reference.wav adequada ao engine, com fallback pra default."""
        if engine in self.references and self.references[engine].is_file():
            return self.references[engine]
        return self.reference_wav


def _latest_dataset_dir(entity_name: str) -> Path | None:
    """Dataset mais recente (por mtime do diretório `top_10_selection/`) ou None."""
    if not DATASETS_ROOT.is_dir():
        return None

    pattern = f"{entity_name}_*"
    candidates = sorted(
        (d for d in DATASETS_ROOT.glob(pattern) if d.is_dir()),
        key=lambda d: (d / "top_10_selection").stat().st_mtime
        if (d / "top_10_selection").is_dir()
        else 0,
        reverse=True,
    )
    for cand in candidates:
        if (cand / "top_10_selection").is_dir():
            return cand
    return None


def _references_for(entity_name: str) -> dict[str, Path]:
    """Retorna `{engine: path}` para todas as engines com reference encontrada
    no dataset mais recente da entidade. Sempre inclui pelo menos a default
    (coqui via unified_reference.wav) se houver dataset.
    """
    dataset = _latest_dataset_dir(entity_name)
    if dataset is None:
        return {}
    selection = dataset / "top_10_selection"

    result: dict[str, Path] = {}
    default_path = selection / REFERENCE_DEFAULT
    if default_path.is_file():
        result["coqui"] = default_path

    for engine, candidates in REFERENCE_BY_ENGINE.items():
        for filename in candidates:
            candidate = selection / filename
            if candidate.is_file():
                result[engine] = candidate
                break
    return result


def load_catalog() -> list[VoiceProfile]:
    """Lê entity_profiles.json e devolve apenas as vozes com reference encontrada."""
    if not ENTITY_PROFILES.is_file():
        raise VoiceCatalogError(
            f"entity_profiles.json não encontrado em {ENTITY_PROFILES}"
        )

    with ENTITY_PROFILES.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    profiles: list[VoiceProfile] = []
    for key, entry in raw.items():
        display = entry.get("name", key)
        description = entry.get("description", "")
        refs = _references_for(display)
        if not refs:
            # Sem reference.wav (nenhum engine), voz fica fora do catálogo
            # para evitar erro no momento da geração.
            continue
        default_ref = refs.get("coqui") or next(iter(refs.values()))
        profiles.append(
            VoiceProfile(
                key=key,
                display_name=display,
                description=description,
                reference_wav=default_ref,
                default_speed=1.0,
                references=refs,
            )
        )

    if not profiles:
        raise VoiceCatalogError(
            "Nenhuma voz encontrada com reference.wav. "
            "Gere ao menos um dataset clone_voice antes."
        )

    return profiles


def get_default(catalog: Iterable[VoiceProfile]) -> VoiceProfile:
    """Retorna Éris se disponível; senão a primeira voz do catálogo."""
    catalog = list(catalog)
    for profile in catalog:
        if profile.key == "eris":
            return profile
    return catalog[0]
