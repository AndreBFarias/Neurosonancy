#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerador de datasets de voz por entidade.

Compoe ElevenLabsClient e PhraseParser para gerar datasets com
voice settings variaveis por categoria de frase. Cada entidade
e definida em entity_profiles.json com perfil de voz unico.

Uso:
    python scripts/generate_entity_dataset.py --entity luna_gothic
    python scripts/generate_entity_dataset.py --entity luna_gothic --dry-run
    python scripts/generate_entity_dataset.py --voice-id XXX --phrases data_input/phrases.md --name Custom
    python scripts/generate_entity_dataset.py --entity luna_gothic --api-key sk_xxx
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from src.modules.clone_voice.core.generator.elevenlabs_client import (
    ElevenLabsClient,
    GenerationResult,
    VoiceSettings,
)
from src.modules.clone_voice.core.generator.phrase_parser import PhraseParser

logger = logging.getLogger("entity_dataset")

PROFILES_FILE = ROOT_DIR / "data_input" / "entity_profiles.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data_output" / "clone_voice"

_SSML_TAG_PATTERN = re.compile(r'\[[\w\s]+\]\s*')
_BREAK_TAG_PATTERN = re.compile(r'<break[^/]*/>\s*')


def _clean_phrase_for_training(phrase: str) -> str:
    """Remove tags ElevenLabs (SSML-like) para obter texto limpo de treinamento."""
    cleaned = _SSML_TAG_PATTERN.sub('', phrase)
    cleaned = _BREAK_TAG_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def _setup_logging(verbose: bool = False) -> None:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    handler = RotatingFileHandler(
        log_dir / "entity_dataset.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
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
    root.addHandler(handler)
    root.addHandler(console)


@dataclass
class EntityProfile:
    name: str
    voice_id: str
    model_id: str
    phrases_file: Path
    description: str
    language: str
    category_voice_settings: Dict[str, VoiceSettings]
    default_voice_settings: VoiceSettings

    @classmethod
    def from_dict(cls, key: str, data: dict) -> "EntityProfile":
        cat_settings = {}
        for cat_name, settings in data.get("category_voice_settings", {}).items():
            cat_settings[cat_name] = VoiceSettings(
                stability=settings["stability"],
                similarity_boost=settings["similarity_boost"],
                style=settings["style"],
                use_speaker_boost=settings.get("use_speaker_boost", True),
            )

        default_s = data.get("default_voice_settings", {})
        default_vs = VoiceSettings(
            stability=default_s.get("stability", 0.45),
            similarity_boost=default_s.get("similarity_boost", 0.85),
            style=default_s.get("style", 0.25),
            use_speaker_boost=default_s.get("use_speaker_boost", True),
        )

        phrases_path = Path(data["phrases_file"])
        if not phrases_path.is_absolute():
            phrases_path = ROOT_DIR / phrases_path

        return cls(
            name=data.get("name", key),
            voice_id=data["voice_id"],
            model_id=data.get("model_id", "eleven_multilingual_v2"),
            phrases_file=phrases_path,
            description=data.get("description", ""),
            language=data.get("language", "pt-BR"),
            category_voice_settings=cat_settings,
            default_voice_settings=default_vs,
        )


def load_profiles() -> Dict[str, EntityProfile]:
    if not PROFILES_FILE.exists():
        logger.error("Arquivo de perfis não encontrado: %s", PROFILES_FILE)
        return {}

    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    profiles = {}
    for key, data in raw.items():
        try:
            profiles[key] = EntityProfile.from_dict(key, data)
        except (KeyError, TypeError) as e:
            logger.error("Erro ao carregar perfil '%s': %s", key, e)

    return profiles


@dataclass
class GenerationEntry:
    index: int
    phrase: str
    category: str
    audio_file: str
    transcript_file: str
    duration_ms: int
    characters: int
    voice_settings_applied: Dict[str, float]
    timestamp: str


@dataclass
class GenerationStats:
    total_requested: int = 0
    total_generated: int = 0
    total_failed: int = 0
    total_characters: int = 0
    total_duration_ms: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requested == 0:
            return 0.0
        return (self.total_generated / self.total_requested) * 100


class EntityDatasetGenerator:
    """Gera datasets com voice settings variáveis por categoria."""

    MAX_RETRIES = 3
    INTER_PHRASE_DELAY = 0.5
    RATE_LIMIT_BASE_WAIT = 5
    TIMEOUT_BASE_WAIT = 2

    def __init__(
        self,
        voice_id: str,
        model_id: str,
        name: str,
        phrases_file: Path,
        category_voice_settings: Optional[Dict[str, VoiceSettings]] = None,
        default_voice_settings: Optional[VoiceSettings] = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        api_keys: Optional[List[str]] = None,
        language_code: Optional[str] = None,
    ):
        self.voice_id = voice_id
        self.model_id = model_id
        self.name = name
        self.phrases_file = phrases_file
        self.category_voice_settings = category_voice_settings or {}
        self.default_voice_settings = default_voice_settings or VoiceSettings()
        self.output_dir = output_dir
        self.language_code = language_code

        self.api_keys = api_keys or []
        self._current_key_index = 0

        self.parser = PhraseParser()
        self.client: Optional[ElevenLabsClient] = None
        self.stats = GenerationStats()
        self.entries: List[GenerationEntry] = []

        self._should_stop = False

    def _get_current_api_key(self) -> Optional[str]:
        if not self.api_keys:
            return os.getenv("ELEVENLABS_API_KEY")
        if self._current_key_index < len(self.api_keys):
            return self.api_keys[self._current_key_index]
        return None

    def _rotate_api_key(self) -> bool:
        self._current_key_index += 1
        if self._current_key_index >= len(self.api_keys):
            logger.error("Todas as API keys atingiram limite")
            return False

        new_key = self.api_keys[self._current_key_index]
        logger.info(
            "Rotacionando para API key %d/%d (***%s)",
            self._current_key_index + 1,
            len(self.api_keys),
            new_key[-4:],
        )

        self.client = ElevenLabsClient(
            api_key=new_key,
            voice_id=self.voice_id,
            model_id=self.model_id,
            language_code=self.language_code,
        )
        return self.client.initialize()

    def initialize(self, skip_client: bool = False) -> bool:
        if not self.phrases_file.exists():
            logger.error("Arquivo de frases não encontrado: %s", self.phrases_file)
            return False

        if not self.parser.parse_file(self.phrases_file):
            logger.error("Falha ao parsear frases: %s", self.phrases_file)
            return False

        stats = self.parser.get_stats()
        logger.info("Frases carregadas: %d total, %d categorias", stats["total"], len(stats) - 1)
        for cat_name in self.parser.get_category_names():
            cat = self.parser.get_category(cat_name)
            marker = "*" if cat_name in self.category_voice_settings else " "
            logger.info("  [%s] %s: %d frases", marker, cat_name, len(cat.phrases))

        if skip_client:
            return True

        api_key = self._get_current_api_key()
        if not api_key:
            logger.error("Nenhuma API key disponível")
            return False

        self.client = ElevenLabsClient(
            api_key=api_key,
            voice_id=self.voice_id,
            model_id=self.model_id,
            language_code=self.language_code,
        )

        if not self.client.initialize():
            logger.error("Falha ao inicializar ElevenLabsClient")
            return False

        lang_info = f", lang={self.language_code}" if self.language_code else ""
        logger.info("Cliente inicializado: voice=%s, model=%s%s", self.voice_id, self.model_id, lang_info)
        return True

    def _build_phrase_list(self) -> List[Tuple[str, str]]:
        """Constrói lista [(categoria, frase), ...] preservando mapeamento."""
        phrase_list: List[Tuple[str, str]] = []
        for cat_name in self.parser.get_category_names():
            category = self.parser.get_category(cat_name)
            if category:
                for phrase in category.phrases:
                    phrase_list.append((cat_name, phrase))
        return phrase_list

    def _apply_voice_settings(self, category: str) -> VoiceSettings:
        settings = self.category_voice_settings.get(category, self.default_voice_settings)
        if self.client is None:
            logger.warning("Client indisponível ao aplicar settings para [%s]", category)
            return settings
        self.client.set_voice_settings(
            stability=settings.stability,
            similarity_boost=settings.similarity_boost,
            style=settings.style,
            use_speaker_boost=settings.use_speaker_boost,
        )
        return settings

    def _settings_to_dict(self, settings: VoiceSettings) -> Dict[str, float]:
        return {
            "stability": settings.stability,
            "similarity_boost": settings.similarity_boost,
            "style": settings.style,
        }

    def dry_run(self) -> None:
        """Lista frases, categorias, voice settings e estimativa de custo."""
        phrase_list = self._build_phrase_list()
        total_chars = 0

        logger.info("=" * 70)
        logger.info("  DRY RUN: %s", self.name)
        logger.info("  Voice ID: %s", self.voice_id)
        logger.info("  Modelo: %s", self.model_id)
        logger.info("  Frases: %d", len(phrase_list))
        logger.info("=" * 70)

        current_cat = None
        for i, (category, phrase) in enumerate(phrase_list):
            if category != current_cat:
                settings = self.category_voice_settings.get(
                    category, self.default_voice_settings,
                )
                logger.info(
                    "--- [%s] stability=%.2f similarity=%.2f style=%.2f ---",
                    category, settings.stability, settings.similarity_boost, settings.style,
                )
                current_cat = category

            total_chars += len(phrase)
            truncated = phrase[:80] + "..." if len(phrase) > 80 else phrase
            logger.info("  %3d. %s", i + 1, truncated)

        cost_estimate = (total_chars / 1000) * 0.30
        duration_estimate = len(phrase_list) * 4

        logger.info("=" * 70)
        logger.info("  Total de frases: %d", len(phrase_list))
        logger.info("  Total de caracteres: %s", f"{total_chars:,}")
        logger.info("  Custo estimado: ~US$ %.2f", cost_estimate)
        logger.info("  Duração estimada de áudio: ~%ds (%.1f min)", duration_estimate, duration_estimate / 60)
        logger.info("  Categorias com settings customizados: %d", len(self.category_voice_settings))
        logger.info("=" * 70)

    def generate(self) -> GenerationStats:
        """Gera dataset completo com voice settings por categoria."""
        phrase_list = self._build_phrase_list()
        self.stats = GenerationStats(total_requested=len(phrase_list))
        self.entries.clear()
        self._should_stop = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.output_dir / f"{self.name}_{timestamp}"
        (dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "transcripts").mkdir(parents=True, exist_ok=True)

        logger.info("Iniciando geração: %d frases em %s", len(phrase_list), dataset_dir)

        current_cat = None
        for i, (category, phrase) in enumerate(phrase_list):
            if self._should_stop:
                logger.info("Geração interrompida")
                break

            if category != current_cat:
                settings = self._apply_voice_settings(category)
                logger.info(
                    "Voice settings [%s]: stability=%.2f similarity=%.2f style=%.2f",
                    category, settings.stability, settings.similarity_boost, settings.style,
                )
                current_cat = category

            display = phrase[:120] + "..." if len(phrase) > 120 else phrase
            logger.info("  [%d/%d] %s", i + 1, len(phrase_list), display)

            result = self._generate_with_retry(i, phrase, dataset_dir)

            if result.success:
                self.stats.total_generated += 1
                self.stats.total_characters += result.characters_used
                self.stats.total_duration_ms += result.duration_ms

                entry = GenerationEntry(
                    index=i,
                    phrase=phrase,
                    category=category,
                    audio_file=result.audio_path.name if result.audio_path else "",
                    transcript_file=f"{i:04d}.txt",
                    duration_ms=result.duration_ms,
                    characters=result.characters_used,
                    voice_settings_applied=self._settings_to_dict(
                        self.category_voice_settings.get(category, self.default_voice_settings)
                    ),
                    timestamp=datetime.now().isoformat(),
                )
                self.entries.append(entry)
            else:
                self.stats.total_failed += 1
                error_msg = f"[{i}] {category}: {result.error}"
                self.stats.errors.append(error_msg)
                logger.error("  FALHA: %s", error_msg)

            if i < len(phrase_list) - 1:
                time.sleep(self.INTER_PHRASE_DELAY)

        self._save_metadata(dataset_dir)
        self._save_training_files(dataset_dir)

        logger.info(
            "Geração concluída: %d/%d (%.1f%%) em %s",
            self.stats.total_generated,
            self.stats.total_requested,
            self.stats.success_rate,
            dataset_dir.name,
        )

        return self.stats

    def _generate_with_retry(
        self, index: int, phrase: str, dataset_dir: Path,
    ) -> GenerationResult:
        audio_path = dataset_dir / "wavs" / f"{index:04d}.mp3"
        transcript_path = dataset_dir / "transcripts" / f"{index:04d}.txt"

        for attempt in range(self.MAX_RETRIES):
            if self._should_stop:
                return GenerationResult(success=False, phrase=phrase, error="Interrompido")

            clean_text = _clean_phrase_for_training(phrase)
            result = self.client.generate_audio(clean_text, audio_path)

            if result.success:
                transcript_path.write_text(phrase, encoding="utf-8")
                return result

            error_str = str(result.error).lower()

            if "429" in str(result.error) or "rate" in error_str:
                if self.api_keys and self._rotate_api_key():
                    logger.info("Key rotacionada, retentando frase %d", index)
                    continue
                if self.api_keys:
                    break

                wait_time = (attempt + 1) * self.RATE_LIMIT_BASE_WAIT
                logger.warning("Rate limit, aguardando %ds (tentativa %d/%d)", wait_time, attempt + 1, self.MAX_RETRIES)
                time.sleep(wait_time)
                continue

            if "timeout" in error_str:
                wait_time = (attempt + 1) * self.TIMEOUT_BASE_WAIT
                logger.warning("Timeout, aguardando %ds (tentativa %d/%d)", wait_time, attempt + 1, self.MAX_RETRIES)
                time.sleep(wait_time)
                continue

            if "insufficient" in error_str or "401" in str(result.error):
                if self.api_keys and self._rotate_api_key():
                    continue
                break

            break

        return result

    def _save_metadata(self, dataset_dir: Path) -> None:
        category_settings_serialized = {}
        for cat_name, vs in self.category_voice_settings.items():
            category_settings_serialized[cat_name] = {
                "stability": vs.stability,
                "similarity_boost": vs.similarity_boost,
                "style": vs.style,
                "use_speaker_boost": vs.use_speaker_boost,
            }

        metadata = {
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "phrases_file": str(self.phrases_file),
            "category_voice_settings": category_settings_serialized,
            "stats": asdict(self.stats),
            "entries": [asdict(e) for e in self.entries],
        }

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Metadata salvo: %s", metadata_path)

    def _save_training_files(self, dataset_dir: Path) -> None:
        ljspeech_path = dataset_dir / "metadata.csv"
        with open(ljspeech_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                audio_name = entry.audio_file.replace(".mp3", "").replace(".wav", "")
                clean_text = _clean_phrase_for_training(entry.phrase)
                clean_text = clean_text.replace("|", " ").replace("\n", " ")
                f.write(f"{audio_name}|{clean_text}|{clean_text}\n")

        logger.info("LJSpeech metadata: %s", ljspeech_path)

        coqui_entries = []
        for entry in self.entries:
            coqui_entries.append({
                "audio_file": f"wavs/{entry.audio_file}",
                "text": _clean_phrase_for_training(entry.phrase),
                "speaker_name": self.name,
            })

        coqui_path = dataset_dir / "coqui_manifest.json"
        with open(coqui_path, "w", encoding="utf-8") as f:
            json.dump(coqui_entries, f, indent=2, ensure_ascii=False)

        logger.info("Coqui manifest: %s", coqui_path)

        chatterbox_path = dataset_dir / "chatterbox_data.jsonl"
        with open(chatterbox_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                line = json.dumps({
                    "audio_path": f"wavs/{entry.audio_file}",
                    "text": _clean_phrase_for_training(entry.phrase),
                }, ensure_ascii=False)
                f.write(line + "\n")

        logger.info("Chatterbox data: %s", chatterbox_path)

    def stop(self) -> None:
        self._should_stop = True
        logger.info("Parada solicitada")


def _parse_api_keys(raw: Optional[str]) -> List[str]:
    """Parseia uma ou mais API keys separadas por vírgula."""
    if not raw:
        env_keys = os.getenv("ELEVENLABS_API_KEYS", "")
        if env_keys:
            return [k.strip() for k in env_keys.split(",") if k.strip()]

        single = os.getenv("ELEVENLABS_API_KEY", "")
        return [single] if single else []

    return [k.strip() for k in raw.split(",") if k.strip()]


def _build_generator_from_profile(
    profile: EntityProfile,
    api_keys: List[str],
    output_dir: Path,
) -> EntityDatasetGenerator:
    return EntityDatasetGenerator(
        voice_id=profile.voice_id,
        model_id=profile.model_id,
        name=profile.name,
        phrases_file=profile.phrases_file,
        category_voice_settings=profile.category_voice_settings,
        default_voice_settings=profile.default_voice_settings,
        output_dir=output_dir,
        api_keys=api_keys,
        language_code=profile.language,
    )


def _build_generator_from_args(
    voice_id: str,
    phrases_file: Path,
    name: str,
    model_id: str,
    api_keys: List[str],
    output_dir: Path,
) -> EntityDatasetGenerator:
    return EntityDatasetGenerator(
        voice_id=voice_id,
        model_id=model_id,
        name=name,
        phrases_file=phrases_file,
        output_dir=output_dir,
        api_keys=api_keys,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera datasets de voz por entidade com voice settings por categoria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s --entity luna_gothic
  %(prog)s --entity luna_gothic --dry-run
  %(prog)s --voice-id XXX --phrases data_input/phrases.md --name Custom
  %(prog)s --entity luna_gothic --api-key sk_key1,sk_key2
        """,
    )

    parser.add_argument(
        "--entity",
        help="Nome da entidade definida em entity_profiles.json",
    )
    parser.add_argument(
        "--voice-id",
        help="Voice ID do ElevenLabs (modo manual, sem perfil)",
    )
    parser.add_argument(
        "--phrases",
        type=Path,
        help="Arquivo de frases markdown (modo manual)",
    )
    parser.add_argument(
        "--name",
        help="Nome do dataset (modo manual)",
    )
    parser.add_argument(
        "--model-id",
        default="eleven_multilingual_v2",
        help="Modelo do ElevenLabs (padrão: eleven_multilingual_v2)",
    )
    parser.add_argument(
        "--api-key",
        help="API key(s) separadas por vírgula (sobrepõe .env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída (padrão: data_output/clone_voice)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista frases e estimativas sem gerar áudio",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ativa logs detalhados (DEBUG)",
    )
    parser.add_argument(
        "--list-entities",
        action="store_true",
        help="Lista entidades disponíveis em entity_profiles.json",
    )

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    if args.list_entities:
        profiles = load_profiles()
        if not profiles:
            logger.info("Nenhuma entidade encontrada.")
            return 1
        logger.info("Entidades disponíveis (%d):", len(profiles))
        for key, profile in profiles.items():
            logger.info("  %s | %s | voice=%s...", key.ljust(20), profile.name, profile.voice_id[:12])
            logger.info("  %s | %s", " " * 20, profile.description)
        return 0

    api_keys = _parse_api_keys(args.api_key)

    if args.entity:
        profiles = load_profiles()
        if args.entity not in profiles:
            logger.error("Entidade '%s' não encontrada. Use --list-entities.", args.entity)
            return 1

        generator = _build_generator_from_profile(
            profiles[args.entity], api_keys, args.output_dir,
        )

    elif args.voice_id and args.phrases and args.name:
        phrases_path = args.phrases
        if not phrases_path.is_absolute():
            phrases_path = ROOT_DIR / phrases_path

        generator = _build_generator_from_args(
            voice_id=args.voice_id,
            phrases_file=phrases_path,
            name=args.name,
            model_id=args.model_id,
            api_keys=api_keys,
            output_dir=args.output_dir,
        )

    else:
        parser.error("Use --entity NOME ou --voice-id + --phrases + --name")
        return 1

    if not generator.initialize(skip_client=args.dry_run):
        logger.error("Falha na inicialização")
        return 1

    if args.dry_run:
        generator.dry_run()
        return 0

    stats = generator.generate()

    logger.info(
        "Resultado: %d/%d (%.1f%%)",
        stats.total_generated, stats.total_requested, stats.success_rate,
    )
    logger.info("Caracteres: %s", f"{stats.total_characters:,}")

    if stats.errors:
        logger.warning("Erros (%d):", len(stats.errors))
        for err in stats.errors[:10]:
            logger.warning("  %s", err)

    return 0 if stats.total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# "A simplicidade é a sofisticação suprema." — Leonardo da Vinci
