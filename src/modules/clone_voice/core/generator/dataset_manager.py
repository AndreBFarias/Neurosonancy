# -*- coding: utf-8 -*-

import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

from .phrase_parser import PhraseParser, generate_default_phrases
from .elevenlabs_client import ElevenLabsClient, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    voice_id: str
    model_id: str = "eleven_multilingual_v2"
    total_phrases: int = 50
    output_dir: Path = field(default_factory=lambda: Path("data_output/clone_voice"))
    phrases_file: Optional[Path] = None
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


@dataclass
class DatasetStats:
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


@dataclass
class DatasetEntry:
    index: int
    phrase: str
    audio_file: str
    transcript_file: str
    duration_ms: int
    characters: int
    timestamp: str


class DatasetManager:

    def __init__(self, config: DatasetConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key

        self.parser = PhraseParser()
        self.client: Optional[ElevenLabsClient] = None

        self.stats = DatasetStats()
        self.entries: List[DatasetEntry] = []

        self._is_running = False
        self._should_stop = False

        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_complete: Optional[Callable[[DatasetStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_phrase_generated: Optional[Callable[[int, str, Path], None]] = None

    def initialize(self) -> bool:
        self.client = ElevenLabsClient(
            api_key=self.api_key,
            voice_id=self.config.voice_id,
            model_id=self.config.model_id,
        )

        self.client.set_voice_settings(
            stability=self.config.stability,
            similarity_boost=self.config.similarity_boost,
            style=self.config.style,
            use_speaker_boost=self.config.use_speaker_boost,
        )

        if not self.client.initialize():
            return False

        if self.config.phrases_file and self.config.phrases_file.exists():
            if not self.parser.parse_file(self.config.phrases_file):
                logger.warning("Falha ao parsear arquivo de frases, usando default")
                self.parser.parse_content(generate_default_phrases())
        else:
            self.parser.parse_content(generate_default_phrases())

        return True

    def _setup_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.config.output_dir / f"{self.config.name}_{timestamp}"

        (dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "transcripts").mkdir(parents=True, exist_ok=True)

        return dataset_dir

    def generate_dataset(self) -> DatasetStats:
        if self._is_running:
            logger.warning("Geracao ja em andamento")
            return self.stats

        self._is_running = True
        self._should_stop = False
        self.stats = DatasetStats(total_requested=self.config.total_phrases)
        self.entries.clear()

        dataset_dir = self._setup_output_dir()
        logger.info(f"Dataset dir: {dataset_dir}")

        phrases = self.parser.sample_balanced(self.config.total_phrases)
        self.stats.total_requested = len(phrases)

        logger.info(f"Gerando {len(phrases)} frases...")

        for i, phrase in enumerate(phrases):
            if self._should_stop:
                logger.info("Geracao interrompida pelo usuario")
                break

            if self.on_progress:
                self.on_progress(i + 1, len(phrases), phrase[:50])

            result = self._generate_single_with_retry(i, phrase, dataset_dir)

            if result.success:
                self.stats.total_generated += 1
                self.stats.total_characters += result.characters_used
                self.stats.total_duration_ms += result.duration_ms

                entry = DatasetEntry(
                    index=i,
                    phrase=phrase,
                    audio_file=result.audio_path.name if result.audio_path else "",
                    transcript_file=f"{i:04d}.txt",
                    duration_ms=result.duration_ms,
                    characters=result.characters_used,
                    timestamp=datetime.now().isoformat(),
                )
                self.entries.append(entry)

                if self.on_phrase_generated and result.audio_path:
                    self.on_phrase_generated(i, phrase, result.audio_path)
            else:
                self.stats.total_failed += 1
                if result.error:
                    self.stats.errors.append(f"[{i}] {result.error}")

                if self.on_error:
                    self.on_error(f"Frase {i}: {result.error}")

            if i < len(phrases) - 1:
                time.sleep(0.5)

        self._save_metadata(dataset_dir)
        self._save_training_files(dataset_dir)

        self._is_running = False

        if self.on_complete:
            self.on_complete(self.stats)

        logger.info(f"Dataset gerado: {self.stats.total_generated}/{self.stats.total_requested}")
        logger.info(f"Diretorio: {dataset_dir}")

        return self.stats

    def _generate_single_with_retry(self, index: int, phrase: str, dataset_dir: Path, max_retries: int = 3) -> GenerationResult:
        audio_path = dataset_dir / "wavs" / f"{index:04d}.wav"
        transcript_path = dataset_dir / "transcripts" / f"{index:04d}.txt"

        for attempt in range(max_retries):
            if self._should_stop:
                return GenerationResult(success=False, phrase=phrase, error="Interrompido")

            result = self.client.generate_audio(phrase, audio_path)

            if result.success:
                transcript_path.write_text(phrase, encoding='utf-8')
                return result

            if "429" in str(result.error) or "rate" in str(result.error).lower():
                wait_time = (attempt + 1) * 5
                logger.warning(f"Rate limit, aguardando {wait_time}s...")
                time.sleep(wait_time)
                continue

            if "timeout" in str(result.error).lower():
                wait_time = (attempt + 1) * 2
                logger.warning(f"Timeout, tentativa {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            break

        return result

    def _generate_single(self, index: int, phrase: str, dataset_dir: Path) -> GenerationResult:
        audio_path = dataset_dir / "wavs" / f"{index:04d}.wav"
        transcript_path = dataset_dir / "transcripts" / f"{index:04d}.txt"

        result = self.client.generate_audio(phrase, audio_path)

        if result.success:
            transcript_path.write_text(phrase, encoding='utf-8')

        return result

    def _save_metadata(self, dataset_dir: Path) -> None:
        metadata = {
            "name": self.config.name,
            "created_at": datetime.now().isoformat(),
            "voice_id": self.config.voice_id,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "use_speaker_boost": self.config.use_speaker_boost,
            },
            "stats": asdict(self.stats),
            "entries": [asdict(e) for e in self.entries],
        }

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata salvo: {metadata_path}")

    def _save_training_files(self, dataset_dir: Path) -> None:
        ljspeech_path = dataset_dir / "metadata.csv"
        with open(ljspeech_path, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                audio_name = entry.audio_file.replace('.wav', '')
                phrase_clean = entry.phrase.replace('|', ' ').replace('\n', ' ')
                f.write(f"{audio_name}|{phrase_clean}|{phrase_clean}\n")

        logger.info(f"LJSpeech metadata: {ljspeech_path}")

        coqui_path = dataset_dir / "coqui_manifest.json"
        coqui_entries = []
        for entry in self.entries:
            coqui_entries.append({
                "audio_file": f"wavs/{entry.audio_file}",
                "text": entry.phrase,
                "speaker_name": self.config.name,
            })

        with open(coqui_path, 'w', encoding='utf-8') as f:
            json.dump(coqui_entries, f, indent=2, ensure_ascii=False)

        logger.info(f"Coqui manifest: {coqui_path}")

        chatterbox_path = dataset_dir / "chatterbox_data.jsonl"
        with open(chatterbox_path, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                line = json.dumps({
                    "audio_path": f"wavs/{entry.audio_file}",
                    "text": entry.phrase,
                }, ensure_ascii=False)
                f.write(line + "\n")

        logger.info(f"Chatterbox data: {chatterbox_path}")

        readme_content = f"""# Dataset: {self.config.name}

## Info
- Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Voice ID: {self.config.voice_id}
- Modelo: {self.config.model_id}
- Total de amostras: {self.stats.total_generated}

## Estrutura
```
{dataset_dir.name}/
├── wavs/                  # Arquivos de audio
├── transcripts/           # Transcricoes individuais
├── metadata.csv           # Formato LJSpeech (Coqui TTS)
├── coqui_manifest.json    # Manifest para Coqui TTS
├── chatterbox_data.jsonl  # Formato Chatterbox
└── metadata.json          # Metadados completos
```

## Uso com Coqui TTS
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text="Teste", file_path="output.wav", speaker_wav="wavs/0000.wav")
```

## Uso com Chatterbox
```python
from chatterbox import ChatterboxTTS
model = ChatterboxTTS.from_pretrained("./checkpoint")
audio = model.generate("Teste")
```

## Voice Settings
- Stability: {self.config.stability}
- Similarity Boost: {self.config.similarity_boost}
- Style: {self.config.style}
- Speaker Boost: {self.config.use_speaker_boost}
"""
        readme_path = dataset_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')

    def stop(self) -> None:
        self._should_stop = True
        logger.info("Solicitada parada da geracao")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_output_dir(self) -> Path:
        return self.config.output_dir


class AsyncDatasetManager(DatasetManager):

    async def generate_dataset_async(self) -> DatasetStats:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.generate_dataset)
