# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field

from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    total_characters: int
    total_phrases: int
    estimated_cost_usd: float
    estimated_duration_minutes: float
    price_per_1k_chars: float = 0.03

    def __str__(self) -> str:
        return (
            f"Frases: {self.total_phrases}\n"
            f"Caracteres: {self.total_characters:,}\n"
            f"Duracao estimada: ~{self.estimated_duration_minutes:.1f} min\n"
            f"Custo estimado: ${self.estimated_cost_usd:.2f} USD"
        )


@dataclass
class DatasetMetadata:
    created_at: str
    voice_id: str
    voice_name: str
    model_id: str
    total_samples: int
    total_duration_seconds: float
    total_characters: int
    estimated_cost_usd: float
    levels_included: List[str]
    audio_format: Dict[str, Any]
    source_file: Optional[str] = None
    
    def save(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'DatasetMetadata':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 10
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 10.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 300.0
    jitter_range: float = 0.5


class RateLimiter:
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._last_request_time = 0.0
        self._consecutive_errors = 0
        self._current_delay = self.config.min_delay_seconds
    
    def wait(self) -> float:
        now = time.time()
        elapsed = now - self._last_request_time
        
        delay = self._current_delay
        jitter = random.uniform(-self.config.jitter_range, self.config.jitter_range)
        delay = max(0, delay + jitter)
        
        if elapsed < delay:
            wait_time = delay - elapsed
            logger.debug(f"Rate limiter: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self._last_request_time = time.time()
        return delay
    
    def on_success(self):
        self._consecutive_errors = 0
        self._current_delay = max(
            self.config.min_delay_seconds,
            self._current_delay * 0.9
        )
    
    def on_rate_limit(self, retry_after: Optional[float] = None) -> float:
        self._consecutive_errors += 1
        
        if retry_after:
            self._current_delay = retry_after
        else:
            self._current_delay = min(
                self._current_delay * self.config.backoff_multiplier,
                self.config.max_backoff_seconds
            )
        
        logger.warning(
            f"Rate limit hit! Backing off to {self._current_delay:.1f}s "
            f"(consecutive errors: {self._consecutive_errors})"
        )
        
        return self._current_delay
    
    def on_error(self) -> float:
        self._consecutive_errors += 1
        self._current_delay = min(
            self._current_delay * 1.5,
            self.config.max_backoff_seconds
        )
        return self._current_delay


def parse_phrases_from_markdown(filepath: str) -> List[str]:
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    content = path.read_text(encoding='utf-8')
    
    phrases = []
    
    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, content)
    
    for match in matches:
        phrase = match.strip()
        if len(phrase) > 10:
            phrases.append(phrase)
    
    if not phrases:
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                phrase = line.lstrip('-*').strip()
                phrase = phrase.strip('"\'')
                if len(phrase) > 10:
                    phrases.append(phrase)
    
    return phrases


class ElevenLabsDatasetGenerator:
    
    SUPPORTED_FORMATS = {
        'mp3_44100_128': {'ext': 'mp3', 'sample_rate': 44100},
        'mp3_44100_192': {'ext': 'mp3', 'sample_rate': 44100},
        'pcm_16000': {'ext': 'wav', 'sample_rate': 16000},
        'pcm_22050': {'ext': 'wav', 'sample_rate': 22050},
        'pcm_44100': {'ext': 'wav', 'sample_rate': 44100},
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.model_id = model_id
        self.output_format = output_format
        
        if not self.api_key:
            raise ValueError("API key is required. Set ELEVENLABS_API_KEY or pass api_key parameter.")
        
        self._client = None
        self._voices_cache = None
        self._rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
    
    @property
    def client(self):
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                raise ImportError("elevenlabs package not installed. Run: pip install elevenlabs")
        return self._client
    
    def list_voices(self) -> List[Dict[str, str]]:
        if self._voices_cache is None:
            self._rate_limiter.wait()
            try:
                response = self.client.voices.get_all()
                self._voices_cache = [
                    {
                        'voice_id': v.voice_id,
                        'name': v.name,
                        'category': getattr(v, 'category', 'unknown'),
                        'labels': getattr(v, 'labels', {}),
                    }
                    for v in response.voices
                ]
                self._rate_limiter.on_success()
            except Exception as e:
                self._rate_limiter.on_error()
                raise
        return self._voices_cache
    
    def get_voice_name(self, voice_id: str) -> str:
        voices = self.list_voices()
        for v in voices:
            if v['voice_id'] == voice_id:
                return v['name']
        return "Unknown Voice"
    
    def estimate_cost(
        self,
        phrases: List[str],
        price_per_1k_chars: float = 0.03
    ) -> CostEstimate:
        total_chars = sum(len(p) for p in phrases)
        cost = (total_chars / 1000) * price_per_1k_chars
        
        avg_chars = total_chars / len(phrases) if phrases else 0
        chars_per_second = 15
        estimated_seconds = total_chars / chars_per_second
        estimated_minutes = estimated_seconds / 60
        
        return CostEstimate(
            total_characters=total_chars,
            total_phrases=len(phrases),
            estimated_cost_usd=round(cost, 4),
            estimated_duration_minutes=round(estimated_minutes, 1),
            price_per_1k_chars=price_per_1k_chars
        )
    
    def generate_sample(
        self,
        text: str,
        output_path: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        max_retries: int = 5
    ) -> Tuple[bool, float, Optional[str]]:
        if not self.voice_id:
            raise ValueError("Voice ID is required. Set ELEVENLABS_VOICE_ID or pass voice_id parameter.")
        
        for attempt in range(max_retries):
            self._rate_limiter.wait()
            
            try:
                audio_generator = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id=self.model_id,
                    output_format=self.output_format,
                    voice_settings={
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style,
                        "use_speaker_boost": True
                    }
                )
                
                audio_bytes = b"".join(audio_generator)
                
                temp_path = output_path + ".tmp"
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                
                audio = AudioSegment.from_file(temp_path)
                
                audio = audio.set_frame_rate(22050).set_channels(1)
                audio.export(output_path, format="wav")
                
                os.remove(temp_path)
                
                duration = len(audio) / 1000.0
                
                self._rate_limiter.on_success()
                
                return True, duration, None
                
            except Exception as e:
                error_str = str(e).lower()
                
                if '429' in error_str or 'rate' in error_str or 'limit' in error_str:
                    retry_after = None
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                retry_after = float(retry_after)
                            except ValueError:
                                retry_after = None
                    
                    wait_time = self._rate_limiter.on_rate_limit(retry_after)
                    logger.warning(f"429 Rate limit on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                elif '5' in str(getattr(e, 'status_code', '')) or 'server' in error_str:
                    wait_time = self._rate_limiter.on_error()
                    logger.warning(f"Server error on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"Error generating sample: {e}")
                    self._rate_limiter.on_error()
                    return False, 0.0, str(e)
        
        return False, 0.0, f"Max retries ({max_retries}) exceeded"
    
    def generate_dataset(
        self,
        phrases: List[str],
        output_dir: str,
        level_name: str = "custom",
        source_file: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        error_callback: Optional[callable] = None,
        start_index: int = 0
    ) -> Optional[DatasetMetadata]:
        if not self.voice_id:
            raise ValueError("Voice ID is required.")
        
        output_path = Path(output_dir)
        audios_dir = output_path / "audios"
        transcripts_dir = output_path / "transcripts"
        errors_file = output_path / "errors.json"
        progress_file = output_path / "progress.json"
        
        audios_dir.mkdir(parents=True, exist_ok=True)
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        total_duration = 0.0
        total_chars = 0
        successful = 0
        errors = []
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    start_index = progress_data.get('last_completed_index', 0)
                    total_duration = progress_data.get('total_duration', 0.0)
                    total_chars = progress_data.get('total_chars', 0)
                    successful = progress_data.get('successful', 0)
                    logger.info(f"Resuming from index {start_index + 1}")
            except Exception:
                pass
        
        voice_name = self.get_voice_name(self.voice_id)
        
        for i, phrase in enumerate(phrases[start_index:], start_index + 1):
            audio_filename = f"{i:04d}_{level_name}.wav"
            transcript_filename = f"{i:04d}_{level_name}.txt"
            
            audio_path = audios_dir / audio_filename
            transcript_path = transcripts_dir / transcript_filename
            
            if audio_path.exists() and transcript_path.exists():
                logger.debug(f"Skipping already generated: {audio_filename}")
                continue
            
            success, duration, error = self.generate_sample(phrase, str(audio_path))
            
            if success:
                total_duration += duration
                total_chars += len(phrase)
                successful += 1
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(phrase)
                
                with open(progress_file, 'w') as f:
                    json.dump({
                        'last_completed_index': i,
                        'total_duration': total_duration,
                        'total_chars': total_chars,
                        'successful': successful,
                        'timestamp': datetime.now().isoformat()
                    }, f)
            else:
                errors.append({
                    'index': i,
                    'phrase': phrase[:100] + '...' if len(phrase) > 100 else phrase,
                    'error': error
                })
                
                if error_callback:
                    error_callback(i, phrase, error)
                
                with open(errors_file, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, indent=2, ensure_ascii=False)
            
            if progress_callback:
                progress_callback(
                    i, 
                    len(phrases), 
                    phrase[:50] + "..." if len(phrase) > 50 else phrase,
                    total_duration
                )
        
        estimate = self.estimate_cost(phrases)
        
        metadata = DatasetMetadata(
            created_at=datetime.now().isoformat(),
            voice_id=self.voice_id,
            voice_name=voice_name,
            model_id=self.model_id,
            total_samples=successful,
            total_duration_seconds=round(total_duration, 2),
            total_characters=total_chars,
            estimated_cost_usd=estimate.estimated_cost_usd,
            levels_included=[level_name],
            audio_format={
                "sample_rate": 22050,
                "channels": 1,
                "format": "wav"
            },
            source_file=source_file
        )
        
        metadata.save(output_path / "metadata.json")
        
        if progress_file.exists():
            progress_file.unlink()
        
        logger.info(f"Dataset generated: {successful}/{len(phrases)} samples, {total_duration:.1f}s total")
        
        if errors:
            logger.warning(f"Errors: {len(errors)} phrases failed. See {errors_file}")
        
        return metadata
    
    def validate_dataset(self, dataset_dir: str) -> Dict[str, Any]:
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            return {"valid": False, "error": "Dataset directory not found"}
        
        metadata_path = dataset_path / "metadata.json"
        if not metadata_path.exists():
            return {"valid": False, "error": "metadata.json not found"}
        
        try:
            metadata = DatasetMetadata.load(metadata_path)
        except Exception as e:
            return {"valid": False, "error": f"Invalid metadata: {e}"}
        
        audios_dir = dataset_path / "audios"
        transcripts_dir = dataset_path / "transcripts"
        
        audio_files = list(audios_dir.glob("*.wav")) if audios_dir.exists() else []
        transcript_files = list(transcripts_dir.glob("*.txt")) if transcripts_dir.exists() else []
        
        audio_names = {f.stem for f in audio_files}
        transcript_names = {f.stem for f in transcript_files}
        
        missing_transcripts = audio_names - transcript_names
        missing_audios = transcript_names - audio_names
        
        total_duration = 0.0
        for audio_file in audio_files:
            try:
                audio = AudioSegment.from_wav(audio_file)
                total_duration += len(audio) / 1000.0
            except Exception:
                pass
        
        return {
            "valid": True,
            "metadata": asdict(metadata),
            "audio_count": len(audio_files),
            "transcript_count": len(transcript_files),
            "missing_transcripts": list(missing_transcripts),
            "missing_audios": list(missing_audios),
            "actual_duration_seconds": round(total_duration, 2),
            "metadata_duration_seconds": metadata.total_duration_seconds
        }
