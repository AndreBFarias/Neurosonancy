# -*- coding: utf-8 -*-

import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    success: bool
    audio_path: Optional[Path] = None
    phrase: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    characters_used: int = 0


@dataclass
class VoiceSettings:
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class ElevenLabsClient:

    MODELS = {
        "eleven_multilingual_v2": "Multilingual v2 (Recomendado)",
        "eleven_turbo_v2_5": "Turbo v2.5 (Rapido)",
        "eleven_monolingual_v1": "English v1",
        "eleven_flash_v2": "Flash v2 (Mais rapido)",
        "eleven_flash_v2_5": "Flash v2.5",
    }

    DEFAULT_MODEL = "eleven_multilingual_v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id or self.DEFAULT_MODEL
        self.voice_settings = VoiceSettings()

        self._client = None
        self._is_initialized = False

        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None

    def initialize(self) -> bool:
        if not self.api_key:
            logger.error("API key nao configurada")
            return False

        try:
            from elevenlabs.client import ElevenLabs
            import httpx

            timeout = httpx.Timeout(120.0, connect=30.0)
            self._client = ElevenLabs(api_key=self.api_key, timeout=timeout)
            self._is_initialized = True
            logger.info("Cliente ElevenLabs inicializado (timeout: 120s)")
            return True
        except ImportError:
            logger.error("Pacote elevenlabs nao instalado. Execute: pip install elevenlabs")
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar ElevenLabs: {e}")
            return False

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def set_model(self, model_id: str) -> None:
        if model_id in self.MODELS:
            self.model_id = model_id

    def set_voice_settings(
        self,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> None:
        self.voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
        )

    def generate_audio(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
    ) -> GenerationResult:
        if not self._is_initialized:
            if not self.initialize():
                return GenerationResult(
                    success=False,
                    phrase=text,
                    error="Cliente nao inicializado",
                )

        voice = voice_id or self.voice_id
        if not voice:
            return GenerationResult(
                success=False,
                phrase=text,
                error="Voice ID nao configurado",
            )

        start_time = time.time()

        try:
            from elevenlabs import VoiceSettings as ELVoiceSettings

            audio_generator = self._client.text_to_speech.convert(
                voice_id=voice,
                text=text,
                model_id=self.model_id,
                voice_settings=ELVoiceSettings(
                    stability=self.voice_settings.stability,
                    similarity_boost=self.voice_settings.similarity_boost,
                    style=self.voice_settings.style,
                    use_speaker_boost=self.voice_settings.use_speaker_boost,
                ),
                output_format="pcm_44100",
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            audio_data = b"".join(audio_generator)

            wav_path = output_path.with_suffix(".wav")
            self._save_as_wav(audio_data, wav_path, sample_rate=44100)

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Audio gerado: {wav_path.name} ({duration_ms}ms)")

            return GenerationResult(
                success=True,
                audio_path=wav_path,
                phrase=text,
                duration_ms=duration_ms,
                characters_used=len(text),
            )

        except Exception as e:
            error_msg = str(e)

            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                error_msg = "API Key invalida ou expirada"
            elif "403" in error_msg:
                error_msg = "Acesso negado - verifique permissoes da API Key"
            elif "429" in error_msg:
                error_msg = "Limite de requisicoes excedido - aguarde"
            elif "voice_not_found" in error_msg.lower() or "404" in error_msg:
                error_msg = "Voice ID nao encontrado"
            elif "insufficient" in error_msg.lower():
                error_msg = "Creditos insuficientes na conta ElevenLabs"

            logger.error(f"Erro ao gerar audio: {error_msg}")

            if self.on_error:
                self.on_error(text, error_msg)

            return GenerationResult(
                success=False,
                phrase=text,
                error=error_msg,
            )

    def _save_as_wav(self, pcm_data: bytes, output_path: Path, sample_rate: int = 44100) -> None:
        import struct

        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(pcm_data)
        chunk_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            chunk_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size,
        )

        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(pcm_data)

    def get_voices(self) -> Dict[str, str]:
        if not self._is_initialized:
            if not self.initialize():
                return {}

        try:
            response = self._client.voices.get_all()
            voices = {}
            for voice in response.voices:
                voices[voice.voice_id] = voice.name
            return voices
        except Exception as e:
            logger.error(f"Erro ao listar vozes: {e}")
            return {}

    def get_subscription_info(self) -> Dict[str, Any]:
        if not self._is_initialized:
            if not self.initialize():
                return {}

        try:
            info = self._client.user.get_subscription()
            return {
                "tier": info.tier,
                "character_count": info.character_count,
                "character_limit": info.character_limit,
                "characters_remaining": info.character_limit - info.character_count,
            }
        except Exception as e:
            logger.error(f"Erro ao obter info da subscription: {e}")
            return {}

    def validate_api_key(self) -> bool:
        if not self.api_key:
            return False

        try:
            if not self._is_initialized:
                if not self.initialize():
                    return False

            self._client.user.get_subscription()
            return True
        except Exception:
            return False
