# -*- coding: utf-8 -*-

import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHUNK_SIZE = 1024


class Mood(Enum):
    CHILL = "chill"
    NORMAL = "normal"
    INTENSE = "intense"
    DARK = "dark"


@dataclass
class SynthParams:
    bpm: int = 138
    base_freq: float = 196.00
    detune: float = 0.02
    attack: float = 0.01
    release: float = 0.3
    filter_cutoff: float = 0.8
    noise_level: float = 0.0
    sidechain_depth: float = 0.7


MOOD_PRESETS: Dict[Mood, SynthParams] = {
    Mood.CHILL: SynthParams(bpm=120, detune=0.01, attack=0.05, sidechain_depth=0.4),
    Mood.NORMAL: SynthParams(bpm=138, detune=0.02, attack=0.02, sidechain_depth=0.6),
    Mood.INTENSE: SynthParams(bpm=150, detune=0.05, attack=0.005, sidechain_depth=0.85),
    Mood.DARK: SynthParams(bpm=130, detune=0.08, attack=0.01, filter_cutoff=0.4, sidechain_depth=0.7),
}


class SuperSawOscillator:

    def __init__(self, num_voices: int = 7):
        self.num_voices = num_voices
        self._phase = np.zeros(num_voices)

    def generate(self, freq: float, detune: float, num_samples: int) -> np.ndarray:
        output = np.zeros(num_samples)
        spread = np.linspace(-detune, detune, self.num_voices)

        for i, det in enumerate(spread):
            voice_freq = freq * (1.0 + det)
            phase_inc = 2.0 * np.pi * voice_freq / SAMPLE_RATE
            phases = self._phase[i] + np.arange(num_samples) * phase_inc
            saw = 2.0 * (phases / (2.0 * np.pi) % 1.0) - 1.0
            output += saw
            self._phase[i] = phases[-1] % (2.0 * np.pi)

        return output / self.num_voices


class SidechainCompressor:

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._phase = 0.0

    def apply(self, audio: np.ndarray, bpm: int, depth: float) -> np.ndarray:
        beat_samples = int(self.sample_rate * 60 / bpm)
        num_samples = len(audio)

        envelope = np.zeros(num_samples)
        for i in range(num_samples):
            pos = (self._phase + i) % beat_samples
            t = pos / beat_samples
            envelope[i] = 1.0 - depth * np.exp(-t * 8)

        self._phase = (self._phase + num_samples) % beat_samples
        return audio * envelope


class LowPassFilter:

    def __init__(self):
        self._prev = 0.0

    def apply(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        alpha = np.clip(cutoff, 0.01, 1.0)
        output = np.zeros_like(audio)
        prev = self._prev

        for i in range(len(audio)):
            output[i] = prev + alpha * (audio[i] - prev)
            prev = output[i]

        self._prev = prev
        return output


class NeurosonancyMusic:

    def __init__(self, metrics_getter: Optional[Callable[[], Dict[str, Any]]] = None):
        self._metrics_getter = metrics_getter
        self._params = SynthParams()
        self._mood = Mood.NORMAL

        self._osc = SuperSawOscillator()
        self._sidechain = SidechainCompressor()
        self._lpf = LowPassFilter()

        self._is_playing = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        self._audio_buffer: deque = deque(maxlen=64)
        self._waveform_callback: Optional[Callable[[list], None]] = None

        self._note_sequence = [0, 0, 7, 7, 5, 5, 3, 3]
        self._note_idx = 0

        logger.info("NeurosonancyMusic initialized")

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_bpm(self) -> int:
        return self._params.bpm

    @property
    def current_mood(self) -> str:
        return self._mood.value

    def set_waveform_callback(self, callback: Callable[[list], None]) -> None:
        self._waveform_callback = callback

    def set_mood(self, mood_name: str) -> bool:
        try:
            mood = Mood(mood_name.lower())
            self._mood = mood
            preset = MOOD_PRESETS[mood]
            self._params.bpm = preset.bpm
            self._params.detune = preset.detune
            self._params.attack = preset.attack
            self._params.sidechain_depth = preset.sidechain_depth
            self._params.filter_cutoff = preset.filter_cutoff
            logger.info(f"Mood changed to: {mood_name}")
            return True
        except ValueError:
            return False

    def set_bpm(self, bpm: int) -> bool:
        if 60 <= bpm <= 200:
            self._params.bpm = bpm
            logger.info(f"BPM set to: {bpm}")
            return True
        return False

    def _update_from_metrics(self) -> None:
        if not self._metrics_getter:
            return

        try:
            metrics = self._metrics_getter()

            latency = metrics.get("latency", {})
            llm_avg = latency.get("llm", {}).get("avg", 0)
            self._params.detune = MOOD_PRESETS[self._mood].detune + (llm_avg * 0.02)

            api = metrics.get("api", {})
            total = api.get("total_requests", 1)
            failed = api.get("failed", 0)
            fail_rate = failed / max(total, 1)
            self._params.noise_level = min(fail_rate * 0.3, 0.15)

            if api.get("circuit_open", False):
                self._params.filter_cutoff = 0.2
            else:
                self._params.filter_cutoff = MOOD_PRESETS[self._mood].filter_cutoff

            queues = metrics.get("queues", {})
            proc = queues.get("processing", {})
            size = proc.get("size", 0)
            maxsize = proc.get("maxsize", 20)
            pressure = size / max(maxsize, 1)
            self._params.attack = max(0.001, MOOD_PRESETS[self._mood].attack * (1 - pressure * 0.8))

        except Exception as e:
            logger.debug(f"Metrics update failed: {e}")

    def _generate_chunk(self) -> np.ndarray:
        self._update_from_metrics()

        semitone = self._note_sequence[self._note_idx]
        freq = self._params.base_freq * (2 ** (semitone / 12.0))
        self._note_idx = (self._note_idx + 1) % len(self._note_sequence)

        audio = self._osc.generate(freq, self._params.detune, CHUNK_SIZE)

        if self._params.noise_level > 0:
            noise = np.random.randn(CHUNK_SIZE) * self._params.noise_level
            audio = audio + noise

        audio = self._lpf.apply(audio, self._params.filter_cutoff)
        audio = self._sidechain.apply(audio, self._params.bpm, self._params.sidechain_depth)

        audio = np.clip(audio * 0.5, -1.0, 1.0)

        return audio

    def _audio_thread(self) -> None:
        logger.info("Audio thread started")

        try:
            import sounddevice as sd
            has_audio = True
        except ImportError:
            has_audio = False
            logger.warning("sounddevice not available - visual only mode")

        stream = None
        if has_audio:
            try:
                stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=CHUNK_SIZE
                )
                stream.start()
            except Exception as e:
                logger.error(f"Audio output failed: {e}")
                has_audio = False

        beat_interval = 60.0 / self._params.bpm / 4

        while not self._shutdown.is_set():
            if not self._is_playing:
                time.sleep(0.1)
                continue

            start = time.perf_counter()

            chunk = self._generate_chunk()

            if has_audio and stream:
                try:
                    stream.write(chunk.astype(np.float32))
                except Exception:
                    pass

            waveform = chunk[::CHUNK_SIZE // 32].tolist()
            self._audio_buffer.append(waveform)

            if self._waveform_callback:
                try:
                    self._waveform_callback(waveform)
                except Exception:
                    pass

            elapsed = time.perf_counter() - start
            sleep_time = max(0, beat_interval - elapsed)
            time.sleep(sleep_time)

        if stream:
            stream.stop()
            stream.close()

        logger.info("Audio thread stopped")

    def start(self) -> str:
        if self._is_playing:
            return "Already playing"

        self._is_playing = True
        self._shutdown.clear()

        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._audio_thread, daemon=True)
            self._thread.start()

        return f"Trance Engine: [START] BPM={self._params.bpm} Mood={self._mood.value}"

    def stop(self) -> str:
        self._is_playing = False
        return "Trance Engine: [STOP]"

    def shutdown(self) -> None:
        self._is_playing = False
        self._shutdown.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_waveform(self) -> list:
        if self._audio_buffer:
            return list(self._audio_buffer[-1])
        return [0.0] * 32

    def execute_command(self, cmd: str) -> Optional[str]:
        parts = cmd.lower().split()
        if not parts:
            return None

        action = parts[0]

        if action in ("play", "start"):
            return self.start()

        elif action == "stop":
            return self.stop()

        elif action == "mood" and len(parts) > 1:
            if self.set_mood(parts[1]):
                return f"Mood set to: {parts[1]}"
            return f"Unknown mood: {parts[1]} (try: chill, normal, intense, dark)"

        elif action == "bpm" and len(parts) > 1:
            try:
                bpm = int(parts[1])
                if self.set_bpm(bpm):
                    return f"BPM set to: {bpm}"
                return "BPM must be 60-200"
            except ValueError:
                return "Invalid BPM value"

        elif action == "status":
            return f"Engine: {'ON' if self._is_playing else 'OFF'} | BPM: {self._params.bpm} | Mood: {self._mood.value}"

        return None
