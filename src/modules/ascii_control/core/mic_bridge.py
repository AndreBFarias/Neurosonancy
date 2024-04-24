# -*- coding: utf-8 -*-

import threading
import time
import logging
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not available - mic input disabled")


class MicBridge:

    def __init__(
        self,
        audio_callback: Optional[Callable[[np.ndarray], None]] = None,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ):
        self._audio_callback = audio_callback
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None

        logger.info("MicBridge initialized")

    @property
    def is_listening(self) -> bool:
        return self._running

    def set_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        self._audio_callback = callback

    def _audio_stream_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug(f"Mic status: {status}")

        if self._audio_callback and self._running:
            try:
                audio_data = indata[:, 0].copy()
                self._audio_callback(audio_data)
            except Exception as e:
                logger.debug(f"Callback error: {e}")

    def start(self) -> str:
        if not SOUNDDEVICE_AVAILABLE:
            return "Mic unavailable: sounddevice not installed"

        if self._running:
            return "Already listening"

        try:
            self._running = True
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self._chunk_size,
                callback=self._audio_stream_callback
            )
            self._stream.start()
            logger.info("Mic listening started")
            return "Mic: [ON] Listening..."

        except Exception as e:
            self._running = False
            logger.error(f"Mic start failed: {e}")
            return f"Mic error: {e}"

    def stop(self) -> str:
        if not self._running:
            return "Mic already off"

        self._running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        logger.info("Mic listening stopped")
        return "Mic: [OFF]"

    def toggle(self) -> str:
        if self._running:
            return self.stop()
        return self.start()


class SimulatedMic:

    def __init__(self, audio_callback: Optional[Callable[[np.ndarray], None]] = None):
        self._audio_callback = audio_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._phase = 0.0

    @property
    def is_listening(self) -> bool:
        return self._running

    def set_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        self._audio_callback = callback

    def _generate_loop(self) -> None:
        while self._running:
            t = np.linspace(0, 0.064, 1024)
            freq = 220 + 50 * np.sin(self._phase * 0.1)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            audio += 0.1 * np.random.randn(len(t))
            self._phase += 1

            if self._audio_callback:
                try:
                    self._audio_callback(audio.astype(np.float32))
                except Exception:
                    pass

            time.sleep(0.05)

    def start(self) -> str:
        if self._running:
            return "Already simulating"

        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        return "Mic: [SIMULATED]"

    def stop(self) -> str:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return "Mic: [OFF]"

    def toggle(self) -> str:
        if self._running:
            return self.stop()
        return self.start()


def create_mic_bridge(
    callback: Optional[Callable[[np.ndarray], None]] = None,
    use_real: bool = True
):
    if use_real and SOUNDDEVICE_AVAILABLE:
        return MicBridge(audio_callback=callback)
    return SimulatedMic(audio_callback=callback)
