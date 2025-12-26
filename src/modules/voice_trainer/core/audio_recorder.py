# -*- coding: utf-8 -*-

import logging
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import sounddevice as sd
import scipy.io.wavfile as wavfile

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logging.warning("webrtcvad not available, using energy-based VAD only")

logger = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_id: Optional[int] = None,
        vad_silence_duration: float = 2.0,
        vad_energy_threshold: int = 300,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.vad_silence_duration = vad_silence_duration
        self.vad_energy_threshold = vad_energy_threshold
        
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.silence_start_time = None
        
        if HAS_WEBRTCVAD and sample_rate in [8000, 16000, 32000, 48000]:
            self.vad = webrtcvad.Vad(1)
            self.use_webrtc = True
        else:
            self.vad = None
            self.use_webrtc = False
            
        self.visualization_callback: Optional[Callable] = None
        self.auto_stop_callback: Optional[Callable] = None
        
        logger.info(f"AudioRecorder initialized (SR: {sample_rate}Hz, Device: {device_id})")
        logger.info(f"VAD: {'WebRTC' if self.use_webrtc else 'Energy-based'} (threshold: {vad_energy_threshold})")
    
    def set_visualization_callback(self, callback: Callable[[np.ndarray], None]):
        self.visualization_callback = callback
    
    def set_auto_stop_callback(self, callback: Callable[[], None]):
        self.auto_stop_callback = callback
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if not self.is_recording:
            return
        
        audio_chunk = indata[:, 0].copy() if self.channels == 1 else indata.copy()
        self.audio_data.append(audio_chunk)
        
        if self.visualization_callback:
            try:
                self.visualization_callback(audio_chunk)
            except Exception as e:
                logger.error(f"Visualization callback error: {e}")
        
        is_speech = self._detect_speech(audio_chunk)
        
        if is_speech:
            self.silence_start_time = None
        else:
            if self.silence_start_time is None:
                self.silence_start_time = time.time()
            elif time.time() - self.silence_start_time >= self.vad_silence_duration:
                logger.info(f"Silence detected for {self.vad_silence_duration}s, auto-stopping")
                
                if self.auto_stop_callback:
                    try:
                        self.auto_stop_callback()
                    except Exception as e:
                        logger.error(f"Auto-stop callback error: {e}")
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_int16 = (audio_chunk * 32768).astype(np.int16)
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        
        energy_speech = rms > self.vad_energy_threshold
        
        if not self.use_webrtc:
            return energy_speech
        
        try:
            frame_bytes = audio_int16.tobytes()
            
            if len(frame_bytes) < 320:
                return energy_speech
            
            frame_duration_ms = int(len(audio_chunk) / self.sample_rate * 1000)
            if frame_duration_ms not in [10, 20, 30]:
                return energy_speech
            
            webrtc_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            return energy_speech and webrtc_speech
            
        except Exception as e:
            logger.debug(f"WebRTC VAD error (using energy only): {e}")
            return energy_speech
    
    def start_recording(self, output_path: str) -> bool:
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        try:
            self.audio_data = []
            self.is_recording = True
            self.silence_start_time = None
            self.output_path = output_path
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_id,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.03),
                dtype='float32'
            )
            
            self.stream.start()
            logger.info(f"Recording started: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True)
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[str]:
        if not self.is_recording:
            logger.warning("Not currently recording, nothing to stop")
            return None
        
        self.is_recording = False
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            if not self.audio_data:
                logger.warning("No audio data recorded")
                return None
            
            audio = np.concatenate(self.audio_data)
            
            audio_int16 = (audio * 32768).astype(np.int16)
            
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            wavfile.write(self.output_path, self.sample_rate, audio_int16)
            
            duration = len(audio) / self.sample_rate
            logger.info(f"Recording saved: {self.output_path} ({duration:.2f}s)")
            
            return self.output_path
            
        except Exception as e:
            logger.error(f"Failed to save recording: {e}", exc_info=True)
            return None
    
    def play_audio(self, audio_path: str) -> bool:
        try:
            sample_rate, data = wavfile.read(audio_path)
            
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            sd.play(data, sample_rate)
            sd.wait()
            
            logger.info(f"Playback completed: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}", exc_info=True)
            return False
    
    def test_device(self):
        print("=" * 60)
        print("AUDIO DEVICE TEST")
        print("=" * 60)
        print()
        
        if self.device_id is not None:
            try:
                device_info = sd.query_devices(self.device_id)
                print(f"Device [{self.device_id}]: {device_info['name']}")
                print(f"  Max Input Channels: {device_info['max_input_channels']}")
                print(f"  Default Sample Rate: {device_info['default_samplerate']}Hz")
                print()
            except Exception as e:
                print(f"Error querying device: {e}")
                return False
        else:
            print("Using default device")
            print()
        
        print(f"Testing recording for 2 seconds...")
        print(f"Configuration: {self.sample_rate}Hz, {self.channels} channel(s)")
        print()
        
        test_path = "/tmp/test_recording.wav"
        
        if not self.start_recording(test_path):
            print("Failed to start recording")
            return False
        
        time.sleep(2)
        
        saved_path = self.stop_recording()
        
        if saved_path:
            print(f"✓ Recording successful: {saved_path}")
            
            sample_rate, data = wavfile.read(saved_path)
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
            print(f"  Duration: {len(data) / sample_rate:.2f}s")
            print(f"  RMS: {rms:.2f}")
            print()
            print("AUDIO DEVICE TEST: PASSED")
            return True
        else:
            print("✗ Recording failed")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        device_id=9,
        vad_silence_duration=2.0,
        vad_energy_threshold=300
    )
    
    recorder.test_device()
