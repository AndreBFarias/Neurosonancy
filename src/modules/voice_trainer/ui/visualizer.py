# -*- coding: utf-8 -*-

import logging
import numpy as np
from textual.widgets import Static
from rich.text import Text
from rich.style import Style

logger = logging.getLogger(__name__)


class AudioVisualizer(Static):
    BAR_CHARS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bars = 30
        self.height_scale = 8
        self.data_buffer = np.zeros(512)
        self.is_recording = False
    
    def set_recording(self, recording: bool):
        self.is_recording = recording
        if not recording:
            self.update("")
    
    def update_audio(self, audio_chunk: np.ndarray):
        if not self.is_recording:
            return
        
        try:
            if len(audio_chunk) < 64:
                return
            
            data = audio_chunk.astype(np.float32)
            
            window = np.hanning(len(data))
            fft_data = np.fft.rfft(data * window)
            magnitude = np.abs(fft_data)
            
            magnitude = np.log10(magnitude + 1)
            
            magnitude = magnitude[4:]
            
            if len(magnitude) < self.bars:
                return
            
            chunk_size = len(magnitude) // self.bars
            sliced = magnitude[:chunk_size * self.bars].reshape(self.bars, chunk_size)
            energy = np.mean(sliced, axis=1)
            
            max_val = np.max(energy)
            if max_val > 0:
                energy = energy / max_val
            
            self._render_bars(energy)
            
        except Exception as e:
            logger.error(f"Visualizer error: {e}")
    
    def _render_bars(self, energy_array):
        w, h = self.size.width, self.size.height
        if w == 0 or h == 0:
            return
        
        num_bars = min(w // 2, len(energy_array))
        
        if num_bars != len(energy_array):
            energy_array = np.interp(
                np.linspace(0, len(energy_array), num_bars),
                np.arange(len(energy_array)),
                energy_array
            )
        
        rich_text = Text()
        
        for val in energy_array:
            level = int(val * (len(self.BAR_CHARS) - 1))
            char = self.BAR_CHARS[level]
            
            if val > 0.8:
                color = "#ff0055"
            elif val > 0.5:
                color = "#00ccff"
            elif val > 0.2:
                color = "#50fa7b"
            else:
                color = "#444444"
            
            rich_text.append(char, style=Style(color=color))
            rich_text.append(" ")
        
        self.update(rich_text)
