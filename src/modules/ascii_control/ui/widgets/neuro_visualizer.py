# -*- coding: utf-8 -*-

import math
import random
import numpy as np
from typing import Optional, List
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
from rich.style import Style


class NeuroVisualizer(Static):

    DEFAULT_CSS = """
    NeuroVisualizer {
        height: 7;
        border: heavy #bd93f9;
        background: #282a36;
        padding: 0 1;
    }

    NeuroVisualizer.active {
        border: heavy #50fa7b;
    }

    NeuroVisualizer.acid {
        border: double #ff5555;
    }
    """

    GRADIENT_BASS = ["#6272a4", "#bd93f9", "#bd93f9"]
    GRADIENT_MID = ["#8be9fd", "#8be9fd", "#50fa7b"]
    GRADIENT_HIGH = ["#ff79c6", "#ff79c6", "#ffb86c"]

    CHARS_INTENSITY = ["·", "╌", "─", "━", "█"]
    GLITCH_CHARS = ["░", "▒", "▓", "█", "▄", "▀", "▐", "▌"]

    is_active: reactive[bool] = reactive(False)
    acid_mode: reactive[bool] = reactive(False)
    bpm: reactive[int] = reactive(138)

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__("[dim]Initializing...[/]", name=name, id=id, classes=classes)
        self._smoothed_rms = 0.0
        self._smoothed_energies = [0.0] * 8
        self._phase = 0.0
        self._beat_phase = 0.0
        self._width = 60
        self._height = 5

    def on_mount(self) -> None:
        self._width = max(self.size.width - 2, 40)
        self._height = max(self.size.height - 2, 3)
        self._render_wave()

    def update_audio(self, audio_chunk: np.ndarray) -> None:
        if len(audio_chunk) == 0:
            return

        data = audio_chunk.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

        rms = float(np.sqrt(np.mean(data ** 2)))
        self._smoothed_rms = self._smoothed_rms * 0.7 + rms * 0.3

        try:
            window = np.hanning(len(data))
            fft_data = np.abs(np.fft.rfft(data * window))

            n_bins = len(fft_data)
            if n_bins > 8:
                chunk_size = n_bins // 8
                energies = []
                for i in range(8):
                    start = i * chunk_size
                    end = start + chunk_size
                    energies.append(float(np.mean(fft_data[start:end])))

                for i in range(8):
                    self._smoothed_energies[i] = (
                        self._smoothed_energies[i] * 0.6 + energies[i] * 0.4
                    )
        except Exception:
            pass

        self._render_wave()

    def simulate(self) -> None:
        self._beat_phase += 0.15
        beat_factor = 0.5 + 0.5 * math.sin(self._beat_phase)

        fake_rms = 0.3 + 0.4 * beat_factor + random.uniform(-0.05, 0.05)
        self._smoothed_rms = self._smoothed_rms * 0.7 + fake_rms * 0.3

        for i in range(8):
            if i < 3:
                base = 0.6 * beat_factor
            elif i < 6:
                base = 0.4
            else:
                base = 0.2 + 0.3 * (1 - beat_factor)

            self._smoothed_energies[i] = (
                self._smoothed_energies[i] * 0.6 +
                (base + random.uniform(-0.1, 0.1)) * 0.4
            )

        self._render_wave()

    def _get_color_for_x(self, x: int, width: int) -> str:
        pos = x / max(width - 1, 1)

        bass = sum(self._smoothed_energies[:3]) / 3
        mid = sum(self._smoothed_energies[3:6]) / 3
        high = sum(self._smoothed_energies[6:]) / 2

        total = bass + mid + high + 0.001
        bass_w = bass / total
        mid_w = mid / total
        high_w = high / total

        if bass_w > mid_w and bass_w > high_w:
            gradient = self.GRADIENT_BASS
        elif high_w > mid_w:
            gradient = self.GRADIENT_HIGH
        else:
            gradient = self.GRADIENT_MID

        idx = int(pos * (len(gradient) - 1))
        return gradient[idx]

    def _get_char_for_intensity(self, intensity: float) -> str:
        if self.acid_mode and random.random() < 0.15:
            return random.choice(self.GLITCH_CHARS)

        idx = int(intensity * (len(self.CHARS_INTENSITY) - 1))
        idx = max(0, min(idx, len(self.CHARS_INTENSITY) - 1))
        return self.CHARS_INTENSITY[idx]

    def _render_wave(self) -> None:
        try:
            width = self._width
            height = self._height
            center_y = height / 2

            amplitude = self._smoothed_rms * height * 1.5
            self._phase += 0.12

            lines: List[List[tuple]] = [[(" ", "#282a36")] * width for _ in range(height)]

            for x in range(width):
                freq_mod = 1.0 + self._smoothed_energies[x % 8] * 0.5
                wave_y = math.sin(x * 0.15 * freq_mod + self._phase) * amplitude

                sidechain = 0.5 + 0.5 * math.sin(self._beat_phase * 2)
                wave_y *= sidechain

                y_pos = int(center_y + wave_y)
                y_pos = max(0, min(height - 1, y_pos))

                intensity = min(1.0, self._smoothed_rms * 2)
                char = self._get_char_for_intensity(intensity)
                color = self._get_color_for_x(x, width)

                lines[y_pos][x] = (char, color)

                if amplitude > 1.0:
                    y_above = max(0, y_pos - 1)
                    y_below = min(height - 1, y_pos + 1)
                    dim_color = "#6272a4"
                    if lines[y_above][x][0] == " ":
                        lines[y_above][x] = ("·", dim_color)
                    if lines[y_below][x][0] == " ":
                        lines[y_below][x] = ("·", dim_color)

            text = Text()

            status = "[#50fa7b]●[/]" if self.is_active else "[#6272a4]○[/]"
            acid_str = " [#ff5555]ACID[/]" if self.acid_mode else ""
            header = f"[bold #ff79c6]♫ NEURO[/] {status} [#8be9fd]BPM:{self.bpm}[/]{acid_str}\n"
            text.append_text(Text.from_markup(header))

            for row in lines:
                for char, color in row:
                    text.append(char, style=Style(color=color))
                text.append("\n")

            self.update(text)

        except Exception:
            pass

    def set_acid_mode(self, enabled: bool) -> None:
        self.acid_mode = enabled
        self.remove_class("acid")
        if enabled:
            self.add_class("acid")

    def watch_is_active(self, active: bool) -> None:
        self.remove_class("active")
        if active:
            self.add_class("active")
