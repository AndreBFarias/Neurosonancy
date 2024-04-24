# -*- coding: utf-8 -*-

import random
from typing import Optional, Callable
from textual.widgets import Static
from textual.reactive import reactive


class WaveformVisualizer(Static):

    DEFAULT_CSS = """
    WaveformVisualizer {
        height: 5;
        border: heavy #bd93f9;
        background: #282a36;
        padding: 0 1;
    }

    WaveformVisualizer.playing {
        border: heavy #50fa7b;
    }

    WaveformVisualizer.intense {
        border: heavy #ff5555;
    }
    """

    BLOCK_CHARS = " ▁▂▃▄▅▆▇█"
    BLOCK_CHARS_NEG = "█▇▆▅▄▃▂▁ "

    samples: reactive[list] = reactive(list, always_update=True)
    is_playing: reactive[bool] = reactive(False)
    bpm: reactive[int] = reactive(138)
    mood: reactive[str] = reactive("normal")

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__("[dim]Initializing...[/]", name=name, id=id, classes=classes)
        self._target_width = 48
        self._samples: list = [0.0] * self._target_width

    def on_mount(self) -> None:
        self._do_render()

    def update_waveform(self, samples: list) -> None:
        if len(samples) < self._target_width:
            factor = self._target_width // max(len(samples), 1)
            expanded = []
            for s in samples:
                expanded.extend([s] * factor)
            samples = expanded[:self._target_width]
        elif len(samples) > self._target_width:
            step = len(samples) / self._target_width
            samples = [samples[int(i * step)] for i in range(self._target_width)]

        self._samples = samples
        self._do_render()

    def watch_samples(self, samples: list) -> None:
        pass

    def watch_is_playing(self, playing: bool) -> None:
        self.remove_class("playing", "intense")
        if playing:
            self.add_class("playing")
            if self.mood == "intense":
                self.add_class("intense")
        self._do_render()

    def watch_mood(self, mood: str) -> None:
        self.remove_class("intense")
        if self.is_playing and mood == "intense":
            self.add_class("intense")

    def _do_render(self) -> None:
        samples = self._samples
        if not samples:
            self.update("[dim]No data[/]")
            return

        top_row = []
        bot_row = []

        for val in samples:
            val = max(-1.0, min(1.0, val))

            if val >= 0:
                idx = int(val * (len(self.BLOCK_CHARS) - 1))
                top_row.append(self.BLOCK_CHARS[idx])
                bot_row.append(" ")
            else:
                idx = int(-val * (len(self.BLOCK_CHARS) - 1))
                top_row.append(" ")
                bot_row.append(self.BLOCK_CHARS[idx])

        top_str = "".join(top_row)
        bot_str = "".join(bot_row)

        if self.is_playing:
            status = f"[#50fa7b]▶ PLAYING[/]"
            color = "#50fa7b" if self.mood != "intense" else "#ff5555"
        else:
            status = "[#6272a4]◼ STOPPED[/]"
            color = "#6272a4"

        info = f"[#8be9fd]BPM:{self.bpm}[/] [#ffb86c]{self.mood.upper()}[/]"

        content = (
            f"[bold #ff79c6]♫ SYNTH[/] {status} {info}\n"
            f"[{color}]{top_str}[/]\n"
            f"[{color}]{bot_str}[/]"
        )

        self.update(content)

    def simulate(self) -> None:
        if self.is_playing:
            fake = [random.uniform(-0.8, 0.8) for _ in range(self._target_width)]
            for i in range(1, len(fake)):
                fake[i] = 0.5 * fake[i] + 0.5 * fake[i - 1]
            self.update_waveform(fake)
        else:
            self.update_waveform([0.0] * self._target_width)
