# -*- coding: utf-8 -*-

import random
from textual.widgets import Static
from textual.reactive import reactive


class AudioVisualizer(Static):

    DEFAULT_CSS = """
    AudioVisualizer {
        height: 5;
        border: solid #ff79c6;
        background: #282a36;
        padding: 0 1;
    }
    """

    BRAILLE_BLOCKS = " ▁▂▃▄▅▆▇█"

    samples: reactive[list[float]] = reactive(list, always_update=True)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._label = "[bold #ff79c6]◉ AUDIO[/] "

    def on_mount(self) -> None:
        self.samples = [0.0] * 32

    def update_samples(self, samples: list[float]) -> None:
        target_len = 32
        if len(samples) > target_len:
            step = len(samples) / target_len
            samples = [samples[int(i * step)] for i in range(target_len)]
        elif len(samples) < target_len:
            samples = samples + [0.0] * (target_len - len(samples))

        self.samples = [max(0.0, min(1.0, abs(s))) for s in samples]

    def watch_samples(self, samples: list[float]) -> None:
        self._render_visualization()

    def _render_visualization(self) -> None:
        bars = []
        for sample in self.samples:
            idx = int(sample * (len(self.BRAILLE_BLOCKS) - 1))
            bars.append(self.BRAILLE_BLOCKS[idx])

        viz_str = "".join(bars)
        self.update(f"{self._label}[#50fa7b]{viz_str}[/]")

    def simulate(self) -> None:
        fake_samples = [random.uniform(0.0, 1.0) for _ in range(32)]
        for i in range(len(fake_samples)):
            if i > 0:
                fake_samples[i] = 0.7 * fake_samples[i] + 0.3 * fake_samples[i - 1]
        self.update_samples(fake_samples)
