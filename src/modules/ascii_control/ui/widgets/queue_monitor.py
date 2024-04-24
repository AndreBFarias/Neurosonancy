# -*- coding: utf-8 -*-

from typing import Dict, Any
from textual.widgets import Static, ProgressBar
from textual.containers import Vertical


class QueueBar(Static):

    DEFAULT_CSS = """
    QueueBar {
        height: 2;
        margin-bottom: 1;
    }

    QueueBar ProgressBar {
        height: 1;
    }

    QueueBar ProgressBar Bar {
        color: #50fa7b;
    }

    QueueBar.warning ProgressBar Bar {
        color: #ffb86c;
    }

    QueueBar.critical ProgressBar Bar {
        color: #ff5555;
    }
    """

    def __init__(
        self,
        label: str,
        maxsize: int = 100,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._label = label
        self._maxsize = maxsize
        self._progress: ProgressBar | None = None

    def compose(self):
        yield Static(f"[bold #8be9fd]{self._label}[/]")
        self._progress = ProgressBar(total=self._maxsize, show_eta=False)
        yield self._progress

    def update_value(self, current: int, maxsize: int | None = None) -> None:
        if maxsize:
            self._maxsize = maxsize
            if self._progress:
                self._progress.total = maxsize

        if self._progress:
            self._progress.progress = current

        ratio = current / self._maxsize if self._maxsize > 0 else 0

        self.remove_class("warning", "critical")
        if ratio >= 0.9:
            self.add_class("critical")
        elif ratio >= 0.7:
            self.add_class("warning")


class QueueMonitor(Vertical):

    DEFAULT_CSS = """
    QueueMonitor {
        border: solid #6272a4;
        background: #44475a;
        padding: 1;
    }
    """

    def __init__(
        self,
        queues: list[tuple[str, str, int]] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._queues = queues or [
            ("processing", "Processing", 20),
            ("audio_input", "Audio In", 100),
            ("transcription", "Transcription", 50),
            ("tts", "TTS", 30),
        ]
        self._bars: Dict[str, QueueBar] = {}

    def compose(self):
        yield Static("[bold #8be9fd]FILAS / BACKPRESSURE[/]")
        for queue_id, label, maxsize in self._queues:
            bar = QueueBar(label, maxsize, id=f"q_{queue_id}")
            self._bars[queue_id] = bar
            yield bar

    def update_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        for queue_id, data in stats.items():
            if queue_id in self._bars:
                self._bars[queue_id].update_value(
                    data.get("size", 0),
                    data.get("maxsize")
                )
