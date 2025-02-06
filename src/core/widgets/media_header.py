# -*- coding: utf-8 -*-

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static, Select, ProgressBar

from src.core.theme import COLORS
from src.core.media_bridge import TrackInfo


class MediaHeader(Widget):

    track_title: reactive[str] = reactive("-- Sem player --")
    track_artist: reactive[str] = reactive("")
    is_playing: reactive[bool] = reactive(False)
    volume: reactive[float] = reactive(0.8)
    current_model: reactive[str] = reactive("")
    active_module: reactive[str] = reactive("monitor")

    _MODULE_LABELS: dict[str, str] = {
        "monitor": "MONITOR",
        "trainer": "VOICE TRAINER",
        "clone": "CLONE VOICE",
    }

    class PlayPauseRequested(Message):
        pass

    class NextTrackRequested(Message):
        pass

    class PrevTrackRequested(Message):
        pass

    class VolumeChanged(Message):
        def __init__(self, volume: float) -> None:
            super().__init__()
            self.volume = volume

    class ModelChangeRequested(Message):
        def __init__(self, model_display_name: str) -> None:
            super().__init__()
            self.model_display_name = model_display_name

    DEFAULT_CSS = f"""
    MediaHeader {{
        height: 7;
        background: {COLORS['bg_surface']};
        border: solid {COLORS['border_normal']};
        border-title-color: {COLORS['accent_tertiary']};
        border-title-style: bold;
        dock: top;
        layout: horizontal;
        padding: 0 2;
        align: left middle;
    }}

    MediaHeader #track-info {{
        width: 1fr;
        height: 100%;
        content-align: left middle;
        padding: 0 2;
        color: {COLORS['text_primary']};
    }}

    MediaHeader #controls {{
        width: auto;
        height: 100%;
        align: center middle;
        padding: 0 1;
    }}

    MediaHeader #controls Button {{
        min-width: 5;
        height: 3;
        margin: 0 1;
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['border_normal']};
        color: {COLORS['text_primary']};
    }}

    MediaHeader #controls Button:hover {{
        background: {COLORS['accent_primary']};
        color: {COLORS['bg_darkest']};
        border: solid {COLORS['accent_primary']};
    }}

    MediaHeader #model-select {{
        width: 30;
        height: 3;
        margin: 0 1;
    }}

    MediaHeader #track-progress {{
        width: 18;
        height: 1;
        margin: 0 2;
    }}

    MediaHeader #track-progress > .bar--bar {{
        color: {COLORS['border_normal']};
    }}

    MediaHeader #track-progress > .bar--complete {{
        color: {COLORS['neon_cyan']};
    }}

    MediaHeader #vol-label {{
        width: 4;
        height: 1;
        color: {COLORS['text_muted']};
        margin: 0 0;
        content-align: right middle;
    }}

    MediaHeader #vol-bar {{
        width: 10;
        height: 1;
        margin: 0 1;
    }}

    MediaHeader #vol-bar > .bar--bar {{
        color: {COLORS['border_normal']};
    }}

    MediaHeader #vol-bar > .bar--complete {{
        color: {COLORS['accent_primary']};
    }}
    """

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal
        yield Static("", id="track-info")
        with Horizontal(id="controls"):
            yield Button("|<", id="btn-prev", tooltip="Anterior")
            yield Button("||", id="btn-playpause", tooltip="Play/Pause")
            yield Button(">|", id="btn-next", tooltip="Proxima")
            yield ProgressBar(total=100, show_eta=False, id="track-progress")
        yield Static("VOL", id="vol-label")
        yield ProgressBar(total=100, show_eta=False, id="vol-bar")
        yield Select(
            options=[],
            id="model-select",
            prompt="-- Modelo --",
        )

    def on_mount(self) -> None:
        self.border_title = self._MODULE_LABELS.get(self.active_module, "MONITOR")
        self._refresh_track_display()
        self._populate_models()
        self.watch_volume(self.volume)

    def _populate_models(self) -> None:
        try:
            from src.core.model_manager import ModelManager
            names = ModelManager().get_display_names()
            sel = self.query_one("#model-select", Select)
            sel.set_options([(name, name) for name in names])
        except Exception:
            pass

    def watch_track_title(self, value: str) -> None:
        self._refresh_track_display()

    def watch_track_artist(self, value: str) -> None:
        self._refresh_track_display()

    def watch_is_playing(self, value: bool) -> None:
        try:
            btn = self.query_one("#btn-playpause", Button)
            btn.label = "||" if value else " >"
        except Exception:
            pass

    def watch_volume(self, value: float) -> None:
        try:
            self.query_one("#vol-bar", ProgressBar).update(progress=int(value * 100))
        except Exception:
            pass

    def watch_active_module(self, value: str) -> None:
        label = self._MODULE_LABELS.get(value, value.upper())
        self.border_title = label

    def _refresh_track_display(self) -> None:
        try:
            info = self.query_one("#track-info", Static)
            if self.track_artist:
                text = (
                    f"[bold {COLORS['text_primary']}]{self.track_title}[/] "
                    f"[{COLORS['text_muted']}]— {self.track_artist}[/]"
                )
            else:
                text = f"[{COLORS['text_muted']}]{self.track_title}[/]"
            info.update(text)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-playpause":
            self.post_message(self.PlayPauseRequested())
        elif btn_id == "btn-next":
            self.post_message(self.NextTrackRequested())
        elif btn_id == "btn-prev":
            self.post_message(self.PrevTrackRequested())

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select" and event.value:
            self.post_message(self.ModelChangeRequested(str(event.value)))

    def update_track(self, info: TrackInfo) -> None:
        self.track_title = info.title or "-- Sem faixa --"
        self.track_artist = info.artist
        if info.duration_us > 0 and info.position_us >= 0:
            try:
                progress = int(info.position_us / info.duration_us * 100)
                self.query_one("#track-progress", ProgressBar).update(progress=progress)
            except Exception:
                pass


# "O que retarda o progresso externo não retarda o progresso interior." — Marco Aurélio
