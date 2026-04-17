# -*- coding: utf-8 -*-

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, RichLog, Input, Button
from textual.containers import VerticalScroll, Horizontal
from textual.binding import Binding

from src.core.theme import COLORS
from src.core.media_bridge import MPRISBridge, PlaybackState, TrackInfo

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[4]
_MPV_SOCK = "/tmp/mpv-neuro.sock"

try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil nao instalado — stats de sistema limitados")


def _pactl_get_volume() -> int:
    try:
        out = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        m = re.search(r"(\d+)%", out)
        return int(m.group(1)) if m else 80
    except Exception:
        return 80


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return chr(9608) * filled + chr(9617) * (width - filled)


class NowPlayingWidget(Widget):
    DEFAULT_CSS = f"""
    NowPlayingWidget {{
        height: 11;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["neon_cyan"]};
        border-title-color: {COLORS["neon_cyan"]};
        padding: 1 2;
    }}

    NowPlayingWidget .np-info {{
        color: {COLORS["text_primary"]};
        height: 1;
    }}

    NowPlayingWidget .np-meta {{
        color: {COLORS["text_muted"]};
        height: 1;
    }}

    NowPlayingWidget .np-progress {{
        color: {COLORS["neon_cyan"]};
        height: 1;
    }}

    NowPlayingWidget #np-controls {{
        height: 3;
        align: left middle;
        margin-top: 1;
    }}

    NowPlayingWidget #np-controls Button {{
        min-width: 6;
        height: 3;
        margin: 0 1 0 0;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["border_normal"]};
        color: {COLORS["text_primary"]};
    }}

    NowPlayingWidget #np-controls Button:hover {{
        background: {COLORS["accent_primary"]};
        color: {COLORS["bg_darkest"]};
        border: solid {COLORS["accent_primary"]};
    }}

    NowPlayingWidget #np-vol-label {{
        color: {COLORS["text_muted"]};
        width: auto;
        min-width: 18;
        content-align: left middle;
        margin-left: 2;
    }}
    """

    def __init__(self, mpris: MPRISBridge, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mpris = mpris
        self._track = TrackInfo()
        self._is_playing: bool = False
        self._reconnect_ticks: int = 0

    def compose(self) -> ComposeResult:
        self.border_title = "TOCANDO AGORA"
        yield Static(
            "Nenhum player ativo — abra Spotify ou outro player MPRIS",
            id="np-title",
            classes="np-info",
        )
        yield Static("", id="np-artist", classes="np-meta")
        yield Static("", id="np-progress", classes="np-progress")
        with Horizontal(id="np-controls"):
            yield Button("|<", id="np-prev", tooltip="Anterior")
            yield Button(" >", id="np-playpause", tooltip="Play/Pause")
            yield Button(">|", id="np-next", tooltip="Proxima")
            yield Static("", id="np-vol-label")

    def on_mount(self) -> None:
        self._refresh_vol_label()
        self.set_interval(1.0, self._poll_mpris)

    def _refresh_vol_label(self) -> None:
        vol = _pactl_get_volume()
        bar = _bar(vol / 100, 8)
        try:
            self.query_one("#np-vol-label", Static).update(
                f"[{COLORS['text_muted']}]VOL[/] [{COLORS['accent_primary']}]{bar}[/] [{COLORS['text_muted']}]{vol}%[/]"
            )
        except Exception:
            pass

    async def _poll_mpris(self) -> None:
        if not self._mpris:
            return

        # Tenta reconectar se sem proxy (a cada 5 ciclos = 5s)
        if not self._mpris._player_proxy:
            self._reconnect_ticks += 1
            if self._reconnect_ticks >= 5:
                self._reconnect_ticks = 0
                await self._mpris.connect()
            return

        try:
            track = await self._mpris.get_current_track()
            self._track = track

            status = await self._mpris.get_playback_status()
            self._is_playing = status == PlaybackState.PLAYING
            try:
                btn = self.query_one("#np-playpause", Button)
                btn.label = "||" if self._is_playing else " >"
            except Exception:
                pass

            if track.title:
                player_tag = (
                    f"[{COLORS['accent_primary']}][{track.player_name.upper()}][/]  "
                    if track.player_name
                    else ""
                )
                self.query_one("#np-title", Static).update(
                    f"{player_tag}[bold {COLORS['text_primary']}]{track.title}[/]"
                )
                self.query_one("#np-artist", Static).update(
                    f"[{COLORS['text_muted']}]{track.artist}[/]"
                )
            else:
                self.query_one("#np-title", Static).update(
                    f"[{COLORS['text_muted']}]Nenhum player ativo — abra Spotify ou outro player MPRIS[/]"
                )
                self.query_one("#np-artist", Static).update("")

            if track.duration_us > 0:
                progress = track.position_us / track.duration_us
                dur_s = track.duration_us // 1_000_000
                pos_s = track.position_us // 1_000_000
                bar = _bar(progress, 22)
                self.query_one("#np-progress", Static).update(
                    f"[{COLORS['neon_cyan']}]{bar}[/] "
                    f"[{COLORS['text_muted']}]{pos_s // 60}:{pos_s % 60:02d} / {dur_s // 60}:{dur_s % 60:02d}[/]"
                )
            else:
                self.query_one("#np-progress", Static).update("")

        except Exception as e:
            logger.debug("NowPlaying poll error: %s", e)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "np-playpause":
            self.run_worker(self._mpris.play_pause, exclusive=False)
        elif btn == "np-next":
            self.run_worker(self._mpris.next_track, exclusive=False)
        elif btn == "np-prev":
            self.run_worker(self._mpris.prev_track, exclusive=False)


class YouTubePlayerWidget(Widget):
    DEFAULT_CSS = f"""
    YouTubePlayerWidget {{
        height: 11;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["accent_secondary"]};
        border-title-color: {COLORS["accent_secondary"]};
        padding: 1 2;
    }}

    YouTubePlayerWidget #yt-row {{
        height: 3;
        align: left middle;
    }}

    YouTubePlayerWidget #yt-url {{
        width: 1fr;
        height: 3;
        background: {COLORS["bg_dark"]};
        border: solid {COLORS["border_normal"]};
        color: {COLORS["text_primary"]};
        margin-right: 1;
    }}

    YouTubePlayerWidget #yt-url:focus {{
        border: solid {COLORS["accent_primary"]};
    }}

    YouTubePlayerWidget #yt-play {{
        width: 10;
        height: 3;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["accent_secondary"]};
        color: {COLORS["accent_secondary"]};
    }}

    YouTubePlayerWidget #yt-play:hover {{
        background: {COLORS["accent_secondary"]};
        color: {COLORS["bg_darkest"]};
    }}

    YouTubePlayerWidget #yt-stop {{
        width: 8;
        height: 3;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["neon_pink"]};
        color: {COLORS["neon_pink"]};
        margin-left: 1;
    }}

    YouTubePlayerWidget #yt-stop:hover {{
        background: {COLORS["neon_pink"]};
        color: {COLORS["bg_darkest"]};
    }}

    YouTubePlayerWidget #yt-log {{
        height: 4;
        background: {COLORS["bg_dark"]};
        border: none;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mpv_proc: Optional[asyncio.subprocess.Process] = None

    def compose(self) -> ComposeResult:
        self.border_title = "YOUTUBE / MPV"
        with Horizontal(id="yt-row"):
            yield Input(placeholder="URL do YouTube ou caminho local...", id="yt-url")
            yield Button("TOCAR", id="yt-play")
            yield Button("STOP", id="yt-stop")
        yield RichLog(id="yt-log", markup=True, highlight=False, wrap=True)

    def on_mount(self) -> None:
        self._log(f"[{COLORS['text_muted']}]Cole uma URL e pressione TOCAR[/]")

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#yt-log", RichLog).write(msg)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yt-play":
            url = self.query_one("#yt-url", Input).value.strip()
            if url:
                self.run_worker(self._play(url), exclusive=True, name="mpv-play")
            else:
                self._log(f"[{COLORS['warning']}]Informe uma URL primeiro[/]")
        elif event.button.id == "yt-stop":
            self.run_worker(self._stop(), exclusive=False, name="mpv-stop")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "yt-url":
            url = event.value.strip()
            if url:
                self.run_worker(self._play(url), exclusive=True, name="mpv-play")

    async def _play(self, url: str) -> None:
        await self._stop()
        label = url[:55] + "..." if len(url) > 55 else url
        self._log(f"[{COLORS['accent_primary']}]>> {label}[/]")
        try:
            self._mpv_proc = await asyncio.create_subprocess_exec(
                "mpv",
                "--no-video",
                "--really-quiet",
                f"--input-ipc-server={_MPV_SOCK}",
                url,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._log(f"[{COLORS['success']}]mpv PID {self._mpv_proc.pid} — tocando[/]")
        except FileNotFoundError:
            self._log(
                f"[{COLORS['error']}]mpv nao encontrado — instale: sudo apt install mpv[/]"
            )
        except Exception as e:
            self._log(f"[{COLORS['error']}]Erro: {e}[/]")

    async def _stop(self) -> None:
        if self._mpv_proc and self._mpv_proc.returncode is None:
            try:
                self._mpv_proc.terminate()
                await asyncio.wait_for(self._mpv_proc.wait(), timeout=3.0)
                self._log(f"[{COLORS['text_muted']}]mpv encerrado[/]")
            except Exception:
                pass
        self._mpv_proc = None


class SystemInfoWidget(Widget):
    DEFAULT_CSS = f"""
    SystemInfoWidget {{
        height: 5;
        background: {COLORS["bg_elevated"]};
        border: solid {COLORS["success"]};
        border-title-color: {COLORS["success"]};
        padding: 0 2;
    }}

    SystemInfoWidget #sys-row {{
        height: 3;
        align: left middle;
    }}

    SystemInfoWidget .sys-stat {{
        width: 1fr;
        height: 1;
        color: {COLORS["text_primary"]};
    }}
    """

    def compose(self) -> ComposeResult:
        self.border_title = "SISTEMA"
        with Horizontal(id="sys-row"):
            yield Static("CPU  --%", id="sys-cpu", classes="sys-stat")
            yield Static("RAM  --/--GB", id="sys-ram", classes="sys-stat")
            yield Static("Datasets: --", id="sys-datasets", classes="sys-stat")
            yield Static("Modelos: --", id="sys-models", classes="sys-stat")

    def on_mount(self) -> None:
        self.set_interval(3.0, self._refresh)
        self.run_worker(self._refresh, exclusive=False)

    async def _refresh(self) -> None:
        try:
            if _PSUTIL_AVAILABLE:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                ram_used = mem.used / 1024**3
                ram_total = mem.total / 1024**3
                self.query_one("#sys-cpu", Static).update(
                    f"[{COLORS['text_muted']}]CPU[/] [{COLORS['accent_primary']}]{_bar(cpu / 100, 6)}[/] "
                    f"[{COLORS['text_muted']}]{cpu:.0f}%[/]"
                )
                self.query_one("#sys-ram", Static).update(
                    f"[{COLORS['text_muted']}]RAM[/] [{COLORS['accent_primary']}]{_bar(mem.percent / 100, 6)}[/] "
                    f"[{COLORS['text_muted']}]{ram_used:.1f}/{ram_total:.1f}GB[/]"
                )
            else:
                self.query_one("#sys-cpu", Static).update(
                    f"[{COLORS['text_muted']}]CPU (psutil ausente)[/]"
                )
                self.query_one("#sys-ram", Static).update(
                    f"[{COLORS['text_muted']}]RAM (psutil ausente)[/]"
                )

            datasets_dir = _PROJECT_ROOT / "outputs" / "datasets"
            models_dir = _PROJECT_ROOT / "outputs" / "models"
            n_datasets = (
                len(list(datasets_dir.glob("*"))) if datasets_dir.exists() else 0
            )
            n_models = len(list(models_dir.glob("*"))) if models_dir.exists() else 0

            self.query_one("#sys-datasets", Static).update(
                f"[{COLORS['text_muted']}]Datasets:[/] [{COLORS['neon_cyan']}]{n_datasets}[/]"
            )
            self.query_one("#sys-models", Static).update(
                f"[{COLORS['text_muted']}]Modelos:[/] [{COLORS['neon_cyan']}]{n_models}[/]"
            )
        except Exception as e:
            logger.debug("SystemInfo refresh error: %s", e)


class AsciiControlPanel(Widget):
    DEFAULT_CSS = """
    AsciiControlPanel {
        height: 100%;
        width: 100%;
    }

    AsciiControlPanel #media-layout {
        height: 100%;
        padding: 1;
        scrollbar-gutter: stable;
    }

    AsciiControlPanel NowPlayingWidget {
        margin-bottom: 1;
    }

    AsciiControlPanel YouTubePlayerWidget {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_log", "Limpar log", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mpris = MPRISBridge()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="media-layout"):
            yield NowPlayingWidget(mpris=self._mpris, id="now-playing")
            yield YouTubePlayerWidget(id="youtube-player")
            yield SystemInfoWidget(id="system-info")

    def on_mount(self) -> None:
        self.run_worker(
            self._connect_mpris, exclusive=False, name="media-mpris-connect"
        )

    async def _connect_mpris(self) -> None:
        connected = await self._mpris.connect()
        if connected:
            logger.info("MediaPanel: MPRIS conectado a %s", self._mpris._player_name)
        else:
            logger.info("MediaPanel: MPRIS — nenhum player ativo no momento")

    def action_clear_log(self) -> None:
        try:
            self.query_one("#yt-log", RichLog).clear()
        except Exception:
            pass


# "A simplicidade é a sofisticação máxima." — Leonardo da Vinci
