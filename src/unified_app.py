# -*- coding: utf-8 -*-

import logging
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import ContentSwitcher, Footer

from src.core.theme import CSS_COMMON, COLORS
from src.core.widgets import NavSidebar, MediaHeader
from src.core.media_bridge import MPRISBridge, TrackInfo
from src.core.tray_companion import TrayCompanion
from src.modules.ascii_control.ui.panel import AsciiControlPanel
from src.modules.voice_trainer.ui.panel import VoiceTrainerPanel
from src.modules.clone_voice.ui.panel import CloneVoicePanel

logger = logging.getLogger(__name__)


class NeurosonancyUnifiedApp(App):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = "Unified"

    BINDINGS = [
        Binding("1", "switch_module('monitor')", "Monitor", show=True),
        Binding("2", "switch_module('trainer')", "Trainer", show=True),
        Binding("3", "switch_module('clone')", "Clone", show=True),
        Binding("q", "quit", "Sair", show=True),
        Binding("ctrl+t", "toggle_tray", "Tray", show=True),
    ]

    CSS = CSS_COMMON + f"""
    #body-layout {{
        height: 1fr;
        layout: horizontal;
    }}

    ContentSwitcher {{
        width: 1fr;
        min-width: 60;
        height: 100%;
    }}

    #panel-monitor,
    #panel-trainer,
    #panel-clone {{
        width: 100%;
        height: 100%;
    }}
    """

    def __init__(self) -> None:
        super().__init__()
        self._mpris = MPRISBridge()
        self._tray: TrayCompanion | None = None
        self._tray_running: bool = False

    def compose(self) -> ComposeResult:
        yield MediaHeader(id="media-header")
        with Horizontal(id="body-layout"):
            yield NavSidebar(id="nav-sidebar")
            with ContentSwitcher(initial="panel-monitor"):
                yield AsciiControlPanel(id="panel-monitor")
                yield VoiceTrainerPanel(id="panel-trainer")
                yield CloneVoicePanel(id="panel-clone")
        yield Footer()

    def on_mount(self) -> None:
        self._tray = TrayCompanion(self, self._mpris)
        tray_started = self._tray.start()
        if tray_started:
            self._tray_running = True
            sidebar = self.query_one("#nav-sidebar", NavSidebar)
            sidebar.set_tray_status(True)
            logger.info("System tray iniciado")

        self.run_worker(self._connect_mpris, exclusive=False, name="mpris-connect")
        self.set_interval(2.0, self._refresh_media_info)

    async def _connect_mpris(self) -> None:
        try:
            connected = await self._mpris.connect()
            if connected:
                logger.info("MPRIS conectado")
            else:
                logger.info("MPRIS nao disponivel")
        except Exception as e:
            logger.warning("Falha ao conectar MPRIS: %s", e)

    async def _refresh_media_info(self) -> None:
        try:
            track = await self._mpris.get_current_track()
            header = self.query_one("#media-header", MediaHeader)
            header.update_track(track)

            if self._tray:
                self._tray.update_state(header.is_playing, track.title)
        except Exception:
            pass

    def _switch_panel(self, module_id: str) -> None:
        self.query_one(ContentSwitcher).current = f"panel-{module_id}"
        self.query_one("#nav-sidebar", NavSidebar).set_active(module_id)
        self.query_one("#media-header", MediaHeader).active_module = module_id

    def on_nav_sidebar_module_selected(self, event: NavSidebar.ModuleSelected) -> None:
        self._switch_panel(event.module_id)

    def action_switch_module(self, module_id: str) -> None:
        self._switch_panel(module_id)

    def action_toggle_tray(self) -> None:
        if not self._tray:
            return
        if self._tray_running:
            self._tray.stop()
            self._tray_running = False
            self.query_one("#nav-sidebar", NavSidebar).set_tray_status(False)
            self.notify("Tray desativado", severity="information")
        else:
            started = self._tray.start()
            if started:
                self._tray_running = True
                self.query_one("#nav-sidebar", NavSidebar).set_tray_status(True)
                self.notify("Tray ativado", severity="information")

    def on_media_header_play_pause_requested(self, event: MediaHeader.PlayPauseRequested) -> None:
        self.run_worker(self._mpris.play_pause, exclusive=False)

    def on_media_header_next_track_requested(self, event: MediaHeader.NextTrackRequested) -> None:
        self.run_worker(self._mpris.next_track, exclusive=False)

    def on_media_header_prev_track_requested(self, event: MediaHeader.PrevTrackRequested) -> None:
        self.run_worker(self._mpris.prev_track, exclusive=False)

    def on_media_header_volume_changed(self, event: MediaHeader.VolumeChanged) -> None:
        self.run_worker(
            lambda: self._mpris.set_volume(event.volume),
            exclusive=False,
        )

    def on_media_header_model_change_requested(self, event: MediaHeader.ModelChangeRequested) -> None:
        self.run_worker(
            lambda: self._switch_model(event.model_display_name),
            exclusive=True,
            name="model-switch",
        )

    async def _switch_model(self, display_name: str) -> None:
        from src.core.model_manager import ModelManager
        manager = ModelManager()
        spec = manager.get_spec_by_display_name(display_name)
        if spec is None:
            self.notify(f"Modelo nao encontrado: {display_name}", severity="error")
            return

        self.notify(f"Carregando {display_name}...", severity="information")
        success = await manager.switch(spec.category, spec.engine, spec.variant)
        if success:
            self.notify(f"Modelo ativo: {display_name}", severity="information")
            if self._tray:
                self._tray.update_model(display_name)
        else:
            self.notify(f"Falha ao carregar {display_name}", severity="error")

    async def _tray_playpause(self) -> None:
        await self._mpris.play_pause()

    async def _tray_next(self) -> None:
        await self._mpris.next_track()

    async def _tray_prev(self) -> None:
        await self._mpris.prev_track()

    async def _tray_show(self) -> None:
        pass

    async def _tray_quit(self) -> None:
        self.exit()

    def on_unmount(self) -> None:
        if self._tray and self._tray_running:
            self._tray.stop()
        self.run_worker(self._mpris.disconnect, exclusive=False)


# "A virtude está em agir bem, não em apenas intencioná-lo." — Epicteto
