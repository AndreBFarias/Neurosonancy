# -*- coding: utf-8 -*-

import atexit
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRAY_AVAILABLE = False
AppIndicator = None
GLib = None
Gtk = None

try:
    import gi  # type: ignore
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator  # type: ignore
    from gi.repository import GLib, Gtk  # type: ignore
    TRAY_AVAILABLE = True
    logger.info("Tray: usando AyatanaAppIndicator3")
except (ImportError, ValueError):
    try:
        import gi  # type: ignore
        gi.require_version("AppIndicator3", "0.1")
        from gi.repository import AppIndicator3 as AppIndicator  # type: ignore
        from gi.repository import GLib, Gtk  # type: ignore
        TRAY_AVAILABLE = True
        logger.info("Tray: usando AppIndicator3")
    except (ImportError, ValueError):
        logger.warning(
            "AppIndicator3/AyatanaAppIndicator3 nao disponivel — "
            "system tray desativado. "
            "Instale: sudo apt install gir1.2-ayatanaappindicator3-0.1"
        )

_PROJECT_ROOT = Path(__file__).parents[2]
_ICON_PATH = str(_PROJECT_ROOT / "assets" / "icon.png")


class TrayCompanion:
    def __init__(self, app: Any, mpris: Any) -> None:
        self._app = app
        self._mpris = mpris
        self._indicator: Optional[object] = None
        self._loop: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_playing: bool = False
        self._track_title: str = ""
        self._model_name: str = ""
        self._running: bool = False

    def start(self) -> bool:
        if not TRAY_AVAILABLE:
            return False
        if self._running:
            return True
        self._thread = threading.Thread(target=self._run, daemon=True, name="tray-companion")
        self._thread.start()
        self._running = True
        atexit.register(self.stop)
        return True

    def stop(self) -> None:
        if self._loop:
            try:
                GLib.idle_add(self._loop.quit)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False

    def _build_menu(self) -> "Gtk.Menu":
        with self._lock:
            play_label = "Pausar" if self._is_playing else "Play"
            track = self._track_title[:30] or "Sem faixa"
            model = self._model_name[:25] or "Sem modelo"

        menu = Gtk.Menu()

        item_track = Gtk.MenuItem(label=f"Faixa: {track}")
        item_track.set_sensitive(False)
        menu.append(item_track)

        item_model = Gtk.MenuItem(label=f"Modelo: {model}")
        item_model.set_sensitive(False)
        menu.append(item_model)

        menu.append(Gtk.SeparatorMenuItem())

        item_play = Gtk.MenuItem(label=play_label)
        item_play.connect("activate", self._on_playpause)
        menu.append(item_play)

        item_next = Gtk.MenuItem(label="Proxima")
        item_next.connect("activate", self._on_next)
        menu.append(item_next)

        item_prev = Gtk.MenuItem(label="Anterior")
        item_prev.connect("activate", self._on_prev)
        menu.append(item_prev)

        menu.append(Gtk.SeparatorMenuItem())

        item_show = Gtk.MenuItem(label="Mostrar TUI")
        item_show.connect("activate", self._on_show_tui)
        menu.append(item_show)

        item_quit = Gtk.MenuItem(label="Sair")
        item_quit.connect("activate", self._on_quit)
        menu.append(item_quit)

        menu.show_all()
        return menu

    def _run(self) -> None:
        try:
            self._indicator = AppIndicator.Indicator.new(
                "neurosonancy",
                _ICON_PATH,
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self._indicator.set_menu(self._build_menu())

            self._loop = GLib.MainLoop()
            self._loop.run()
        except Exception as e:
            logger.error("Erro no tray AppIndicator: %s", e)
            self._running = False

    def _refresh_menu(self) -> None:
        if self._indicator and GLib:
            try:
                GLib.idle_add(self._do_refresh_menu)
            except Exception:
                pass

    def _do_refresh_menu(self) -> bool:
        try:
            self._indicator.set_menu(self._build_menu())
        except Exception:
            pass
        return False

    def _on_playpause(self, *_: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_playpause)
        except Exception as e:
            logger.debug("tray playpause: %s", e)

    def _on_next(self, *_: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_next)
        except Exception as e:
            logger.debug("tray next: %s", e)

    def _on_prev(self, *_: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_prev)
        except Exception as e:
            logger.debug("tray prev: %s", e)

    def _on_show_tui(self, *_: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_show)
        except Exception as e:
            logger.debug("tray show: %s", e)

    def _on_quit(self, *_: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_quit)
        except Exception as e:
            logger.debug("tray quit: %s", e)

    def update_state(self, is_playing: bool, track_title: str) -> None:
        with self._lock:
            self._is_playing = is_playing
            self._track_title = track_title
        self._refresh_menu()

    def update_model(self, model_name: str) -> None:
        with self._lock:
            self._model_name = model_name
        self._refresh_menu()


# "Nada é bom ou mau salvo a ação virtuosa ou viciosa da vontade." — Epicteto
