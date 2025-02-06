# -*- coding: utf-8 -*-

import atexit
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import pystray  # type: ignore
    from PIL import Image, ImageDraw  # type: ignore
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    logger.warning("pystray/Pillow nao instalado — system tray desativado")


class TrayCompanion:
    def __init__(self, app: Any, mpris: Any) -> None:
        self._app = app
        self._mpris = mpris
        self._icon: Optional[object] = None
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
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False

    def _build_icon_image(self) -> "Image.Image":
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, size - 4, size - 4], fill=(139, 92, 246, 255))
        draw.ellipse([16, 16, size - 16, size - 16], fill=(20, 22, 32, 255))
        return img

    def _build_menu(self) -> "pystray.Menu":
        with self._lock:
            play_label = "Pausar" if self._is_playing else "Play"
            track = self._track_title[:30] or "Sem faixa"
            model = self._model_name[:25] or "Sem modelo"

        return pystray.Menu(
            pystray.MenuItem(f"Faixa: {track}", None, enabled=False),
            pystray.MenuItem(f"Modelo: {model}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(play_label, self._on_playpause),
            pystray.MenuItem("Proxima", self._on_next),
            pystray.MenuItem("Anterior", self._on_prev),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Mostrar TUI", self._on_show_tui),
            pystray.MenuItem("Sair", self._on_quit),
        )

    def _run(self) -> None:
        try:
            self._icon = pystray.Icon(
                name="neurosonancy",
                icon=self._build_icon_image(),
                title="Neurosonancy",
                menu=self._build_menu(),
            )
            self._icon.run()
        except Exception as e:
            logger.error("Erro no tray: %s", e)
            self._running = False

    def _refresh_menu(self) -> None:
        if self._icon:
            try:
                self._icon.menu = self._build_menu()
                self._icon.update_menu()
            except Exception:
                pass

    def _on_playpause(self, icon: Any, item: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_playpause)
        except Exception as e:
            logger.debug("tray playpause: %s", e)

    def _on_next(self, icon: Any, item: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_next)
        except Exception as e:
            logger.debug("tray next: %s", e)

    def _on_prev(self, icon: Any, item: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_prev)
        except Exception as e:
            logger.debug("tray prev: %s", e)

    def _on_show_tui(self, icon: Any, item: Any) -> None:
        try:
            self._app.call_from_thread(self._app._tray_show)
        except Exception as e:
            logger.debug("tray show: %s", e)

    def _on_quit(self, icon: Any, item: Any) -> None:
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
