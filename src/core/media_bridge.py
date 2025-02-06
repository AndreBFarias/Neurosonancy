# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from dbus_next.aio import MessageBus  # type: ignore
    from dbus_next import BusType, Variant  # type: ignore
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    logger.warning("dbus-next nao instalado — suporte MPRIS desativado")


@dataclass
class TrackInfo:
    title: str = ""
    artist: str = ""
    album: str = ""
    player_name: str = ""
    duration_us: int = 0
    position_us: int = 0


class PlaybackState(str, Enum):
    PLAYING = "Playing"
    PAUSED = "Paused"
    STOPPED = "Stopped"


class MPRISBridge:
    MPRIS_PREFIX = "org.mpris.MediaPlayer2."

    def __init__(self) -> None:
        self._bus: Optional[object] = None
        self._player_proxy: Optional[object] = None
        self._player_name: str = ""
        self._has_tracklist: bool = False
        self.on_track_changed: Optional[Callable[[TrackInfo], None]] = None
        self.on_state_changed: Optional[Callable[[PlaybackState], None]] = None

    async def connect(self, player_bus_name: Optional[str] = None) -> bool:
        if not DBUS_AVAILABLE:
            return False

        try:
            self._bus = await MessageBus(bus_type=BusType.SESSION).connect()
            reply = await self._bus.call_method(
                "org.freedesktop.DBus",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus",
                "ListNames",
            )
            names: list[str] = reply.body[0]
            players = [n for n in names if n.startswith(self.MPRIS_PREFIX)]

            if not players:
                logger.info("Nenhum player MPRIS encontrado")
                return False

            target = player_bus_name if player_bus_name in players else players[0]
            self._player_name = target.replace(self.MPRIS_PREFIX, "")

            introspection = await self._bus.introspect(target, "/org/mpris/MediaPlayer2")
            obj = self._bus.get_proxy_object(target, "/org/mpris/MediaPlayer2", introspection)
            self._player_proxy = obj.get_interface("org.mpris.MediaPlayer2.Player")

            await self._try_tracklist(obj)
            logger.info("MPRIS conectado: %s", self._player_name)
            return True

        except Exception as e:
            logger.warning("Falha ao conectar MPRIS: %s", e)
            return False

    async def _try_tracklist(self, obj: object) -> None:
        try:
            iface = obj.get_interface("org.mpris.MediaPlayer2.TrackList")
            _ = await iface.call_get_tracks_metadata([])
            self._has_tracklist = True
        except Exception:
            self._has_tracklist = False

    async def get_current_track(self) -> TrackInfo:
        if not self._player_proxy:
            return TrackInfo()
        try:
            metadata = await self._player_proxy.get_metadata()
            title = _extract_variant(metadata, "xesam:title", "")
            artist_raw = _extract_variant(metadata, "xesam:artist", [])
            artist = ", ".join(artist_raw) if isinstance(artist_raw, list) else str(artist_raw)
            album = _extract_variant(metadata, "xesam:album", "")
            duration = int(_extract_variant(metadata, "mpris:length", 0))
            return TrackInfo(
                title=title,
                artist=artist,
                album=album,
                player_name=self._player_name,
                duration_us=duration,
            )
        except Exception as e:
            logger.debug("Erro ao obter faixa: %s", e)
            return TrackInfo()

    async def play_pause(self) -> None:
        if not self._player_proxy:
            logger.warning("MPRIS: play_pause ignorado — sem player")
            return
        try:
            await self._player_proxy.call_play_pause()
        except Exception as e:
            logger.warning("MPRIS play_pause erro: %s", e)

    async def next_track(self) -> None:
        if not self._player_proxy:
            return
        try:
            await self._player_proxy.call_next()
        except Exception as e:
            logger.warning("MPRIS next erro: %s", e)

    async def prev_track(self) -> None:
        if not self._player_proxy:
            return
        try:
            await self._player_proxy.call_previous()
        except Exception as e:
            logger.warning("MPRIS prev erro: %s", e)

    async def set_volume(self, volume: float) -> None:
        volume = max(0.0, min(1.0, volume))
        if not self._player_proxy:
            return
        try:
            await self._player_proxy.set_volume(volume)
        except Exception as e:
            logger.warning("MPRIS volume erro: %s", e)

    async def get_queue(self) -> list[TrackInfo]:
        if not self._has_tracklist or not self._bus or not self._player_name:
            return []
        try:
            bus_name = self.MPRIS_PREFIX + self._player_name
            introspection = await self._bus.introspect(bus_name, "/org/mpris/MediaPlayer2")
            obj = self._bus.get_proxy_object(bus_name, "/org/mpris/MediaPlayer2", introspection)
            iface = obj.get_interface("org.mpris.MediaPlayer2.TrackList")
            tracks_raw = await iface.call_get_tracks_metadata([])
            result: list[TrackInfo] = []
            for m in tracks_raw[:20]:
                result.append(TrackInfo(
                    title=_extract_variant(m, "xesam:title", ""),
                    artist=", ".join(_extract_variant(m, "xesam:artist", [])),
                    album=_extract_variant(m, "xesam:album", ""),
                    player_name=self._player_name,
                ))
            return result
        except Exception as e:
            logger.debug("Erro ao obter fila: %s", e)
            return []

    async def disconnect(self) -> None:
        if self._bus:
            try:
                self._bus.disconnect()
            except Exception:
                pass
        self._bus = None
        self._player_proxy = None


def _extract_variant(metadata: dict, key: str, default):
    raw = metadata.get(key)
    if raw is None:
        return default
    if DBUS_AVAILABLE and hasattr(raw, "value"):
        return raw.value
    return raw


# "A tranquilidade é a ordem perfeita da alma que dirige a si mesma." — Marco Aurélio
