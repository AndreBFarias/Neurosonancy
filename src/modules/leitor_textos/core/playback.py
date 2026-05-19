"""Playback simples via `ffplay` subprocess com terminate() limpo."""
from __future__ import annotations

import atexit
import shutil
import subprocess
import threading
import time
from pathlib import Path


class PlaybackError(RuntimeError):
    """Erro de playback (binário ausente, arquivo inválido)."""


class Player:
    """Encapsula `ffplay -nodisp -autoexit` em um Popen com stop limpo.

    Uma única instância gerencia uma reprodução por vez. Chamadas a `play()`
    durante reprodução ativa fazem stop implícito antes de tocar o novo path.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._binary = shutil.which("ffplay")
        atexit.register(self._cleanup)

    @property
    def available(self) -> bool:
        return self._binary is not None

    def play(self, path: Path) -> None:
        if not self.available:
            raise PlaybackError("ffplay não encontrado no PATH")
        path = Path(path).resolve()
        if not path.is_file():
            raise PlaybackError(f"arquivo inexistente: {path}")

        with self._lock:
            self._stop_locked()
            self._proc = subprocess.Popen(
                [
                    self._binary,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    str(path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        self._proc = None

    def is_playing(self) -> bool:
        proc = self._proc
        if proc is None:
            return False
        return proc.poll() is None

    def _cleanup(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
