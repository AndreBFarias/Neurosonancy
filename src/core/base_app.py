# -*- coding: utf-8 -*-

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical

from .theme import CSS_COMMON, COLORS


class NeurosonancyBaseApp(App):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = ""
    CSS = CSS_COMMON

    BINDINGS = [
        Binding("escape", "back_to_menu", "Menu", show=True, priority=True),
        Binding("q", "quit", "Sair", show=True),
    ]

    def __init__(self, return_to_menu: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._return_to_menu = return_to_menu

    def action_back_to_menu(self) -> None:
        if self._return_to_menu:
            self.exit(result="menu")
        else:
            self.exit()

    def action_quit(self) -> None:
        self.exit(result="quit")

    def notify_success(self, message: str) -> None:
        self.notify(message, severity="information")

    def notify_warning(self, message: str) -> None:
        self.notify(message, severity="warning")

    def notify_error(self, message: str) -> None:
        self.notify(message, severity="error")

    @property
    def colors(self):
        return COLORS
