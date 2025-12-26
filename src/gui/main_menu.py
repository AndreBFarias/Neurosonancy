# -*- coding: utf-8 -*-

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Vertical, Horizontal, Grid

from src.core.theme import CSS_COMMON, COLORS


class ModuleCard(Static):
    def __init__(
        self,
        title: str,
        description: str,
        module_id: str,
        shortcut: str,
        icon: str = "",
        accent_color: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._title = title
        self._description = description
        self._module_id = module_id
        self._shortcut = shortcut
        self._icon = icon
        self._accent = accent_color or COLORS["accent_primary"]

    def compose(self) -> ComposeResult:
        with Vertical(classes="card-inner"):
            yield Static(
                f"[bold {self._accent}]{self._icon}[/]",
                classes="card-icon"
            )
            yield Static(
                f"[bold {COLORS['text_primary']}]{self._title}[/]",
                classes="card-title"
            )
            yield Static(
                f"[{COLORS['text_muted']}]{self._description}[/]",
                classes="card-desc"
            )
            yield Static(
                f"[{COLORS['text_secondary']}]Pressione [bold {self._accent}]{self._shortcut}[/] ou clique[/]",
                classes="card-hint"
            )
            yield Button(
                "INICIAR",
                id=f"btn-{self._module_id}",
                classes="card-btn"
            )


class NeurosonancyLauncher(App):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = "Unified Audio Toolkit"

    CSS = CSS_COMMON + """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
        align: center middle;
        padding: 1 2;
    }

    #header-section {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 2 0;
    }

    #logo {
        text-align: center;
        width: 100%;
    }

    #version {
        text-align: center;
        width: 100%;
        color: #64748b;
        padding-top: 1;
    }

    #cards-grid {
        width: auto;
        height: auto;
        align: center middle;
        layout: horizontal;
    }

    ModuleCard {
        width: 36;
        height: 16;
        background: #1a1d2e;
        border: solid #2d3250;
        padding: 1;
        margin: 0 1;
    }

    ModuleCard:hover {
        border: solid #8b5cf6;
        background: #232738;
    }

    ModuleCard:focus {
        border: solid #a78bfa;
    }

    .card-inner {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    .card-icon {
        width: 100%;
        text-align: center;
        padding-bottom: 1;
    }

    .card-title {
        width: 100%;
        text-align: center;
        padding-bottom: 1;
    }

    .card-desc {
        width: 100%;
        text-align: center;
        height: 3;
        padding: 0 1;
    }

    .card-hint {
        width: 100%;
        text-align: center;
        padding: 1 0;
    }

    .card-btn {
        width: 100%;
        height: 3;
        background: #8b5cf6;
        color: #0d0f18;
        border: none;
        text-style: bold;
    }

    .card-btn:hover {
        background: #a78bfa;
    }

    #nav-hint {
        width: 100%;
        text-align: center;
        color: #64748b;
        padding: 2 0 1 0;
    }

    #status-bar {
        width: 100%;
        text-align: center;
        color: #3d4370;
        padding: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Sair", show=True),
        Binding("1", "launch_ascii", "Monitor", show=True),
        Binding("2", "launch_voice", "Trainer", show=True),
        Binding("3", "launch_clone", "Clone", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="main-container"):
            with Vertical(id="header-section"):
                yield Static(
                    f"[bold {COLORS['accent_primary']}]N E U R O S O N A N C Y[/]",
                    id="logo"
                )
                yield Static(
                    "Unified Audio Toolkit v2.0",
                    id="version"
                )

            with Horizontal(id="cards-grid"):
                yield ModuleCard(
                    title="MONITOR",
                    description="Sistema Luna com metricas e visualizacao em tempo real",
                    module_id="ascii",
                    shortcut="1",
                    icon="[#22d3ee][[1]][/]",
                    accent_color=COLORS["neon_cyan"],
                    id="card-ascii"
                )
                yield ModuleCard(
                    title="TRAINER",
                    description="Gravador de amostras com transcricao Whisper",
                    module_id="voice",
                    shortcut="2",
                    icon="[#4ade80][[2]][/]",
                    accent_color=COLORS["neon_green"],
                    id="card-voice"
                )
                yield ModuleCard(
                    title="CLONE",
                    description="Gerador de datasets via ElevenLabs TTS",
                    module_id="clone",
                    shortcut="3",
                    icon="[#ec4899][[3]][/]",
                    accent_color=COLORS["neon_pink"],
                    id="card-clone"
                )

            yield Static(
                f"[{COLORS['text_muted']}]Use [bold]1[/], [bold]2[/] ou [bold]3[/] para selecionar | [bold]Q[/] para sair[/]",
                id="nav-hint"
            )

            yield Static(
                f"[{COLORS['border_dim']}]Sistema pronto[/]",
                id="status-bar"
            )

        yield Footer()

    def on_mount(self) -> None:
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "btn-ascii":
            self.action_launch_ascii()
        elif button_id == "btn-voice":
            self.action_launch_voice()
        elif button_id == "btn-clone":
            self.action_launch_clone()

    def action_launch_ascii(self) -> None:
        self.exit(result="ascii")

    def action_launch_voice(self) -> None:
        self.exit(result="voice")

    def action_launch_clone(self) -> None:
        self.exit(result="clone")


def main():
    app = NeurosonancyLauncher()
    result = app.run()
    return result


if __name__ == "__main__":
    main()
