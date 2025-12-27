# -*- coding: utf-8 -*-

import os
import json
import logging
import threading
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import (
    Header, Footer, Static, Button, Label,
    Input, ProgressBar, RichLog, RadioSet, RadioButton, Select, Checkbox, TextArea
)
from textual.containers import Vertical, Horizontal, VerticalScroll, Center
from textual.reactive import reactive

from src.core.base_app import NeurosonancyBaseApp
from src.core.theme import CSS_COMMON, COLORS

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
LUNA_PHRASES_FILE = ROOT_DIR / "data_input" / "phrases_luna_example.md"
ERIS_PHRASES_FILE = ROOT_DIR / "data_input" / "phrases_eris.md"
DEFAULT_PHRASES_FILE = ROOT_DIR / "data_input" / "phrases_template.md"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data_output" / "clone_voice"
CONFIG_FILE = ROOT_DIR / "data_input" / "clone_voice_config.json"


def load_saved_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ao carregar config: {e}")
    return {}


def save_config(config: Dict[str, Any]) -> None:
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erro ao salvar config: {e}")


class StatusPanel(Static):
    status = reactive("idle")

    STATUS_MAP = {
        "idle": ("[dim]AGUARDANDO[/]", ""),
        "initializing": (f"[bold {COLORS['info']}]INICIALIZANDO[/]", ""),
        "generating": (f"[bold {COLORS['warning']}]GERANDO[/]", ""),
        "completed": (f"[bold {COLORS['success']}]CONCLUIDO[/]", ""),
        "error": (f"[bold {COLORS['error']}]ERRO[/]", ""),
        "stopped": (f"[bold {COLORS['warning']}]PARADO[/]", ""),
    }

    def render(self) -> str:
        text, _ = self.STATUS_MAP.get(self.status, ("[dim]--[/]", ""))
        return text


class TrainingCompleteModal(ModalScreen):

    CSS = """
    TrainingCompleteModal {
        align: center middle;
    }

    #modal-container {
        width: 50;
        height: 14;
        background: #1a1d2e;
        border: solid #22c55e;
        padding: 1 2;
    }

    #modal-title {
        text-align: center;
        text-style: bold;
        color: #22c55e;
        margin-bottom: 1;
    }

    #modal-message {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1;
    }

    #modal-path {
        text-align: center;
        color: #22d3ee;
        margin-bottom: 1;
    }

    .modal-buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    .modal-buttons Button {
        width: 1fr;
        margin: 0 1;
    }

    #btn-modal-ok {
        background: #6366f1;
        color: #0d0f18;
    }

    #btn-modal-open {
        background: #22c55e;
        color: #0d0f18;
    }
    """

    def __init__(self, output_path: Path, model_type: str):
        super().__init__()
        self._output_path = output_path
        self._model_type = model_type

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Static("TREINAMENTO CONCLUIDO", id="modal-title")
            yield Static(f"Modelo: {self._model_type.upper()}", id="modal-message")
            yield Static(f"{self._output_path.name}", id="modal-path")
            with Horizontal(classes="modal-buttons"):
                yield Button("OK", id="btn-modal-ok")
                yield Button("ABRIR PASTA", id="btn-modal-open")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-modal-open":
            subprocess.Popen(["xdg-open", str(self._output_path)])
        self.dismiss()


class CloneVoiceApp(NeurosonancyBaseApp):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = "Clone Voice"

    SAVE_DEBOUNCE_MS = 1500
    _save_timer = None
    _config_dirty = False

    CSS = CSS_COMMON + """
    #main-container {
        layout: horizontal;
        height: 100%;
        padding: 0 1;
    }

    #left-column {
        width: 30%;
        height: 100%;
        padding: 0;
    }

    #right-column {
        width: 70%;
        height: 100%;
        padding: 0 0 0 1;
    }

    #config-panel {
        height: 100%;
        background: #1a1d2e;
        border: solid #ec4899;
        padding: 1 2;
        scrollbar-gutter: stable;
    }

    #config-panel:focus-within {
        scrollbar-color: #ec4899;
    }

    #right-top {
        layout: horizontal;
        height: 55%;
    }

    #generator-panel {
        width: 1fr;
        height: 100%;
        background: #1a1d2e;
        border: solid #8b5cf6;
        padding: 1 2;
    }

    #training-panel {
        width: 1fr;
        height: 100%;
        background: #1a1d2e;
        border: solid #22d3ee;
        padding: 1 2;
        margin-left: 1;
        overflow-y: auto;
    }

    #test-panel {
        layout: horizontal;
        height: auto;
        min-height: 14;
        background: #1a1d2e;
        border: solid #f59e0b;
        padding: 2 3;
        margin-top: 1;
    }

    #test-left {
        width: 1fr;
        height: auto;
        padding: 0 3 0 0;
        border-right: solid #2d3250;
    }

    #test-right {
        width: 2fr;
        height: auto;
        padding: 0 0 0 3;
    }

    #model-selector-test {
        width: 100%;
        margin-bottom: 1;
    }

    #input-test-text {
        width: 100%;
        height: 5;
        min-height: 5;
        margin-bottom: 1;
        background: #141620;
        border: solid #2d3250;
    }

    #btn-play {
        width: 100%;
        background: #f59e0b;
        color: #0d0f18;
        text-style: bold;
        margin-top: 1;
    }

    #btn-play:hover { background: #fbbf24; }
    #btn-play:disabled { background: #2d3250; color: #64748b; }

    #btn-refresh-models {
        width: 100%;
        background: #6366f1;
        color: #0d0f18;
        margin-top: 1;
    }

    Select {
        width: 100%;
    }

    #dataset-selector {
        width: 100%;
    }

    .panel-title {
        text-style: bold;
        text-align: center;
        padding: 0;
        margin-bottom: 1;
    }

    #config-panel .panel-title { color: #ec4899; }
    #generator-panel .panel-title { color: #8b5cf6; }
    #training-panel .panel-title { color: #22d3ee; }
    #test-panel .panel-title { color: #f59e0b; }

    .hidden-log {
        display: none;
    }

    .setting-label {
        color: #94a3b8;
        padding: 0;
        margin-top: 0;
        height: 1;
    }

    Input {
        height: 3;
        padding: 0 1;
        margin: 0;
    }

    #training-panel Input,
    #generator-panel Input {
        height: 3;
        margin-bottom: 1;
    }

    Button {
        height: 3;
        min-width: 8;
        margin: 0;
    }

    #btn-generate {
        width: 100%;
        background: #22c55e;
        color: #0d0f18;
        text-style: bold;
        margin-top: 1;
    }

    #btn-generate:hover { background: #4ade80; }
    #btn-generate:disabled { background: #2d3250; color: #64748b; }

    #btn-stop {
        width: 100%;
        background: #ef4444;
        color: #0d0f18;
        text-style: bold;
        margin-top: 1;
    }

    #btn-stop:disabled { background: #2d3250; color: #64748b; }

    #status-indicator {
        text-align: center;
        padding: 0;
        height: 1;
    }

    .counter-display {
        text-align: center;
        color: #22c55e;
        text-style: bold;
        padding: 0;
        height: 1;
    }

    .current-phrase {
        text-align: center;
        color: #94a3b8;
        padding: 0;
        height: 1;
    }

    #output-log {
        height: 100%;
        background: #141620;
        border: solid #2d3250;
    }

    RadioSet {
        background: #141620;
        border: solid #2d3250;
        height: auto;
        padding: 0;
    }

    RadioButton {
        height: 1;
        padding: 0;
    }

    #btn-browse {
        width: 100%;
        height: 3;
        margin-top: 1;
        background: #8b5cf6;
        color: #0d0f18;
    }

    #btn-browse:hover { background: #a78bfa; }

    .file-hint {
        color: #4ade80;
        text-style: italic;
        margin-top: 0;
        height: 1;
    }

    #dataset-selector {
        height: 6;
        background: #141620;
        border: solid #2d3250;
        padding: 0;
    }

    #dataset-selector > .option-list--option {
        padding: 0 1;
    }

    #dataset-selector > .option-list--option-highlighted {
        background: #8b5cf6;
        color: #0d0f18;
    }

    .train-buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }

    .train-buttons Button {
        width: 1fr;
    }

    #btn-refresh-datasets {
        width: 100%;
        background: #6366f1;
        color: #0d0f18;
        margin-top: 1;
    }

    Checkbox {
        height: 3;
        margin: 0;
        padding: 0 1;
        width: auto;
    }

    .use-top-row {
        layout: horizontal;
        height: 3;
        margin: 0;
        align: left middle;
    }

    #input-top-n {
        width: 6;
        margin-left: 1;
    }

    .model-hint {
        color: #64748b;
        text-style: italic;
        height: 1;
        padding: 0;
        margin: 0;
    }

    #btn-train-chatterbox {
        background: #f97316;
        color: #0d0f18;
        text-style: bold;
        margin-right: 1;
    }

    #btn-train-chatterbox:hover { background: #fb923c; }

    #btn-train-coqui {
        background: #10b981;
        color: #0d0f18;
        text-style: bold;
    }

    #btn-train-coqui:hover { background: #34d399; }

    .training-status {
        text-align: center;
        padding: 0;
        height: 1;
    }

    ProgressBar {
        height: 1;
        margin: 0;
    }

    .section-separator {
        height: 1;
        margin: 1 0;
        border-top: solid #2d3250;
    }

    #save-indicator {
        text-align: right;
        color: #64748b;
        height: 1;
        padding: 0;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "back_to_menu", "Menu", show=True, priority=True),
        Binding("q", "quit", "Sair", show=True),
        Binding("g", "start_generation", "Gerar", show=True),
        Binding("s", "stop_generation", "Parar", show=True),
        Binding("t", "start_training", "Treinar", show=True),
    ]

    is_generating: reactive[bool] = reactive(False)
    is_training: reactive[bool] = reactive(False)
    phrases_generated: reactive[int] = reactive(0)
    target_phrases: reactive[int] = reactive(50)
    training_progress: reactive[int] = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset_manager = None
        self._generation_thread: Optional[threading.Thread] = None
        self._training_thread: Optional[threading.Thread] = None
        self._last_output_dir: Optional[Path] = None
        self._selected_dataset: Optional[Path] = None
        self._datasets_map: Dict[str, Path] = {}
        self._trainer = None
        self._models_map: Dict[str, Path] = {}
        self._selected_model: Optional[Path] = None
        self._is_playing: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            with Vertical(id="left-column"):
                with VerticalScroll(id="config-panel"):
                    yield Static("CONFIGURACAO", classes="panel-title")
                    yield Static("", id="save-indicator")

                    yield Label("API Key:", classes="setting-label")
                    yield Input(placeholder="sk_xxxxxxxx...", id="input-api-key")

                    yield Label("Voice ID:", classes="setting-label")
                    yield Input(placeholder="ID da voz no ElevenLabs", id="input-voice-id")

                    yield Label("Nome do Dataset:", classes="setting-label")
                    yield Input(placeholder="Ex: MinhaVoz", id="input-dataset-name")

                    yield Label("Qtd Frases:", classes="setting-label")
                    yield Input(value="50", id="input-phrase-count", type="integer")

                    yield Label("Modelo TTS:", classes="setting-label")
                    with RadioSet(id="model-selector"):
                        yield RadioButton("Multilingual v2", id="model-multi-v2", value=True)
                        yield RadioButton("Turbo v2.5", id="model-turbo")
                        yield RadioButton("Flash v2", id="model-flash")

                    yield Static("", classes="section-separator")

                    yield Label("Arquivo de Frases:", classes="setting-label")
                    yield Input(placeholder="Caminho do arquivo .md", id="input-phrases-file")
                    yield Button("PROCURAR", id="btn-browse")
                    yield Label("", id="file-status", classes="file-hint")

                    yield Label("Diretorio Saida:", classes="setting-label")
                    yield Input(value=str(DEFAULT_OUTPUT_DIR), id="input-output-dir")

            with Vertical(id="right-column"):
                with Horizontal(id="right-top"):
                    with Vertical(id="generator-panel"):
                        yield Static("GERADOR", classes="panel-title")
                        yield StatusPanel(id="status-indicator")
                        yield Static("0 / 50 frases", id="counter-display", classes="counter-display")
                        yield ProgressBar(id="generation-progress", total=100, show_eta=False)
                        yield Static("", id="current-phrase", classes="current-phrase")
                        yield Button("INICIAR", id="btn-generate")
                        yield Button("PARAR", id="btn-stop", disabled=True)

                    with VerticalScroll(id="training-panel"):
                        yield Static("TREINAR", classes="panel-title")
                        yield Label("Dataset:", classes="setting-label")
                        yield Select([], id="dataset-selector", prompt="Selecione o dataset")
                        yield Button("ATUALIZAR", id="btn-refresh-datasets")
                        yield Label("Usar melhores:", classes="setting-label")
                        with Horizontal(classes="use-top-row"):
                            yield Checkbox("TOP", id="use-top-checkbox", value=True)
                            yield Input(value="10", id="input-top-n", type="integer")
                        yield Label("Epochs (Coqui):", classes="setting-label")
                        yield Input(value="100", id="input-epochs", type="integer")
                        yield ProgressBar(id="training-progress", total=100, show_eta=False)
                        yield Static("[dim]Chatterbox: zero-shot (embeddings)[/]", classes="model-hint")
                        yield Static("[dim]Coqui: fine-tuning real (lento)[/]", classes="model-hint")
                        with Horizontal(classes="train-buttons"):
                            yield Button("CHATTERBOX", id="btn-train-chatterbox")
                            yield Button("COQUI", id="btn-train-coqui")

                with Horizontal(id="test-panel"):
                    with Vertical(id="test-left"):
                        yield Static("TESTAR", classes="panel-title")
                        yield Label("Modelo treinado:", classes="setting-label")
                        yield Select([], id="model-selector-test", prompt="Selecione o modelo")
                        yield Button("ATUALIZAR", id="btn-refresh-models")

                    with Vertical(id="test-right"):
                        yield Label("Texto para sintetizar:", classes="setting-label")
                        yield TextArea(id="input-test-text")
                        yield Button("OUVIR", id="btn-play")

                yield RichLog(id="output-log", markup=True, highlight=True, wrap=True, classes="hidden-log")

        yield Footer()

    def on_mount(self) -> None:
        self._log(f"[bold {COLORS['accent_primary']}]Clone Voice Dataset Generator[/]")
        self._log(f"[{COLORS['text_muted']}]Gera datasets de voz via ElevenLabs para treinar modelos TTS[/]")
        self._log("")

        self._load_saved_settings()
        self._update_file_status()
        self._update_save_indicator("Config carregada")

        if LUNA_PHRASES_FILE.exists():
            self._log(f"[{COLORS['success']}]LUNA: {LUNA_PHRASES_FILE.name}[/]")
        if ERIS_PHRASES_FILE.exists():
            self._log(f"[#ff5555]ERIS: {ERIS_PHRASES_FILE.name}[/]")
        if DEFAULT_PHRASES_FILE.exists():
            self._log(f"[{COLORS['info']}]TEMPLATE: {DEFAULT_PHRASES_FILE.name}[/]")

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            self._log(f"[{COLORS['success']}]API Key encontrada no ambiente[/]")
        else:
            self._log(f"[{COLORS['info']}]Insira sua API Key do ElevenLabs[/]")

        self._log("")
        self._log(f"[{COLORS['text_secondary']}]Configure e clique em INICIAR[/]")

        self._refresh_datasets()
        self._refresh_models()

    def _load_saved_settings(self) -> None:
        config = load_saved_config()
        if not config:
            return

        try:
            if config.get("api_key"):
                self.query_one("#input-api-key", Input).value = config["api_key"]
            if config.get("voice_id"):
                self.query_one("#input-voice-id", Input).value = config["voice_id"]
            if config.get("dataset_name"):
                self.query_one("#input-dataset-name", Input).value = config["dataset_name"]
            if config.get("phrase_count"):
                self.query_one("#input-phrase-count", Input).value = str(config["phrase_count"])
            if config.get("phrases_file"):
                self.query_one("#input-phrases-file", Input).value = config["phrases_file"]
            if config.get("output_dir"):
                self.query_one("#input-output-dir", Input).value = config["output_dir"]

            model_id = config.get("model_id", "eleven_multilingual_v2")
            if model_id == "eleven_turbo_v2_5":
                self.query_one("#model-turbo", RadioButton).value = True
            elif model_id == "eleven_flash_v2":
                self.query_one("#model-flash", RadioButton).value = True
            else:
                self.query_one("#model-multi-v2", RadioButton).value = True

            self._log(f"[{COLORS['success']}]Config carregada[/]")
        except Exception as e:
            logger.warning(f"Erro ao restaurar settings: {e}")

    def _save_current_settings(self, show_feedback: bool = False) -> None:
        try:
            config = {
                "api_key": self.query_one("#input-api-key", Input).value.strip(),
                "voice_id": self.query_one("#input-voice-id", Input).value.strip(),
                "dataset_name": self.query_one("#input-dataset-name", Input).value.strip(),
                "phrase_count": self.query_one("#input-phrase-count", Input).value.strip(),
                "model_id": self._get_model_id(),
                "phrases_file": self.query_one("#input-phrases-file", Input).value.strip(),
                "output_dir": self.query_one("#input-output-dir", Input).value.strip(),
            }
            save_config(config)
            self._config_dirty = False
            if show_feedback:
                self._update_save_indicator("Salvo")
        except Exception as e:
            logger.warning(f"Erro ao salvar settings: {e}")

    def _schedule_save(self) -> None:
        self._config_dirty = True
        self._update_save_indicator("...")
        if self._save_timer:
            self._save_timer.stop()
        self._save_timer = self.set_timer(
            self.SAVE_DEBOUNCE_MS / 1000,
            self._execute_save
        )

    def _execute_save(self) -> None:
        if self._config_dirty:
            self._save_current_settings(show_feedback=True)

    def _update_save_indicator(self, text: str) -> None:
        try:
            indicator = self.query_one("#save-indicator", Static)
            indicator.update(f"[dim]{text}[/]")
        except Exception:
            pass

    def _validate_api_key(self, value: str) -> bool:
        return value.startswith("sk_") and len(value) > 10

    def on_input_changed(self, event: Input.Changed) -> None:
        input_ids = [
            "input-api-key", "input-voice-id", "input-dataset-name",
            "input-phrase-count", "input-phrases-file", "input-output-dir"
        ]
        if event.input.id in input_ids:
            self._schedule_save()

        if event.input.id == "input-api-key":
            api_key = event.value.strip()
            if api_key:
                if self._validate_api_key(api_key):
                    self._update_save_indicator("[#22c55e]API valida[/]")
                else:
                    self._update_save_indicator("[#ef4444]API invalida[/]")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "model-selector":
            self._schedule_save()

    def on_unmount(self) -> None:
        if self._config_dirty:
            self._save_current_settings()

    def _log(self, message: str) -> None:
        try:
            self.query_one("#output-log", RichLog).write(message)
        except Exception:
            pass

    def _update_counter(self) -> None:
        try:
            counter = self.query_one("#counter-display", Static)
            counter.update(f"{self.phrases_generated} / {self.target_phrases} frases")

            progress = self.query_one("#generation-progress", ProgressBar)
            if self.target_phrases > 0:
                pct = (self.phrases_generated / self.target_phrases) * 100
                progress.update(progress=pct)
        except Exception:
            pass

    def _update_current_phrase(self, phrase: str) -> None:
        try:
            display = self.query_one("#current-phrase", Static)
            truncated = phrase[:50] + "..." if len(phrase) > 50 else phrase
            display.update(f"[{COLORS['text_muted']}]{truncated}[/]")
        except Exception:
            pass

    def _get_model_id(self) -> str:
        try:
            radio_set = self.query_one("#model-selector", RadioSet)
            pressed = radio_set.pressed_button
            if pressed:
                if pressed.id == "model-turbo":
                    return "eleven_turbo_v2_5"
                elif pressed.id == "model-flash":
                    return "eleven_flash_v2"
        except Exception:
            pass
        return "eleven_multilingual_v2"

    def action_start_generation(self) -> None:
        if not self.is_generating:
            self._start_generation()

    def action_stop_generation(self) -> None:
        if self.is_generating:
            self._stop_generation()

    def _start_generation(self) -> None:
        self._save_current_settings(show_feedback=False)

        api_key = self.query_one("#input-api-key", Input).value.strip()
        if not api_key:
            api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            self.notify_error("API Key obrigatoria")
            self._log(f"[{COLORS['error']}]API Key nao configurada[/]")
            return

        if not self._validate_api_key(api_key):
            self.notify_error("API Key invalida")
            self._log(f"[{COLORS['error']}]Formato de API Key invalido (deve comecar com sk_)[/]")
            return

        voice_id = self.query_one("#input-voice-id", Input).value.strip()
        if not voice_id:
            self.notify_error("Voice ID obrigatorio")
            self._log(f"[{COLORS['error']}]Voice ID nao configurado[/]")
            return

        try:
            self.target_phrases = int(self.query_one("#input-phrase-count", Input).value)
        except ValueError:
            self.notify_error("Quantidade invalida")
            return

        dataset_name = self.query_one("#input-dataset-name", Input).value.strip()
        if not dataset_name:
            dataset_name = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        phrases_file = Path(self.query_one("#input-phrases-file", Input).value.strip())
        output_dir = Path(self.query_one("#input-output-dir", Input).value.strip())
        model_id = self._get_model_id()

        self.is_generating = True
        self.phrases_generated = 0

        status = self.query_one("#status-indicator", StatusPanel)
        status.status = "initializing"

        self.query_one("#btn-generate", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False

        self._log("")
        self._log(f"[{COLORS['info']}]Inicializando...[/]")
        self._log(f"[{COLORS['text_muted']}]Voice: {voice_id} | Modelo: {model_id}[/]")

        self._update_counter()

        self._generation_thread = threading.Thread(
            target=self._run_generation,
            args=(api_key, voice_id, dataset_name, model_id, phrases_file, output_dir),
            daemon=True
        )
        self._generation_thread.start()

    def _run_generation(
        self,
        api_key: str,
        voice_id: str,
        dataset_name: str,
        model_id: str,
        phrases_file: Path,
        output_dir: Path,
    ) -> None:
        try:
            from src.modules.clone_voice.core.generator import DatasetManager, DatasetConfig

            config = DatasetConfig(
                name=dataset_name,
                voice_id=voice_id,
                model_id=model_id,
                total_phrases=self.target_phrases,
                output_dir=output_dir,
                phrases_file=phrases_file if phrases_file.exists() else None,
            )

            self._dataset_manager = DatasetManager(config, api_key=api_key)

            def on_progress(current: int, total: int, phrase: str):
                self.call_from_thread(self._on_generation_progress, current, total, phrase)

            def on_phrase_generated(index: int, phrase: str, audio_path: Path):
                self.call_from_thread(self._on_phrase_generated, index, phrase, audio_path)

            def on_error(error: str):
                self.call_from_thread(self._on_generation_error, error)

            self._dataset_manager.on_progress = on_progress
            self._dataset_manager.on_phrase_generated = on_phrase_generated
            self._dataset_manager.on_error = on_error

            self.call_from_thread(self._log, f"[{COLORS['success']}]Gerador OK[/]")
            self.call_from_thread(self._set_status, "generating")

            if not self._dataset_manager.initialize():
                self.call_from_thread(self._on_generation_failed, "Falha ao inicializar ElevenLabs")
                return

            stats = self._dataset_manager.generate_dataset()

            if self._dataset_manager._should_stop:
                self.call_from_thread(self._on_generation_stopped, stats)
            else:
                self.call_from_thread(self._on_generation_complete, stats)

        except ImportError as e:
            self.call_from_thread(self._on_generation_failed, f"Pacote nao instalado: {e}")
        except Exception as e:
            self.call_from_thread(self._on_generation_failed, str(e))

    def _set_status(self, status: str) -> None:
        try:
            panel = self.query_one("#status-indicator", StatusPanel)
            panel.status = status
        except Exception:
            pass

    def _on_generation_progress(self, current: int, total: int, phrase: str) -> None:
        self.phrases_generated = current
        self.target_phrases = total
        self._update_counter()
        self._update_current_phrase(phrase)

    def _on_phrase_generated(self, index: int, phrase: str, audio_path: Path) -> None:
        short_phrase = phrase[:35] + "..." if len(phrase) > 35 else phrase
        self._log(f"[{COLORS['success']}][{index+1}][/] {short_phrase}")

    def _on_generation_error(self, error: str) -> None:
        self._log(f"[{COLORS['error']}]Erro: {error}[/]")

    def _on_generation_failed(self, error: str) -> None:
        self.is_generating = False
        self._set_status("error")

        self.query_one("#btn-generate", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True

        self._log("")
        self._log(f"[{COLORS['error']}]Falha: {error}[/]")
        self.notify_error("Falha na geracao")

    def _on_generation_stopped(self, stats) -> None:
        self.is_generating = False
        self._set_status("stopped")

        self.query_one("#btn-generate", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True

        self._log("")
        self._log(f"[{COLORS['warning']}]Interrompido: {stats.total_generated}/{stats.total_requested}[/]")
        self.notify_warning("Geracao interrompida")

    def _on_generation_complete(self, stats) -> None:
        self.is_generating = False
        self._set_status("completed")

        self.query_one("#btn-generate", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True

        self._log("")
        self._log(f"[bold {COLORS['success']}]CONCLUIDO[/]")
        self._log(f"[{COLORS['text_primary']}]Gerados: {stats.total_generated}/{stats.total_requested} | Taxa: {stats.success_rate:.0f}%[/]")

        if self._dataset_manager:
            output_dir = self._dataset_manager.config.output_dir
            dataset_name = self._dataset_manager.config.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = output_dir / f"{dataset_name}_{timestamp}"
            self._log(f"[{COLORS['neon_cyan']}]{full_path}[/]")

        self.notify_success(f"Dataset: {stats.total_generated} amostras")

    def _stop_generation(self) -> None:
        if self._dataset_manager:
            self._dataset_manager.stop()
            self._log(f"[{COLORS['warning']}]Parando...[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "btn-generate":
            self._start_generation()
        elif button_id == "btn-stop":
            self._stop_generation()
        elif button_id == "btn-browse":
            self._browse_phrases_file()
        elif button_id == "btn-refresh-datasets":
            self._refresh_datasets()
        elif button_id == "btn-train-chatterbox":
            self._start_training("chatterbox")
        elif button_id == "btn-train-coqui":
            self._start_training("coqui")
        elif button_id == "btn-refresh-models":
            self._refresh_models()
        elif button_id == "btn-play":
            self._play_test_audio()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "dataset-selector" and event.value != Select.BLANK:
            option_id = str(event.value)
            if option_id in self._datasets_map:
                self._selected_dataset = self._datasets_map[option_id]
                self._log(f"[{COLORS['info']}]Selecionado: {self._selected_dataset.name}[/]")
        elif event.select.id == "model-selector-test" and event.value != Select.BLANK:
            option_id = str(event.value)
            if option_id in self._models_map:
                self._selected_model = self._models_map[option_id]
                self._log(f"[{COLORS['info']}]Modelo: {self._selected_model.name}[/]")

    def _browse_phrases_file(self) -> None:
        try:
            result = subprocess.run(
                [
                    "zenity", "--file-selection",
                    "--title=Selecionar Arquivo de Frases",
                    "--file-filter=Markdown (*.md) | *.md",
                    "--file-filter=Texto (*.txt) | *.txt",
                    "--file-filter=Todos (*) | *"
                ],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                filepath = Path(result.stdout.strip())
                self._set_phrases_file(filepath)
        except Exception as e:
            self._log(f"[{COLORS['error']}]Erro ao abrir dialogo: {e}[/]")

    def _set_phrases_file(self, filepath: Path) -> None:
        input_field = self.query_one("#input-phrases-file", Input)
        status_label = self.query_one("#file-status", Label)

        input_field.value = str(filepath)

        if filepath.exists():
            from src.modules.clone_voice.core.generator import PhraseParser
            parser = PhraseParser()
            if parser.parse_file(filepath):
                stats = parser.get_stats()
                categories = len(parser.get_category_names())
                status_label.update(f"[{COLORS['success']}]{stats['total']} frases, {categories} cats[/]")
                self._log(f"[{COLORS['success']}]{filepath.name}: {stats['total']} frases[/]")
            else:
                status_label.update(f"[{COLORS['error']}]Arquivo vazio[/]")
        else:
            status_label.update(f"[{COLORS['error']}]Nao encontrado[/]")

    def _update_file_status(self) -> None:
        status_label = self.query_one("#file-status", Label)
        current_file = self.query_one("#input-phrases-file", Input).value.strip()
        
        if current_file:
            filepath = Path(current_file)
            if filepath.exists():
                from src.modules.clone_voice.core.generator import PhraseParser
                parser = PhraseParser()
                if parser.parse_file(filepath):
                    stats = parser.get_stats()
                    status_label.update(f"[{COLORS['success']}]{stats['total']} frases[/]")
                else:
                    status_label.update(f"[{COLORS['warning']}]Arquivo vazio[/]")
            else:
                status_label.update(f"[{COLORS['text_muted']}]Selecione um arquivo[/]")
        else:
            status_label.update(f"[{COLORS['text_muted']}]Selecione um arquivo[/]")

    def _refresh_datasets(self) -> None:
        output_dir = Path(self.query_one("#input-output-dir", Input).value.strip())
        selector = self.query_one("#dataset-selector", Select)

        if not output_dir.exists():
            selector.set_options([])
            return

        datasets = []
        for item in output_dir.iterdir():
            if item.is_dir():
                metadata = item / "metadata.json"
                wavs_dir = item / "wavs"
                if metadata.exists() and wavs_dir.exists():
                    audio_count = len(list(wavs_dir.glob("*.mp3"))) + len(list(wavs_dir.glob("*.wav")))
                    if audio_count > 0:
                        datasets.append((item, audio_count))

        if not datasets:
            selector.set_options([])
            self._selected_dataset = None
            self._datasets_map = {}
            return

        datasets.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        self._datasets_map = {}
        options = []
        for i, (ds_path, audio_count) in enumerate(datasets):
            ds_id = f"ds_{i}"
            name = ds_path.name
            label = f"{name} ({audio_count})"
            options.append((label, ds_id))
            self._datasets_map[ds_id] = ds_path

        selector.set_options(options)
        self._selected_dataset = datasets[0][0]
        self._log(f"[{COLORS['success']}]Datasets encontrados: {len(datasets)}[/]")

    def _start_training(self, model_type: str) -> None:
        if self.is_training:
            self.notify_error("Treinamento em andamento")
            return

        if not self._selected_dataset:
            self._refresh_datasets()
            if not self._selected_dataset:
                self.notify_error("Selecione um dataset")
                self._log(f"[{COLORS['error']}]Selecione um dataset primeiro[/]")
                return

        try:
            epochs = int(self.query_one("#input-epochs", Input).value)
        except ValueError:
            self.notify_error("Epochs invalido")
            return

        use_top = self.query_one("#use-top-checkbox", Checkbox).value
        top_n = 10
        try:
            top_n = int(self.query_one("#input-top-n", Input).value)
        except ValueError:
            top_n = 10

        output_dir = self._selected_dataset.parent / "trained_models"

        self.is_training = True
        self.training_progress = 0

        self.query_one("#btn-train-chatterbox", Button).disabled = True
        self.query_one("#btn-train-coqui", Button).disabled = True

        self._log("")
        self._log(f"[{COLORS['info']}]Treinar {model_type.upper()}...[/]")
        if use_top:
            self._log(f"[{COLORS['text_muted']}]Usando TOP {top_n} audios de {self._selected_dataset.name}[/]")
        else:
            self._log(f"[{COLORS['text_muted']}]Usando todos audios de {self._selected_dataset.name}[/]")
        self._log(f"[{COLORS['text_muted']}]Epochs: {epochs}[/]")

        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(model_type, self._selected_dataset, output_dir, epochs, use_top, top_n),
            daemon=True
        )
        self._training_thread.start()

    def _run_training(
        self,
        model_type: str,
        dataset_dir: Path,
        output_dir: Path,
        epochs: int,
        use_top: bool = True,
        top_n: int = 10
    ) -> None:
        try:
            from src.modules.clone_voice.core.training import (
                TrainingConfig, ChatterboxTrainer, CoquiTrainer
            )
            from src.modules.clone_voice.core import AudioQualityAnalyzer

            training_dataset_dir = dataset_dir

            unified_audio_path = None

            if use_top:
                self.call_from_thread(self._log, f"[{COLORS['info']}]Selecionando TOP {top_n} audios...[/]")

                analyzer = AudioQualityAnalyzer()
                result = analyzer.analyze_dataset(dataset_dir)

                if result.analyzed_files == 0:
                    self.call_from_thread(self._on_training_error, "Nenhum audio no dataset")
                    return

                selection = analyzer.get_diverse_selection(top_n)

                if not selection:
                    self.call_from_thread(self._on_training_error, "Falha ao selecionar audios")
                    return

                selection_dir = dataset_dir / f"top_{top_n}_selection"
                analyzer.export_selection(selection, selection_dir)

                self.call_from_thread(
                    self._log,
                    f"[{COLORS['info']}]Concatenando {len(selection)} audios em arquivo unico...[/]"
                )

                unified_audio_path = analyzer.create_unified_reference(selection, selection_dir)

                if unified_audio_path and unified_audio_path.exists():
                    duration = unified_audio_path.stat().st_size / (22050 * 2)
                    self.call_from_thread(
                        self._log,
                        f"[{COLORS['success']}]Audio unificado: {unified_audio_path.name}[/]"
                    )
                else:
                    self.call_from_thread(self._on_training_error, "Falha ao criar audio unificado")
                    return

                wavs_dir = selection_dir / "wavs"
                wavs_dir.mkdir(parents=True, exist_ok=True)
                for m in selection:
                    src = m.file_path
                    dst = wavs_dir / m.file_name
                    if not dst.exists():
                        import shutil
                        shutil.copy(src, dst)

                training_dataset_dir = selection_dir

                self.call_from_thread(
                    self._log,
                    f"[{COLORS['success']}]TOP {top_n}: {len(selection)} audios prontos[/]"
                )

            import re
            base_name = re.sub(r'_\d{8}_\d{6}$', '', dataset_dir.name)

            config = TrainingConfig(
                dataset_dir=training_dataset_dir,
                output_dir=output_dir,
                model_name=base_name,
                epochs=epochs,
                unified_reference_path=unified_audio_path,
            )

            if model_type == "chatterbox":
                trainer = ChatterboxTrainer(config)
            else:
                trainer = CoquiTrainer(config)

            self._trainer = trainer

            trainer.on_progress = lambda step, total, loss: self.call_from_thread(
                self._on_training_progress, step, total, loss
            )
            trainer.on_epoch_complete = lambda epoch, loss: self.call_from_thread(
                self._on_epoch_complete, epoch, loss
            )
            trainer.on_log = lambda msg: self.call_from_thread(self._log, msg)
            trainer.on_complete = lambda stats: self.call_from_thread(
                self._on_training_complete, stats
            )
            trainer.on_error = lambda err: self.call_from_thread(
                self._on_training_error, err
            )

            self.call_from_thread(self._update_training_status, "VALIDANDO")

            if not trainer.validate_dataset():
                self.call_from_thread(self._on_training_error, "Dataset invalido")
                return

            self.call_from_thread(self._update_training_status, "INICIALIZANDO")

            if not trainer.initialize():
                self.call_from_thread(self._on_training_error, "Falha ao inicializar")
                return

            self.call_from_thread(self._update_training_status, "TREINANDO")

            trainer.train()

        except ImportError as e:
            self.call_from_thread(self._on_training_error, f"Dependencia: {e}")
        except Exception as e:
            self.call_from_thread(self._on_training_error, str(e))

    def _update_training_status(self, status: str) -> None:
        self._log(f"[{COLORS['warning']}]{status}[/]")

    def _on_training_progress(self, step: int, total: int, loss: float) -> None:
        try:
            progress = self.query_one("#training-progress", ProgressBar)
            pct = (step / total) * 100 if total > 0 else 0
            progress.update(progress=pct)
            self.training_progress = int(pct)
        except Exception:
            pass

    def _on_epoch_complete(self, epoch: int, loss: float) -> None:
        self._log(f"[{COLORS['info']}]Epoch {epoch} | Loss: {loss:.4f}[/]")

    def _on_training_complete(self, stats) -> None:
        self.is_training = False

        self.query_one("#btn-train-chatterbox", Button).disabled = False
        self.query_one("#btn-train-coqui", Button).disabled = False

        if stats.is_completed:
            self._log("")
            self._log(f"[bold {COLORS['success']}]TREINAMENTO OK[/]")
            self._log(f"[{COLORS['text_muted']}]Epochs: {stats.current_epoch} | Loss: {stats.best_loss:.4f}[/]")

            if stats.output_path:
                self._last_output_dir = stats.output_path
                self._log(f"[{COLORS['neon_cyan']}]{stats.output_path}[/]")
                self.push_screen(TrainingCompleteModal(stats.output_path, stats.model_type))

            self.notify_success("Treinamento concluido")
        else:
            self._log(f"[{COLORS['warning']}]Treinamento interrompido[/]")

    def _on_training_error(self, error: str) -> None:
        self.is_training = False

        self.query_one("#btn-train-chatterbox", Button).disabled = False
        self.query_one("#btn-train-coqui", Button).disabled = False

        self._log(f"[{COLORS['error']}]Erro: {error}[/]")
        self.notify_error("Erro no treinamento")

    def action_start_training(self) -> None:
        if not self.is_training:
            self._start_training("coqui")

    def _select_best_audios(self) -> None:
        if not self._selected_dataset:
            self._refresh_datasets()
            if not self._selected_dataset:
                self.notify_error("Selecione um dataset primeiro")
                return

        self._log("")
        self._log(f"[{COLORS['info']}]Analisando qualidade dos audios...[/]")
        self._log(f"[{COLORS['text_muted']}]Dataset: {self._selected_dataset.name}[/]")

        threading.Thread(
            target=self._run_audio_analysis,
            args=(self._selected_dataset,),
            daemon=True
        ).start()

    def _run_audio_analysis(self, dataset_path: Path) -> None:
        try:
            from src.modules.clone_voice.core import AudioQualityAnalyzer

            analyzer = AudioQualityAnalyzer()

            def on_progress(current: int, total: int, filename: str):
                self.call_from_thread(
                    self._log,
                    f"[{COLORS['text_muted']}]Analisando {current}/{total}: {filename}[/]"
                )

            analyzer.set_progress_callback(on_progress)

            result = analyzer.analyze_dataset(dataset_path)

            if result.analyzed_files == 0:
                self.call_from_thread(
                    self._log,
                    f"[{COLORS['error']}]Nenhum audio encontrado no dataset[/]"
                )
                return

            selection = analyzer.get_diverse_selection(10)

            if not selection:
                self.call_from_thread(
                    self._log,
                    f"[{COLORS['error']}]Nao foi possivel selecionar audios[/]"
                )
                return

            output_path = dataset_path / "top_10_selection"
            exported = analyzer.export_selection(selection, output_path)

            self.call_from_thread(self._on_audio_analysis_complete, result, selection, output_path)

        except Exception as e:
            self.call_from_thread(
                self._log,
                f"[{COLORS['error']}]Erro na analise: {e}[/]"
            )

    def _on_audio_analysis_complete(
        self,
        result,
        selection: list,
        output_path: Path
    ) -> None:
        self._log("")
        self._log(f"[bold {COLORS['success']}]ANALISE COMPLETA[/]")
        self._log(f"[{COLORS['text_primary']}]Arquivos analisados: {result.analyzed_files}[/]")
        self._log(f"[{COLORS['text_primary']}]Duracao total: {result.total_duration_seconds:.1f}s[/]")
        self._log(f"[{COLORS['text_primary']}]Score medio: {result.avg_quality_score:.1f}[/]")
        self._log("")
        self._log(f"[bold {COLORS['neon_cyan']}]TOP 10 AUDIOS SELECIONADOS:[/]")

        for i, m in enumerate(selection, 1):
            self._log(
                f"[{COLORS['success']}]{i:2d}.[/] {m.file_name} "
                f"[{COLORS['text_muted']}]({m.duration_seconds:.1f}s, score: {m.quality_score:.0f})[/]"
            )

        self._log("")
        self._log(f"[{COLORS['info']}]Arquivos exportados em:[/]")
        self._log(f"[{COLORS['neon_cyan']}]{output_path}[/]")
        self._log(f"[{COLORS['text_muted']}]- metadata.csv (Coqui)[/]")
        self._log(f"[{COLORS['text_muted']}]- chatterbox_data.jsonl[/]")
        self._log(f"[{COLORS['text_muted']}]- selection_info.json[/]")

        self.notify_success(f"Top 10 audios selecionados em {output_path.name}")

    def _validate_api_key_before_generate(self) -> bool:
        api_key = self.query_one("#input-api-key", Input).value.strip()
        if not api_key:
            api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            return False

        self._log(f"[{COLORS['info']}]Validando API Key...[/]")

        try:
            from src.modules.clone_voice.core.generator import ElevenLabsClient
            client = ElevenLabsClient(api_key=api_key)
            if client.validate_api_key():
                self._log(f"[{COLORS['success']}]API Key valida[/]")
                return True
            else:
                self._log(f"[{COLORS['error']}]API Key invalida ou expirada[/]")
                return False
        except Exception as e:
            self._log(f"[{COLORS['error']}]Erro ao validar API Key: {e}[/]")
            return False

    def _refresh_models(self) -> None:
        try:
            model_list = self.query_one("#model-selector-test", Select)
            self._models_map.clear()

            trained_models_dir = DEFAULT_OUTPUT_DIR / "trained_models"
            if not trained_models_dir.exists():
                model_list.set_options([])
                return

            models = []
            for item in trained_models_dir.iterdir():
                if item.is_dir() and (item.name.endswith("_chatterbox") or item.name.endswith("_coqui")):
                    models.append(item)

            if not models:
                model_list.set_options([])
                self._selected_model = None
                return

            models.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            options = []
            for model_path in models:
                model_id = model_path.name
                self._models_map[model_id] = model_path
                options.append((model_path.name, model_id))

            model_list.set_options(options)

            if models:
                self._selected_model = models[0]

        except Exception as e:
            self._log(f"[{COLORS['error']}]Erro ao listar modelos: {e}[/]")

    def _play_test_audio(self) -> None:
        if self._is_playing:
            self._log(f"[{COLORS['warning']}]Geracao em andamento...[/]")
            return

        if not self._selected_model:
            self._refresh_models()
            if not self._selected_model:
                self.notify_error("Selecione um modelo")
                return

        text = self.query_one("#input-test-text", TextArea).text.strip()
        if not text:
            self.notify_error("Digite um texto")
            return

        self._is_playing = True
        self.query_one("#btn-play", Button).disabled = True

        self._log("")
        self._log(f"[{COLORS['info']}]Gerando audio com {self._selected_model.name}...[/]")

        threading.Thread(
            target=self._run_inference,
            args=(self._selected_model, text),
            daemon=True
        ).start()

    def _run_inference(self, model_path: Path, text: str) -> None:
        try:
            output_file = model_path / "outputs" / "test_output.wav"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if model_path.name.endswith("_chatterbox"):
                self._run_chatterbox_inference(model_path, text, output_file)
            else:
                self._run_coqui_inference(model_path, text, output_file)

            if output_file.exists():
                self.call_from_thread(self._log, f"[{COLORS['success']}]Audio gerado![/]")
                self.call_from_thread(self._log, f"[{COLORS['text_muted']}]Reproduzindo...[/]")
                self._play_audio_file(output_file)
                self.call_from_thread(self._log, f"[{COLORS['info']}]Reproducao concluida[/]")
            else:
                self.call_from_thread(self._log, f"[{COLORS['error']}]Falha ao gerar audio[/]")

        except Exception as e:
            self.call_from_thread(self._log, f"[{COLORS['error']}]Erro: {e}[/]")
        finally:
            self._is_playing = False
            self.call_from_thread(self._enable_play_button)

    def _play_audio_file(self, audio_path: Path) -> None:
        import sounddevice as sd
        from scipy.io import wavfile

        sample_rate, audio_data = wavfile.read(str(audio_path))

        if audio_data.dtype == 'int16':
            audio_data = audio_data.astype('float32') / 32768.0
        elif audio_data.dtype == 'int32':
            audio_data = audio_data.astype('float32') / 2147483648.0

        sd.play(audio_data, sample_rate)
        sd.wait()

    def _enable_play_button(self) -> None:
        self.query_one("#btn-play", Button).disabled = False

    def _run_chatterbox_inference(self, model_path: Path, text: str, output_file: Path) -> None:
        python_exec = str(ROOT_DIR / "venv_chatterbox" / "bin" / "python")
        reference_audio = model_path / "reference.wav"

        if not reference_audio.exists():
            raise FileNotFoundError(f"Audio de referencia nao encontrado: {reference_audio}")

        script = f'''
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
audio = model.generate(text="{text}", audio_prompt_path="{reference_audio}")
audio_cpu = audio.cpu()
if audio_cpu.dim() == 3:
    audio_cpu = audio_cpu.squeeze(0)
if audio_cpu.dim() == 1:
    audio_cpu = audio_cpu.unsqueeze(0)
torchaudio.save("{output_file}", audio_cpu, 24000)
print("OK")
'''
        result = subprocess.run(
            [python_exec, "-c", script],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    def _run_coqui_inference(self, model_path: Path, text: str, output_file: Path) -> None:
        python_exec = str(ROOT_DIR / "venv_coqui" / "bin" / "python")
        reference_audio = model_path / "reference_speaker.wav"

        if not reference_audio.exists():
            raise FileNotFoundError(f"Audio de referencia nao encontrado: {reference_audio}")

        script = f'''
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text="{text}", file_path="{output_file}", speaker_wav="{reference_audio}", language="pt")
print("OK")
'''
        result = subprocess.run(
            [python_exec, "-c", script],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)


def main():
    app = CloneVoiceApp()
    app.run()


if __name__ == "__main__":
    main()
