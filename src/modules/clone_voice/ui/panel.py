# -*- coding: utf-8 -*-

import os
import logging
import subprocess
import threading
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Static,
    Button,
    Label,
    Input,
    ProgressBar,
    RichLog,
    RadioSet,
    RadioButton,
    Select,
    Checkbox,
    TextArea,
)
from textual.containers import Vertical, Horizontal, VerticalScroll

from src.core.theme import COLORS
from src.modules.clone_voice.ui.panel_helpers import (
    StatusPanel,
    TrainingCompleteModal,
    load_saved_config,
    save_config,
    run_generation_thread,
    run_training_thread,
    run_chatterbox_inference,
    run_coqui_inference,
    CLONE_CSS,
    LUNA_PHRASES_FILE,
    LUNA_GOTHIC_PHRASES_FILE,
    ERIS_PHRASES_FILE,
    DEFAULT_PHRASES_FILE,
    DEFAULT_OUTPUT_DIR,
    ROOT_DIR,
)

logger = logging.getLogger(__name__)


class CloneVoicePanel(Widget):
    BINDINGS = [
        Binding("g", "start_generation", "Gerar", show=True),
        Binding("s", "stop_generation", "Parar", show=True),
        Binding("t", "start_training", "Treinar", show=True),
    ]

    DEFAULT_CSS = CLONE_CSS

    SAVE_DEBOUNCE_MS = 1500
    _save_timer = None
    _config_dirty = False

    is_generating: reactive[bool] = reactive(False)
    is_training: reactive[bool] = reactive(False)
    phrases_generated: reactive[int] = reactive(0)
    target_phrases: reactive[int] = reactive(50)
    training_progress: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
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
        with Horizontal(id="clone-main-container"):
            with Vertical(id="clone-left-column"):
                with VerticalScroll(id="clone-config-panel"):
                    yield Static("CONFIGURACAO", classes="clone-panel-title")
                    yield Static("", id="clone-save-indicator")

                    yield Label("API Key:", classes="clone-setting-label")
                    yield Input(placeholder="sk_xxxxxxxx...", id="clone-input-api-key")

                    yield Label("Voice ID:", classes="clone-setting-label")
                    yield Input(
                        placeholder="ID da voz no ElevenLabs", id="clone-input-voice-id"
                    )

                    yield Label("Nome do Dataset:", classes="clone-setting-label")
                    yield Input(
                        placeholder="Ex: MinhaVoz", id="clone-input-dataset-name"
                    )

                    yield Label("Qtd Frases:", classes="clone-setting-label")
                    yield Input(
                        value="50", id="clone-input-phrase-count", type="integer"
                    )

                    yield Label("Modelo TTS:", classes="clone-setting-label")
                    with RadioSet(id="clone-model-selector"):
                        yield RadioButton(
                            "Multilingual v2", id="clone-model-multi-v2", value=True
                        )
                        yield RadioButton("Turbo v2.5", id="clone-model-turbo")
                        yield RadioButton("Flash v2", id="clone-model-flash")

                    yield Static("", classes="clone-section-separator")

                    yield Label("Arquivo de Frases:", classes="clone-setting-label")
                    yield Input(
                        placeholder="Caminho do arquivo .md",
                        id="clone-input-phrases-file",
                    )
                    yield Button("PROCURAR", id="clone-btn-browse")
                    yield Label("", id="clone-file-status", classes="clone-file-hint")

                    yield Label("Diretorio Saida:", classes="clone-setting-label")
                    yield Input(
                        value=str(DEFAULT_OUTPUT_DIR), id="clone-input-output-dir"
                    )

            with Vertical(id="clone-right-column"):
                with Horizontal(id="clone-right-top"):
                    with Vertical(id="clone-generator-panel"):
                        yield Static("GERADOR", classes="clone-panel-title")
                        yield StatusPanel(id="clone-status-indicator")
                        yield Static(
                            "0 / 50 frases",
                            id="clone-counter-display",
                            classes="clone-counter-display",
                        )
                        yield ProgressBar(
                            id="clone-generation-progress", total=100, show_eta=False
                        )
                        yield Static(
                            "",
                            id="clone-current-phrase",
                            classes="clone-current-phrase",
                        )
                        yield Button("INICIAR", id="clone-btn-generate")
                        yield Button("PARAR", id="clone-btn-stop", disabled=True)

                    with VerticalScroll(id="clone-training-panel"):
                        yield Static("TREINAR", classes="clone-panel-title")
                        yield Label("Dataset:", classes="clone-setting-label")
                        yield Select(
                            [],
                            id="clone-dataset-selector",
                            prompt="Selecione o dataset",
                        )
                        yield Button("ATUALIZAR", id="clone-btn-refresh-datasets")
                        yield Label("Usar melhores:", classes="clone-setting-label")
                        with Horizontal(classes="clone-use-top-row"):
                            yield Checkbox(
                                "TOP", id="clone-use-top-checkbox", value=True
                            )
                            yield Input(
                                value="10", id="clone-input-top-n", type="integer"
                            )
                        yield Label("Epochs (Coqui):", classes="clone-setting-label")
                        yield Input(
                            value="100", id="clone-input-epochs", type="integer"
                        )
                        yield ProgressBar(
                            id="clone-training-progress", total=100, show_eta=False
                        )
                        yield Static(
                            "[dim]Chatterbox: zero-shot (embeddings)[/]",
                            classes="clone-model-hint",
                        )
                        yield Static(
                            "[dim]Coqui: fine-tuning real (lento)[/]",
                            classes="clone-model-hint",
                        )
                        with Horizontal(classes="clone-train-buttons"):
                            yield Button("CHATTERBOX", id="clone-btn-train-chatterbox")
                            yield Button("COQUI", id="clone-btn-train-coqui")

                with Horizontal(id="clone-test-panel"):
                    with Vertical(id="clone-test-left"):
                        yield Static("TESTAR", classes="clone-panel-title")
                        yield Label("Modelo treinado:", classes="clone-setting-label")
                        yield Select(
                            [],
                            id="clone-model-selector-test",
                            prompt="Selecione o modelo",
                        )
                        yield Button("ATUALIZAR", id="clone-btn-refresh-models")

                    with Vertical(id="clone-test-right"):
                        yield Label(
                            "Texto para sintetizar:", classes="clone-setting-label"
                        )
                        yield TextArea(id="clone-input-test-text")
                        yield Button("OUVIR", id="clone-btn-play")

                yield RichLog(
                    id="clone-output-log",
                    markup=True,
                    highlight=True,
                    wrap=True,
                    classes="hidden-log",
                )

    def on_mount(self) -> None:
        self._log(f"[bold {COLORS['accent_primary']}]Clone Voice Dataset Generator[/]")
        self._log(
            f"[{COLORS['text_muted']}]Gera datasets de voz via ElevenLabs para treinar modelos TTS[/]"
        )
        self._log("")

        self._load_saved_settings()
        self._update_file_status()
        self._update_save_indicator("Config carregada")

        if LUNA_PHRASES_FILE.exists():
            self._log(f"[{COLORS['success']}]LUNA: {LUNA_PHRASES_FILE.name}[/]")
        if LUNA_GOTHIC_PHRASES_FILE.exists():
            self._log(
                f"[{COLORS['success']}]LUNA GOTHIC: {LUNA_GOTHIC_PHRASES_FILE.name}[/]"
            )
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

    def on_unmount(self) -> None:
        if self._config_dirty:
            self._save_current_settings()

    def _load_saved_settings(self) -> None:
        cfg = load_saved_config()
        if not cfg:
            return
        try:
            if cfg.get("api_key"):
                self.query_one("#clone-input-api-key", Input).value = cfg["api_key"]
            if cfg.get("voice_id"):
                self.query_one("#clone-input-voice-id", Input).value = cfg["voice_id"]
            if cfg.get("dataset_name"):
                self.query_one("#clone-input-dataset-name", Input).value = cfg[
                    "dataset_name"
                ]
            if cfg.get("phrase_count"):
                self.query_one("#clone-input-phrase-count", Input).value = str(
                    cfg["phrase_count"]
                )
            if cfg.get("phrases_file"):
                self.query_one("#clone-input-phrases-file", Input).value = cfg[
                    "phrases_file"
                ]
            if cfg.get("output_dir"):
                self.query_one("#clone-input-output-dir", Input).value = cfg[
                    "output_dir"
                ]

            model_id = cfg.get("model_id", "eleven_multilingual_v2")
            if model_id == "eleven_turbo_v2_5":
                self.query_one("#clone-model-turbo", RadioButton).value = True
            elif model_id == "eleven_flash_v2":
                self.query_one("#clone-model-flash", RadioButton).value = True
            else:
                self.query_one("#clone-model-multi-v2", RadioButton).value = True

            self._log(f"[{COLORS['success']}]Config carregada[/]")
        except Exception as e:
            logger.warning("Erro ao restaurar settings: %s", e)

    def _save_current_settings(self, show_feedback: bool = False) -> None:
        try:
            cfg = {
                "api_key": self.query_one("#clone-input-api-key", Input).value.strip(),
                "voice_id": self.query_one(
                    "#clone-input-voice-id", Input
                ).value.strip(),
                "dataset_name": self.query_one(
                    "#clone-input-dataset-name", Input
                ).value.strip(),
                "phrase_count": self.query_one(
                    "#clone-input-phrase-count", Input
                ).value.strip(),
                "model_id": self._get_model_id(),
                "phrases_file": self.query_one(
                    "#clone-input-phrases-file", Input
                ).value.strip(),
                "output_dir": self.query_one(
                    "#clone-input-output-dir", Input
                ).value.strip(),
            }
            save_config(cfg)
            self._config_dirty = False
            if show_feedback:
                self._update_save_indicator("Salvo")
        except Exception as e:
            logger.warning("Erro ao salvar settings: %s", e)

    def _schedule_save(self) -> None:
        self._config_dirty = True
        self._update_save_indicator("...")
        if self._save_timer:
            self._save_timer.stop()
        self._save_timer = self.set_timer(
            self.SAVE_DEBOUNCE_MS / 1000, self._execute_save
        )

    def _execute_save(self) -> None:
        if self._config_dirty:
            self._save_current_settings(show_feedback=True)

    def _update_save_indicator(self, text: str) -> None:
        try:
            self.query_one("#clone-save-indicator", Static).update(f"[dim]{text}[/]")
        except Exception:
            pass

    def _validate_api_key(self, value: str) -> bool:
        return value.startswith("sk_") and len(value) > 10

    def on_input_changed(self, event: Input.Changed) -> None:
        tracked_ids = {
            "clone-input-api-key",
            "clone-input-voice-id",
            "clone-input-dataset-name",
            "clone-input-phrase-count",
            "clone-input-phrases-file",
            "clone-input-output-dir",
        }
        if event.input.id in tracked_ids:
            self._schedule_save()

        if event.input.id == "clone-input-api-key":
            api_key = event.value.strip()
            if api_key:
                if self._validate_api_key(api_key):
                    self._update_save_indicator(f"[{COLORS['success']}]API valida[/]")
                else:
                    self._update_save_indicator(f"[{COLORS['error']}]API invalida[/]")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "clone-model-selector":
            self._schedule_save()

    def _log(self, message: str) -> None:
        try:
            self.query_one("#clone-output-log", RichLog).write(message)
        except Exception:
            pass

    def _update_counter(self) -> None:
        try:
            self.query_one("#clone-counter-display", Static).update(
                f"{self.phrases_generated} / {self.target_phrases} frases"
            )
            progress = self.query_one("#clone-generation-progress", ProgressBar)
            if self.target_phrases > 0:
                pct = (self.phrases_generated / self.target_phrases) * 100
                progress.update(progress=pct)
        except Exception:
            pass

    def _update_current_phrase(self, phrase: str) -> None:
        try:
            truncated = phrase[:50] + "..." if len(phrase) > 50 else phrase
            self.query_one("#clone-current-phrase", Static).update(
                f"[{COLORS['text_muted']}]{truncated}[/]"
            )
        except Exception:
            pass

    def _get_model_id(self) -> str:
        try:
            radio_set = self.query_one("#clone-model-selector", RadioSet)
            pressed = radio_set.pressed_button
            if pressed:
                if pressed.id == "clone-model-turbo":
                    return "eleven_turbo_v2_5"
                elif pressed.id == "clone-model-flash":
                    return "eleven_flash_v2"
        except Exception:
            pass
        return "eleven_multilingual_v2"

    def check_action(self, action: str, parameters: tuple) -> bool:
        controlled = {"start_generation", "stop_generation", "start_training"}
        if action in controlled:
            try:
                from textual.widgets import ContentSwitcher

                switcher = self.app.query_one(ContentSwitcher)
                return switcher.current == "panel-clone"
            except Exception:
                return True
        return True

    def action_start_generation(self) -> None:
        if not self.is_generating:
            self._start_generation()

    def action_stop_generation(self) -> None:
        if self.is_generating:
            self._stop_generation()

    def _set_sidebar_status(self, status: str | None) -> None:
        try:
            from src.core.widgets.nav_sidebar import NavSidebar

            self.app.query_one(NavSidebar).set_module_status("clone", status)
        except Exception:
            pass

    def _start_generation(self) -> None:
        self._save_current_settings(show_feedback=False)

        api_key = self.query_one("#clone-input-api-key", Input).value.strip()
        if not api_key:
            api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            self.app.notify("API Key obrigatoria", severity="error")
            self._log(f"[{COLORS['error']}]API Key nao configurada[/]")
            return

        if not self._validate_api_key(api_key):
            self.app.notify("API Key invalida", severity="error")
            self._log(
                f"[{COLORS['error']}]Formato de API Key invalido (deve comecar com sk_)[/]"
            )
            return

        voice_id = self.query_one("#clone-input-voice-id", Input).value.strip()
        if not voice_id:
            self.app.notify("Voice ID obrigatorio", severity="error")
            self._log(f"[{COLORS['error']}]Voice ID nao configurado[/]")
            return

        try:
            self.target_phrases = int(
                self.query_one("#clone-input-phrase-count", Input).value
            )
        except ValueError:
            self.app.notify("Quantidade invalida", severity="error")
            return

        dataset_name = self.query_one("#clone-input-dataset-name", Input).value.strip()
        if not dataset_name:
            dataset_name = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        phrases_file = Path(
            self.query_one("#clone-input-phrases-file", Input).value.strip()
        )
        output_dir = Path(
            self.query_one("#clone-input-output-dir", Input).value.strip()
        )
        model_id = self._get_model_id()

        self.is_generating = True
        self.phrases_generated = 0
        self._set_sidebar_status("GERANDO")
        self._set_status("initializing")
        self.query_one("#clone-btn-generate", Button).disabled = True
        self.query_one("#clone-btn-stop", Button).disabled = False

        self._log("")
        self._log(f"[{COLORS['info']}]Inicializando...[/]")
        self._log(f"[{COLORS['text_muted']}]Voice: {voice_id} | Modelo: {model_id}[/]")
        self._update_counter()

        self._generation_thread = threading.Thread(
            target=run_generation_thread,
            args=(
                self,
                api_key,
                voice_id,
                dataset_name,
                model_id,
                phrases_file,
                output_dir,
            ),
            daemon=True,
        )
        self._generation_thread.start()

    def _set_status(self, status: str) -> None:
        try:
            self.query_one("#clone-status-indicator", StatusPanel).status = status
        except Exception:
            pass

    def _on_generation_progress(self, current: int, total: int, phrase: str) -> None:
        self.phrases_generated = current
        self.target_phrases = total
        self._update_counter()
        self._update_current_phrase(phrase)

    def _on_phrase_generated(self, index: int, phrase: str, audio_path: Path) -> None:
        short_phrase = phrase[:35] + "..." if len(phrase) > 35 else phrase
        self._log(f"[{COLORS['success']}][{index + 1}][/] {short_phrase}")

    def _on_generation_error(self, error: str) -> None:
        self._log(f"[{COLORS['error']}]Erro: {error}[/]")

    def _on_generation_failed(self, error: str) -> None:
        self.is_generating = False
        self._set_sidebar_status(None)
        self._set_status("error")
        self.query_one("#clone-btn-generate", Button).disabled = False
        self.query_one("#clone-btn-stop", Button).disabled = True
        self._log(f"[{COLORS['error']}]Falha: {error}[/]")
        self.app.notify("Falha na geracao", severity="error")

    def _on_generation_stopped(self, stats) -> None:
        self.is_generating = False
        self._set_sidebar_status(None)
        self._set_status("stopped")
        self.query_one("#clone-btn-generate", Button).disabled = False
        self.query_one("#clone-btn-stop", Button).disabled = True
        self._log(
            f"[{COLORS['warning']}]Interrompido: {stats.total_generated}/{stats.total_requested}[/]"
        )
        self.app.notify("Geracao interrompida", severity="warning")

    def _on_generation_complete(self, stats) -> None:
        self.is_generating = False
        self._set_sidebar_status(None)
        self._set_status("completed")
        self.query_one("#clone-btn-generate", Button).disabled = False
        self.query_one("#clone-btn-stop", Button).disabled = True

        self._log("")
        self._log(f"[bold {COLORS['success']}]CONCLUIDO[/]")
        self._log(
            f"[{COLORS['text_primary']}]Gerados: {stats.total_generated}/{stats.total_requested} "
            f"| Taxa: {stats.success_rate:.0f}%[/]"
        )

        if self._dataset_manager:
            output_dir = self._dataset_manager.config.output_dir
            dataset_name = self._dataset_manager.config.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = output_dir / f"{dataset_name}_{timestamp}"
            self._log(f"[{COLORS['neon_cyan']}]{full_path}[/]")

        self.app.notify(
            f"Dataset: {stats.total_generated} amostras", severity="information"
        )

    def _set_dataset_manager(self, manager) -> None:
        self._dataset_manager = manager

    def _set_trainer(self, trainer) -> None:
        self._trainer = trainer

    def _stop_generation(self) -> None:
        if self._dataset_manager:
            self._dataset_manager.stop()
            self._log(f"[{COLORS['warning']}]Parando...[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "clone-btn-generate":
            self._start_generation()
        elif btn_id == "clone-btn-stop":
            self._stop_generation()
        elif btn_id == "clone-btn-browse":
            self._browse_phrases_file()
        elif btn_id == "clone-btn-refresh-datasets":
            self._refresh_datasets()
        elif btn_id == "clone-btn-train-chatterbox":
            self._start_training("chatterbox")
        elif btn_id == "clone-btn-train-coqui":
            self._start_training("coqui")
        elif btn_id == "clone-btn-refresh-models":
            self._refresh_models()
        elif btn_id == "clone-btn-play":
            self._play_test_audio()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "clone-dataset-selector" and event.value != Select.BLANK:
            option_id = str(event.value)
            if option_id in self._datasets_map:
                self._selected_dataset = self._datasets_map[option_id]
                self._log(
                    f"[{COLORS['info']}]Selecionado: {self._selected_dataset.name}[/]"
                )
        elif (
            event.select.id == "clone-model-selector-test"
            and event.value != Select.BLANK
        ):
            option_id = str(event.value)
            if option_id in self._models_map:
                self._selected_model = self._models_map[option_id]
                self._log(f"[{COLORS['info']}]Modelo: {self._selected_model.name}[/]")

    def _browse_phrases_file(self) -> None:
        try:
            result = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--title=Selecionar Arquivo de Frases",
                    "--file-filter=Markdown (*.md) | *.md",
                    "--file-filter=Texto (*.txt) | *.txt",
                    "--file-filter=Todos (*) | *",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._set_phrases_file(Path(result.stdout.strip()))
        except Exception as e:
            self._log(f"[{COLORS['error']}]Erro ao abrir dialogo: {e}[/]")

    def _set_phrases_file(self, filepath: Path) -> None:
        self.query_one("#clone-input-phrases-file", Input).value = str(filepath)
        status_label = self.query_one("#clone-file-status", Label)
        if filepath.exists():
            from src.modules.clone_voice.core.generator import PhraseParser

            parser = PhraseParser()
            if parser.parse_file(filepath):
                stats = parser.get_stats()
                categories = len(parser.get_category_names())
                status_label.update(
                    f"[{COLORS['success']}]{stats['total']} frases, {categories} cats[/]"
                )
                self._log(
                    f"[{COLORS['success']}]{filepath.name}: {stats['total']} frases[/]"
                )
            else:
                status_label.update(f"[{COLORS['error']}]Arquivo vazio[/]")
        else:
            status_label.update(f"[{COLORS['error']}]Nao encontrado[/]")

    def _update_file_status(self) -> None:
        status_label = self.query_one("#clone-file-status", Label)
        current_file = self.query_one("#clone-input-phrases-file", Input).value.strip()
        if current_file:
            filepath = Path(current_file)
            if filepath.exists():
                from src.modules.clone_voice.core.generator import PhraseParser

                parser = PhraseParser()
                if parser.parse_file(filepath):
                    stats = parser.get_stats()
                    status_label.update(
                        f"[{COLORS['success']}]{stats['total']} frases[/]"
                    )
                else:
                    status_label.update(f"[{COLORS['warning']}]Arquivo vazio[/]")
            else:
                status_label.update(f"[{COLORS['text_muted']}]Selecione um arquivo[/]")
        else:
            status_label.update(f"[{COLORS['text_muted']}]Selecione um arquivo[/]")

    def _refresh_datasets(self) -> None:
        output_dir = Path(
            self.query_one("#clone-input-output-dir", Input).value.strip()
        )
        selector = self.query_one("#clone-dataset-selector", Select)

        if not output_dir.exists():
            selector.set_options([])
            return

        datasets = []
        for item in output_dir.iterdir():
            if item.is_dir():
                metadata = item / "metadata.json"
                wavs_dir = item / "wavs"
                if metadata.exists() and wavs_dir.exists():
                    audio_count = len(list(wavs_dir.glob("*.mp3"))) + len(
                        list(wavs_dir.glob("*.wav"))
                    )
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
            options.append((f"{ds_path.name} ({audio_count})", ds_id))
            self._datasets_map[ds_id] = ds_path

        selector.set_options(options)
        self._selected_dataset = datasets[0][0]
        self._log(f"[{COLORS['success']}]Datasets encontrados: {len(datasets)}[/]")

    def _start_training(self, model_type: str) -> None:
        if self.is_training:
            self.app.notify("Treinamento em andamento", severity="error")
            return

        if not self._selected_dataset:
            self._refresh_datasets()
            if not self._selected_dataset:
                self.app.notify("Selecione um dataset", severity="error")
                return

        try:
            epochs = int(self.query_one("#clone-input-epochs", Input).value)
        except ValueError:
            self.app.notify("Epochs invalido", severity="error")
            return

        use_top = self.query_one("#clone-use-top-checkbox", Checkbox).value
        top_n = 10
        try:
            top_n = int(self.query_one("#clone-input-top-n", Input).value)
        except ValueError:
            top_n = 10

        output_dir = self._selected_dataset.parent / "trained_models"

        self.is_training = True
        self.training_progress = 0
        self._set_sidebar_status("TREINANDO")
        self.query_one("#clone-btn-train-chatterbox", Button).disabled = True
        self.query_one("#clone-btn-train-coqui", Button).disabled = True

        self._log("")
        self._log(f"[{COLORS['info']}]Treinar {model_type.upper()}...[/]")
        if use_top:
            self._log(
                f"[{COLORS['text_muted']}]Usando TOP {top_n} audios de {self._selected_dataset.name}[/]"
            )
        else:
            self._log(
                f"[{COLORS['text_muted']}]Usando todos audios de {self._selected_dataset.name}[/]"
            )
        self._log(f"[{COLORS['text_muted']}]Epochs: {epochs}[/]")

        self._training_thread = threading.Thread(
            target=run_training_thread,
            args=(
                self,
                model_type,
                self._selected_dataset,
                output_dir,
                epochs,
                use_top,
                top_n,
            ),
            daemon=True,
        )
        self._training_thread.start()

    def _update_training_status(self, status: str) -> None:
        self._log(f"[{COLORS['warning']}]{status}[/]")

    def _on_training_progress(self, step: int, total: int, loss: float) -> None:
        try:
            progress = self.query_one("#clone-training-progress", ProgressBar)
            pct = (step / total) * 100 if total > 0 else 0
            progress.update(progress=pct)
            self.training_progress = int(pct)
        except Exception:
            pass

    def _on_epoch_complete(self, epoch: int, loss: float) -> None:
        self._log(f"[{COLORS['info']}]Epoch {epoch} | Loss: {loss:.4f}[/]")

    def _on_training_complete(self, stats) -> None:
        self.is_training = False
        self._set_sidebar_status(None)
        self.query_one("#clone-btn-train-chatterbox", Button).disabled = False
        self.query_one("#clone-btn-train-coqui", Button).disabled = False

        if stats.is_completed:
            self._log("")
            self._log(f"[bold {COLORS['success']}]TREINAMENTO OK[/]")
            self._log(
                f"[{COLORS['text_muted']}]Epochs: {stats.current_epoch} | Loss: {stats.best_loss:.4f}[/]"
            )
            if stats.output_path:
                self._last_output_dir = stats.output_path
                self._log(f"[{COLORS['neon_cyan']}]{stats.output_path}[/]")
                self.app.push_screen(
                    TrainingCompleteModal(stats.output_path, stats.model_type)
                )
            self.app.notify("Treinamento concluido", severity="information")
        else:
            self._log(f"[{COLORS['warning']}]Treinamento interrompido[/]")

    def _on_training_error(self, error: str) -> None:
        self.is_training = False
        self._set_sidebar_status(None)
        self.query_one("#clone-btn-train-chatterbox", Button).disabled = False
        self.query_one("#clone-btn-train-coqui", Button).disabled = False
        self._log(f"[{COLORS['error']}]Erro: {error}[/]")
        self.app.notify("Erro no treinamento", severity="error")

    def action_start_training(self) -> None:
        if not self.is_training:
            self._start_training("coqui")

    def _refresh_models(self) -> None:
        try:
            model_list = self.query_one("#clone-model-selector-test", Select)
            self._models_map.clear()

            trained_models_dir = DEFAULT_OUTPUT_DIR / "trained_models"
            if not trained_models_dir.exists():
                model_list.set_options([])
                return

            models = [
                item
                for item in trained_models_dir.iterdir()
                if item.is_dir()
                and (item.name.endswith("_chatterbox") or item.name.endswith("_coqui"))
            ]

            if not models:
                model_list.set_options([])
                self._selected_model = None
                return

            models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            options = []
            for model_path in models:
                self._models_map[model_path.name] = model_path
                options.append((model_path.name, model_path.name))

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
                self.app.notify("Selecione um modelo", severity="error")
                return

        text = self.query_one("#clone-input-test-text", TextArea).text.strip()
        if not text:
            self.app.notify("Digite um texto", severity="error")
            return

        self._is_playing = True
        self.query_one("#clone-btn-play", Button).disabled = True

        self._log("")
        self._log(
            f"[{COLORS['info']}]Gerando audio com {self._selected_model.name}...[/]"
        )

        threading.Thread(
            target=self._run_inference, args=(self._selected_model, text), daemon=True
        ).start()

    def _run_inference(self, model_path: Path, text: str) -> None:
        try:
            output_file = model_path / "outputs" / "test_output.wav"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if model_path.name.endswith("_chatterbox"):
                run_chatterbox_inference(model_path, text, output_file, ROOT_DIR)
            else:
                run_coqui_inference(model_path, text, output_file, ROOT_DIR)

            if output_file.exists():
                self.call_from_thread(
                    self._log, f"[{COLORS['success']}]Audio gerado![/]"
                )
                self.call_from_thread(
                    self._log, f"[{COLORS['text_muted']}]Reproduzindo...[/]"
                )
                self._play_audio_file(output_file)
                self.call_from_thread(
                    self._log, f"[{COLORS['info']}]Reproducao concluida[/]"
                )
            else:
                self.call_from_thread(
                    self._log, f"[{COLORS['error']}]Falha ao gerar audio[/]"
                )

        except Exception as e:
            self.call_from_thread(self._log, f"[{COLORS['error']}]Erro: {e}[/]")
        finally:
            self._is_playing = False
            self.call_from_thread(self._enable_play_button)

    def _play_audio_file(self, audio_path: Path) -> None:
        import sounddevice as sd
        from scipy.io import wavfile

        sample_rate, audio_data = wavfile.read(str(audio_path))
        if audio_data.dtype == "int16":
            audio_data = audio_data.astype("float32") / 32768.0
        elif audio_data.dtype == "int32":
            audio_data = audio_data.astype("float32") / 2147483648.0
        sd.play(audio_data, sample_rate)
        sd.wait()

    def _enable_play_button(self) -> None:
        try:
            self.query_one("#clone-btn-play", Button).disabled = False
        except Exception:
            pass


# "O melhor momento para plantar uma árvore foi há 20 anos. O segundo melhor momento é agora." — provérbio
