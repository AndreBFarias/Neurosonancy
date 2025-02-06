# -*- coding: utf-8 -*-

import os
import random
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Input, Static, ProgressBar, RadioSet, RadioButton
from textual.containers import Vertical, Horizontal, Container

from src.core.theme import COLORS
from src.modules.voice_trainer.core import config
from src.modules.voice_trainer.ui.visualizer import AudioVisualizer

if TYPE_CHECKING:
    from src.modules.voice_trainer.core.audio_recorder import AudioRecorder
    from src.modules.voice_trainer.core.audio_comparator import AudioComparator

logger = logging.getLogger(__name__)


class StatusIndicator(Static):
    is_recording: reactive[bool] = reactive(False)

    def render(self) -> str:
        if self.is_recording:
            return f"[bold {COLORS['error']}]GRAVANDO[/]"
        return f"[{COLORS['text_muted']}]Pronto[/]"


class PhraseDisplay(Static):
    current_phrase: reactive[str] = reactive("")
    comparison_result: reactive[str] = reactive("")

    def render(self) -> str:
        if self.comparison_result:
            return self.comparison_result
        elif self.current_phrase:
            return f'[italic {COLORS["text_primary"]}]"{self.current_phrase}"[/]'
        return f"[{COLORS['text_muted']}]Clique em SORTEAR para ver uma frase[/]"


class SampleCounter(Static):
    recorded: reactive[int] = reactive(0)
    required: reactive[int] = reactive(config.TRAINING_CONFIG["TARGET_SAMPLES"])

    def render(self) -> str:
        color = COLORS['success'] if self.recorded >= config.TRAINING_CONFIG["MIN_SAMPLES"] else COLORS['warning']
        return f"[{color}]Amostras: [bold]{self.recorded}[/bold]/{self.required}[/]"


class VoiceTrainerPanel(Widget):

    BINDINGS = [
        Binding("r", "shuffle_phrase", "Sortear", show=True),
        Binding("space", "toggle_recording", "Gravar", show=True),
    ]

    DEFAULT_CSS = f"""
    VoiceTrainerPanel {{
        height: 100%;
        width: 100%;
    }}

    VoiceTrainerPanel #trainer-main-container {{
        width: 100%;
        height: 100%;
        layout: horizontal;
        padding: 1;
    }}

    VoiceTrainerPanel #trainer-left-panel {{
        width: 60%;
        height: 100%;
        background: {COLORS['bg_surface']};
        border: solid {COLORS['neon_green']};
        border-title-color: {COLORS['neon_green']};
        padding: 1 2;
        margin-right: 1;
    }}

    VoiceTrainerPanel #trainer-right-panel {{
        width: 40%;
        height: 100%;
        background: {COLORS['bg_surface']};
        border: solid {COLORS['accent_primary']};
        border-title-color: {COLORS['accent_primary']};
        padding: 1 2;
    }}

    VoiceTrainerPanel .trainer-section-title {{
        width: 100%;
        text-align: center;
        text-style: bold;
        color: {COLORS['neon_green']};
        padding: 0 0 1 0;
        border-bottom: solid {COLORS['border_normal']};
        margin-bottom: 1;
    }}

    VoiceTrainerPanel #trainer-right-panel .trainer-section-title {{
        color: {COLORS['accent_primary']};
    }}

    VoiceTrainerPanel #trainer-phrase-container {{
        height: 1fr;
    }}

    VoiceTrainerPanel #trainer-phrase-display {{
        width: 100%;
        min-height: 5;
        height: 1fr;
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['neon_green']};
        padding: 2;
        content-align: center middle;
        text-align: center;
        margin-bottom: 1;
    }}

    VoiceTrainerPanel #trainer-status-row {{
        width: 100%;
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }}

    VoiceTrainerPanel #trainer-audio-visualizer {{
        width: 100%;
        height: 4;
        background: {COLORS['bg_dark']};
        border: solid {COLORS['neon_green']};
        margin-bottom: 1;
    }}

    VoiceTrainerPanel #trainer-progress-container {{
        height: auto;
    }}

    VoiceTrainerPanel #trainer-save-button-container {{
        height: auto;
    }}

    VoiceTrainerPanel #trainer-recording-progress > .bar--complete {{
        color: {COLORS['neon_green']};
    }}

    VoiceTrainerPanel #trainer-main-buttons {{
        width: 100%;
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }}

    VoiceTrainerPanel #trainer-main-buttons > Button {{
        width: 1fr;
        height: 3;
        margin: 0 1;
        min-width: 10;
    }}

    VoiceTrainerPanel #trainer-btn-shuffle {{
        background: {COLORS['accent_primary']};
        color: {COLORS['bg_darkest']};
    }}

    VoiceTrainerPanel #trainer-btn-record {{
        background: {COLORS['success']};
        color: {COLORS['bg_darkest']};
    }}

    VoiceTrainerPanel #trainer-btn-play {{
        background: {COLORS['info']};
        color: {COLORS['bg_darkest']};
    }}

    VoiceTrainerPanel #trainer-btn-discard {{
        background: {COLORS['error']};
        color: {COLORS['bg_darkest']};
    }}

    VoiceTrainerPanel #trainer-btn-save-sample {{
        width: 100%;
        height: 3;
        background: {COLORS['neon_green']};
        color: {COLORS['bg_darkest']};
        text-style: bold;
        margin-top: 1;
    }}

    VoiceTrainerPanel #trainer-samples-section {{
        width: 100%;
        height: auto;
        border-top: solid {COLORS['border_normal']};
        padding-top: 1;
        margin-top: 1;
    }}

    VoiceTrainerPanel #trainer-samples-list {{
        width: 100%;
        height: auto;
        max-height: 6;
        color: {COLORS['text_secondary']};
        padding: 1;
        text-align: left;
    }}

    VoiceTrainerPanel .trainer-setting-label {{
        width: 100%;
        color: {COLORS['text_primary']};
        padding: 1 0 0 0;
    }}

    VoiceTrainerPanel .trainer-setting-item {{
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }}

    VoiceTrainerPanel RadioSet {{
        width: 100%;
        background: {COLORS['bg_dark']};
        border: solid {COLORS['border_dim']};
        padding: 1;
    }}

    VoiceTrainerPanel RadioButton.-selected {{
        color: {COLORS['accent_primary']};
    }}
    """

    recorded_samples: reactive[List[str]] = reactive(list)
    current_phrase_index: reactive[int] = reactive(-1)
    is_recording: reactive[bool] = reactive(False)
    last_recording_path: Optional[str] = None

    audio_recorder: Optional["AudioRecorder"] = None
    audio_comparator: Optional["AudioComparator"] = None

    def compose(self) -> ComposeResult:
        with Container(id="trainer-main-container"):
            with Vertical(id="trainer-left-panel"):
                yield Label("GRAVACAO", classes="trainer-section-title")

                with Vertical(id="trainer-phrase-container"):
                    yield PhraseDisplay(id="trainer-phrase-display")

                    with Horizontal(id="trainer-status-row"):
                        yield StatusIndicator(id="trainer-status-indicator")
                        yield SampleCounter(id="trainer-sample-counter")

                yield AudioVisualizer(id="trainer-audio-visualizer")

                with Vertical(id="trainer-progress-container"):
                    yield ProgressBar(id="trainer-recording-progress", total=100, show_eta=False)

                with Horizontal(id="trainer-main-buttons"):
                    yield Button("SORTEAR", id="trainer-btn-shuffle", variant="primary")
                    yield Button("GRAVAR", id="trainer-btn-record", variant="success")
                    yield Button("OUVIR", id="trainer-btn-play", disabled=True)
                    yield Button("DESCARTAR", id="trainer-btn-discard", variant="warning", disabled=True)

                with Vertical(id="trainer-save-button-container"):
                    yield Button("SALVAR AMOSTRA", id="trainer-btn-save-sample", variant="primary", disabled=True)

                with Vertical(id="trainer-samples-section"):
                    yield Label("Amostras Salvas", classes="trainer-section-title")
                    yield Static("Nenhuma amostra ainda", id="trainer-samples-list")

            with Vertical(id="trainer-right-panel"):
                yield Label("CONFIGURACOES", classes="trainer-section-title")

                with Vertical(classes="trainer-setting-item"):
                    yield Label("Nivel de Dificuldade:", classes="trainer-setting-label")
                    with RadioSet(id="trainer-level-selector"):
                        yield RadioButton("Baseline (40 frases)", id="trainer-level-baseline", value=True)
                        yield RadioButton("Facil (20 frases)", id="trainer-level-low")
                        yield RadioButton("Medio (40 frases)", id="trainer-level-medium")
                        yield RadioButton("Dificil (60 frases)", id="trainer-level-high")

                with Vertical(classes="trainer-setting-item"):
                    yield Label("Nome do Modelo:", classes="trainer-setting-label")
                    yield Input(
                        placeholder="Ex: Voz_Andre_V1",
                        id="trainer-input-model-name",
                        value=f"Voz_{datetime.now().strftime('%Y%m%d')}"
                    )

                with Vertical(classes="trainer-setting-item"):
                    yield Label("Meta de Amostras:", classes="trainer-setting-label")
                    yield Input(
                        placeholder="10",
                        id="trainer-input-target-samples",
                        value=str(config.TRAINING_CONFIG["TARGET_SAMPLES"]),
                        type="integer"
                    )

    def on_mount(self) -> None:
        self.recorded_samples = []
        self.current_level = "baseline"
        self.action_shuffle_phrase()
        self._update_samples_list()

        from src.modules.voice_trainer.core.audio_recorder import AudioRecorder
        from src.modules.voice_trainer.core.audio_comparator import AudioComparator

        self.audio_recorder = AudioRecorder(
            sample_rate=config.AUDIO_CONFIG["SAMPLE_RATE"],
            channels=config.AUDIO_CONFIG["CHANNELS"],
            device_id=config.AUDIO_CONFIG.get("DEVICE_ID"),
            vad_silence_duration=config.VAD_CONFIG["SILENCE_DURATION"],
            vad_energy_threshold=config.VAD_CONFIG["ENERGY_THRESHOLD"]
        )

        visualizer = self.query_one("#trainer-audio-visualizer", AudioVisualizer)
        self.audio_recorder.set_visualization_callback(visualizer.update_audio)
        self.audio_recorder.set_auto_stop_callback(self._handle_auto_stop)

        self.audio_comparator = AudioComparator(
            whisper_model_size=config.WHISPER_CONFIG["MODEL_SIZE"],
            device=config.WHISPER_CONFIG["DEVICE"],
            compute_type=config.WHISPER_CONFIG["COMPUTE_TYPE"]
        )

    def _get_current_phrases(self) -> list:
        if self.current_level == "low":
            return config.PHRASES_LEVEL_LOW
        elif self.current_level == "medium":
            return config.PHRASES_LEVEL_MEDIUM
        elif self.current_level == "high":
            return config.PHRASES_LEVEL_HIGH
        return config.PHRASES_BASELINE

    def action_shuffle_phrase(self) -> None:
        phrases = self._get_current_phrases()
        available_indices = [i for i in range(len(phrases)) if i != self.current_phrase_index]
        if available_indices:
            self.current_phrase_index = random.choice(available_indices)
            phrase_display = self.query_one("#trainer-phrase-display", PhraseDisplay)
            phrase_display.current_phrase = phrases[self.current_phrase_index]
            phrase_display.comparison_result = ""

    def action_toggle_recording(self) -> None:
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _handle_auto_stop(self) -> None:
        self.call_from_thread(self._auto_save_sample)

    def _set_sidebar_status(self, status: str | None) -> None:
        try:
            from src.core.widgets.nav_sidebar import NavSidebar
            self.app.query_one(NavSidebar).set_module_status("trainer", status)
        except Exception:
            pass

    def _start_recording(self) -> None:
        if self.current_phrase_index < 0:
            self.app.notify("Sorteie uma frase primeiro!", severity="warning")
            return

        self._set_sidebar_status("REC")
        self.is_recording = True
        self.query_one("#trainer-status-indicator", StatusIndicator).is_recording = True

        visualizer = self.query_one("#trainer-audio-visualizer", AudioVisualizer)
        visualizer.set_recording(True)

        self.query_one("#trainer-btn-record", Button).disabled = True
        self.query_one("#trainer-btn-shuffle", Button).disabled = True

        self.query_one("#trainer-recording-progress", ProgressBar).update(progress=0)
        self.last_recording_path = self._get_temp_recording_path()

        logger.info("Iniciando gravacao: %s", self.last_recording_path)

        if not self.audio_recorder.start_recording(self.last_recording_path):
            self.app.notify("Erro ao iniciar gravacao!", severity="error")
            self._reset_recording_state()
            return

        self._animate_progress()
        self.app.notify("Gravando... pare de falar quando terminar.", severity="information")

    def _stop_recording(self) -> None:
        self._set_sidebar_status(None)
        self.is_recording = False
        saved_path = self.audio_recorder.stop_recording()

        self.query_one("#trainer-status-indicator", StatusIndicator).is_recording = False
        self.query_one("#trainer-audio-visualizer", AudioVisualizer).set_recording(False)
        self.query_one("#trainer-btn-record", Button).disabled = False
        self.query_one("#trainer-btn-shuffle", Button).disabled = False

        if not saved_path:
            self.app.notify("Erro ao salvar gravacao!", severity="error")
            self._reset_recording_state()
            return

        self.last_recording_path = saved_path
        self.query_one("#trainer-btn-play", Button).disabled = False
        self.query_one("#trainer-btn-discard", Button).disabled = False
        self.query_one("#trainer-btn-save-sample", Button).disabled = False

        logger.info("Gravacao finalizada")
        self._run_comparison_async()
        self.app.notify("Gravacao finalizada. Analisando...", severity="information")

    def _reset_recording_state(self) -> None:
        self._set_sidebar_status(None)
        self.is_recording = False
        self.last_recording_path = None
        self.query_one("#trainer-status-indicator", StatusIndicator).is_recording = False
        self.query_one("#trainer-audio-visualizer", AudioVisualizer).set_recording(False)
        self.query_one("#trainer-btn-record", Button).disabled = False

    def _auto_save_sample(self) -> None:
        self._set_sidebar_status(None)
        saved_path = self.audio_recorder.stop_recording()
        self.query_one("#trainer-status-indicator", StatusIndicator).is_recording = False
        self.query_one("#trainer-audio-visualizer", AudioVisualizer).set_recording(False)
        self.is_recording = False

        if not saved_path:
            self.app.notify("Erro ao salvar gravacao!", severity="error")
            self._reset_recording_state()
            return

        self.last_recording_path = saved_path
        logger.info("Auto-salvando amostra apos VAD")
        self._run_comparison_async()
        self.recorded_samples = self.recorded_samples + [self.last_recording_path]
        self.last_recording_path = None
        self._update_samples_list()
        self.action_shuffle_phrase()
        self.query_one("#trainer-btn-record", Button).disabled = False
        self.query_one("#trainer-btn-shuffle", Button).disabled = False

        target = config.TRAINING_CONFIG["TARGET_SAMPLES"]
        self.app.notify(
            f"Amostra salva automaticamente! ({len(self.recorded_samples)}/{target})",
            severity="information"
        )

    def _run_comparison_async(self) -> None:
        def compare():
            try:
                phrases = self._get_current_phrases()
                expected_phrase = phrases[self.current_phrase_index]
                transcribed = self.audio_comparator.transcribe_audio(self.last_recording_path)
                if not transcribed:
                    self.call_from_thread(self.app.notify, "Erro na transcricao!", severity="error")
                    return
                comparisons, score = self.audio_comparator.compare_texts(expected_phrase, transcribed)
                formatted = self.audio_comparator.format_comparison(comparisons, score)
                phrase_display = self.query_one("#trainer-phrase-display", PhraseDisplay)
                self.call_from_thread(setattr, phrase_display, "comparison_result", formatted)
            except Exception as e:
                logger.error("Comparison error: %s", e, exc_info=True)
                self.call_from_thread(self.app.notify, f"Erro na comparacao: {e}", severity="error")

        threading.Thread(target=compare, daemon=True).start()

    def _animate_progress(self) -> None:
        progress = self.query_one("#trainer-recording-progress", ProgressBar)

        def update_progress():
            if self.is_recording and progress.progress < 100:
                progress.advance(2)
                self.set_timer(0.1, update_progress)

        self.set_timer(0.1, update_progress)

    def _get_temp_recording_path(self) -> str:
        config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(config.SAMPLES_DIR / f"sample_{timestamp}.wav")

    def _update_samples_list(self) -> None:
        samples_list = self.query_one("#trainer-samples-list", Static)
        counter = self.query_one("#trainer-sample-counter", SampleCounter)
        counter.recorded = len(self.recorded_samples)

        if self.recorded_samples:
            items = [f"  {i+1}. {Path(s).name}" for i, s in enumerate(self.recorded_samples)]
            samples_list.update("\n".join(items[-5:]))
            if len(self.recorded_samples) > 5:
                samples_list.update(f"  ... e mais {len(self.recorded_samples) - 5}")
        else:
            samples_list.update("Nenhuma amostra ainda")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "trainer-btn-shuffle":
            self.action_shuffle_phrase()

        elif btn_id == "trainer-btn-record":
            self._start_recording()

        elif btn_id == "trainer-btn-play":
            if self.last_recording_path:
                logger.info("Reproduzindo: %s", self.last_recording_path)
                self.app.notify("Reproduzindo amostra...", severity="information")

                def play():
                    try:
                        self.audio_recorder.play_audio(self.last_recording_path)
                    except Exception as e:
                        logger.error("Playback error: %s", e)
                        self.call_from_thread(
                            self.app.notify, f"Erro ao reproduzir: {e}", severity="error"
                        )

                threading.Thread(target=play, daemon=True).start()

        elif btn_id == "trainer-btn-discard":
            if self.last_recording_path and os.path.exists(self.last_recording_path):
                os.remove(self.last_recording_path)
            self.last_recording_path = None
            self.query_one("#trainer-btn-play", Button).disabled = True
            self.query_one("#trainer-btn-discard", Button).disabled = True
            self.query_one("#trainer-btn-save-sample", Button).disabled = True
            self.app.notify("Amostra descartada.", severity="warning")

        elif btn_id == "trainer-btn-save-sample":
            if self.last_recording_path and os.path.exists(self.last_recording_path):
                self.recorded_samples = self.recorded_samples + [self.last_recording_path]
                self.last_recording_path = None
                self._update_samples_list()
                self.action_shuffle_phrase()
                self.query_one("#trainer-btn-play", Button).disabled = True
                self.query_one("#trainer-btn-discard", Button).disabled = True
                self.query_one("#trainer-btn-save-sample", Button).disabled = True
                target = config.TRAINING_CONFIG["TARGET_SAMPLES"]
                self.app.notify(
                    f"Amostra salva! ({len(self.recorded_samples)}/{target})",
                    severity="information"
                )

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        btn_id = event.pressed.id
        level_map = {
            "trainer-level-baseline": "baseline",
            "trainer-level-low": "low",
            "trainer-level-medium": "medium",
            "trainer-level-high": "high",
        }
        if btn_id in level_map:
            self.current_level = level_map[btn_id]
            self.current_phrase_index = -1
            self.action_shuffle_phrase()
            self.app.notify(f"Nivel alterado para: {self.current_level}", severity="information")


# "A dor que suportas hoje é a força que carregas amanhã." — Sêneca
