# -*- coding: utf-8 -*-

import random
import os
import logging
import threading
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import (
    Header, Footer, Button, Label, Input,
    Static, ProgressBar, Switch, RadioSet, RadioButton
)
from textual.containers import Vertical, Horizontal, Container
from textual.reactive import reactive

from src.core.base_app import NeurosonancyBaseApp
from src.core.theme import CSS_COMMON, COLORS
from src.modules.voice_trainer.core import config
from src.modules.voice_trainer.core.audio_recorder import AudioRecorder
from src.modules.voice_trainer.core.audio_comparator import AudioComparator
from src.modules.voice_trainer.ui.visualizer import AudioVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatusIndicator(Static):
    is_recording = reactive(False)

    def render(self) -> str:
        if self.is_recording:
            return f"[bold {COLORS['error']}]GRAVANDO[/]"
        return f"[{COLORS['text_muted']}]Pronto[/]"


class PhraseDisplay(Static):
    current_phrase = reactive("")
    comparison_result = reactive("")

    def render(self) -> str:
        if self.comparison_result:
            return self.comparison_result
        elif self.current_phrase:
            return f'[italic {COLORS["text_primary"]}]"{self.current_phrase}"[/]'
        return f"[{COLORS['text_muted']}]Clique em SORTEAR para ver uma frase[/]"


class SampleCounter(Static):
    recorded = reactive(0)
    required = reactive(config.TRAINING_CONFIG["TARGET_SAMPLES"])

    def render(self) -> str:
        color = COLORS['success'] if self.recorded >= config.TRAINING_CONFIG["MIN_SAMPLES"] else COLORS['warning']
        return f"[{color}]Amostras: [bold]{self.recorded}[/bold]/{self.required}[/]"


class VoiceTrainerApp(NeurosonancyBaseApp):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = "Trainer"

    BINDINGS = [
        Binding("escape", "back_to_menu", "Menu", show=True, priority=True),
        Binding("q", "quit", "Sair", show=True),
        Binding("r", "shuffle_phrase", "Sortear", show=True),
        Binding("space", "toggle_recording", "Gravar", show=True),
    ]

    CSS = CSS_COMMON + """
    #main-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
        padding: 1;
    }

    #left-panel {
        width: 60%;
        height: 100%;
        background: #1a1d2e;
        border: solid #4ade80;
        padding: 1 2;
        margin-right: 1;
    }

    #right-panel {
        width: 40%;
        height: 100%;
        background: #1a1d2e;
        border: solid #8b5cf6;
        padding: 1 2;
    }

    .section-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: #4ade80;
        padding: 0 0 1 0;
        border-bottom: solid #2d3250;
        margin-bottom: 1;
    }

    #right-panel .section-title {
        color: #8b5cf6;
    }

    #phrase-display {
        width: 100%;
        min-height: 5;
        height: auto;
        background: #141620;
        border: solid #2d3250;
        padding: 1;
        content-align: center middle;
        text-align: center;
        margin-bottom: 1;
    }

    #status-row {
        width: 100%;
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    #audio-visualizer {
        width: 100%;
        height: 2;
        background: #141620;
        border: solid #2d3250;
        margin-bottom: 1;
    }

    #recording-progress > .bar--complete {
        color: #4ade80;
    }

    #main-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    #main-buttons > Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        min-width: 10;
    }

    #btn-shuffle {
        background: #8b5cf6;
        color: #0d0f18;
    }

    #btn-record {
        background: #22c55e;
        color: #0d0f18;
    }

    #btn-play {
        background: #06b6d4;
        color: #0d0f18;
    }

    #btn-discard {
        background: #ef4444;
        color: #0d0f18;
    }

    #btn-save-sample {
        width: 100%;
        height: 3;
        background: #4ade80;
        color: #0d0f18;
        text-style: bold;
        margin-top: 1;
    }

    #samples-section {
        width: 100%;
        height: auto;
        border-top: solid #2d3250;
        padding-top: 1;
        margin-top: 1;
    }

    #samples-list {
        width: 100%;
        height: auto;
        max-height: 6;
        color: #94a3b8;
        padding: 1;
        text-align: left;
    }

    .setting-label {
        width: 100%;
        color: #e2e8f0;
        padding: 1 0 0 0;
    }

    .setting-item {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #right-panel Input {
        width: 100%;
        height: 3;
        background: #141620;
        border: solid #2d3250;
        padding: 0 1;
        margin-top: 0;
    }

    #right-panel Input:focus {
        border: solid #8b5cf6;
    }

    RadioSet {
        width: 100%;
        background: #141620;
        border: solid #2d3250;
        padding: 1;
    }

    RadioButton.-selected {
        color: #8b5cf6;
    }
    """

    recorded_samples: reactive[List[str]] = reactive(list)
    current_phrase_index: reactive[int] = reactive(-1)
    is_recording: reactive[bool] = reactive(False)
    last_recording_path: Optional[str] = None
    
    audio_recorder: Optional[AudioRecorder] = None
    audio_comparator: Optional[AudioComparator] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        
        with Container(id="main-container"):
            with Vertical(id="left-panel"):
                yield Label("GRAVACAO", classes="section-title")
                
                with Vertical(id="phrase-container"):
                    yield PhraseDisplay(id="phrase-display")
                    
                    with Horizontal(id="status-row"):
                        yield StatusIndicator(id="status-indicator")
                        yield SampleCounter(id="sample-counter")
                
                yield AudioVisualizer(id="audio-visualizer")
                
                with Vertical(id="progress-container"):
                    yield ProgressBar(id="recording-progress", total=100, show_eta=False)
                
                with Horizontal(id="main-buttons"):
                    yield Button("SORTEAR", id="btn-shuffle", variant="primary")
                    yield Button("GRAVAR", id="btn-record", variant="success")
                    yield Button("OUVIR", id="btn-play", disabled=True)
                    yield Button("DESCARTAR", id="btn-discard", variant="warning", disabled=True)
                
                with Vertical(id="save-button-container"):
                    yield Button("SALVAR AMOSTRA", id="btn-save-sample", variant="primary", disabled=True)
                
                with Vertical(id="samples-section"):
                    yield Label("Amostras Salvas", classes="section-title")
                    yield Static("Nenhuma amostra ainda", id="samples-list")
            
            with Vertical(id="right-panel"):
                yield Label("CONFIGURACOES", classes="section-title")
                
                with Vertical(classes="setting-item"):
                    yield Label("Nivel de Dificuldade:", classes="setting-label")
                    with RadioSet(id="level-selector"):
                        yield RadioButton("Baseline (40 frases)", id="level-baseline", value=True)
                        yield RadioButton("Facil (20 frases)", id="level-low")
                        yield RadioButton("Medio (40 frases)", id="level-medium")
                        yield RadioButton("Dificil (60 frases)", id="level-high")
                
                with Vertical(classes="setting-item"):
                    yield Label("Nome do Modelo:", classes="setting-label")
                    yield Input(
                        placeholder="Ex: Voz_Andre_V1",
                        id="input-model-name",
                        value=f"Voz_{datetime.now().strftime('%Y%m%d')}"
                    )
                
                with Vertical(classes="setting-item"):
                    yield Label("Meta de Amostras:", classes="setting-label")
                    yield Input(
                        placeholder="10",
                        id="input-target-samples",
                        value=str(config.TRAINING_CONFIG["TARGET_SAMPLES"]),
                        type="integer"
                    )
        
        yield Footer()

    def on_mount(self) -> None:
        self.recorded_samples = []
        self.current_level = "baseline"
        self.action_shuffle_phrase()
        self._update_samples_list()
        
        self.audio_recorder = AudioRecorder(
            sample_rate=config.AUDIO_CONFIG["SAMPLE_RATE"],
            channels=config.AUDIO_CONFIG["CHANNELS"],
            device_id=config.AUDIO_CONFIG.get("DEVICE_ID"),
            vad_silence_duration=config.VAD_CONFIG["SILENCE_DURATION"],
            vad_energy_threshold=config.VAD_CONFIG["ENERGY_THRESHOLD"]
        )
        
        visualizer = self.query_one("#audio-visualizer", AudioVisualizer)
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
        else:
            return config.PHRASES_BASELINE

    def action_shuffle_phrase(self) -> None:
        phrases = self._get_current_phrases()
        available_indices = [
            i for i in range(len(phrases)) 
            if i != self.current_phrase_index
        ]
        if available_indices:
            self.current_phrase_index = random.choice(available_indices)
            phrase_display = self.query_one("#phrase-display", PhraseDisplay)
            phrase_display.current_phrase = phrases[self.current_phrase_index]
            phrase_display.comparison_result = ""

    def action_toggle_recording(self) -> None:
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _handle_auto_stop(self) -> None:
        self.call_from_thread(self._auto_save_sample)

    def _start_recording(self) -> None:
        if self.current_phrase_index < 0:
            self.notify("Sorteie uma frase primeiro!", severity="warning")
            return
        
        self.is_recording = True
        indicator = self.query_one("#status-indicator", StatusIndicator)
        indicator.is_recording = True
        
        visualizer = self.query_one("#audio-visualizer", AudioVisualizer)
        visualizer.set_recording(True)
        
        self.query_one("#btn-record", Button).disabled = True
        self.query_one("#btn-shuffle", Button).disabled = True
        
        progress = self.query_one("#recording-progress", ProgressBar)
        progress.update(progress=0)
        
        self.last_recording_path = self._get_temp_recording_path()
        
        logger.info(f"Iniciando gravacao: {self.last_recording_path}")
        
        if not self.audio_recorder.start_recording(self.last_recording_path):
            self.notify("Erro ao iniciar gravacao!", severity="error")
            self._reset_recording_state()
            return
        
        self._animate_progress()
        
        self.notify("Gravando... pare de falar quando terminar.", severity="information")

    def _stop_recording(self) -> None:
        self.is_recording = False
        
        saved_path = self.audio_recorder.stop_recording()
        
        indicator = self.query_one("#status-indicator", StatusIndicator)
        indicator.is_recording = False
        
        visualizer = self.query_one("#audio-visualizer", AudioVisualizer)
        visualizer.set_recording(False)
        
        self.query_one("#btn-record", Button).disabled = False
        self.query_one("#btn-shuffle", Button).disabled = False
        
        if not saved_path:
            self.notify("Erro ao salvar gravacao!", severity="error")
            self._reset_recording_state()
            return
        
        self.last_recording_path = saved_path
        self.query_one("#btn-play", Button).disabled = False
        self.query_one("#btn-discard", Button).disabled = False
        self.query_one("#btn-save-sample", Button).disabled = False
        
        logger.info("Gravacao finalizada")
        
        self._run_comparison_async()
        
        self.notify("Gravacao finalizada. Analisando...", severity="information")
    
    def _reset_recording_state(self) -> None:
        self.is_recording = False
        self.last_recording_path = None
        
        indicator = self.query_one("#status-indicator", StatusIndicator)
        indicator.is_recording = False
        
        visualizer = self.query_one("#audio-visualizer", AudioVisualizer)
        visualizer.set_recording(False)
        
        self.query_one("#btn-record", Button).disabled = False
    
    def _auto_save_sample(self) -> None:
        saved_path = self.audio_recorder.stop_recording()
        
        indicator = self.query_one("#status-indicator", StatusIndicator)
        indicator.is_recording = False
        
        visualizer = self.query_one("#audio-visualizer", AudioVisualizer)
        visualizer.set_recording(False)
        
        self.is_recording = False
        
        if not saved_path:
            self.notify("Erro ao salvar gravacao!", severity="error")
            self._reset_recording_state()
            return
        
        self.last_recording_path = saved_path
        
        logger.info("Auto-salvando amostra após VAD")
        
        self._run_comparison_async()
        self.recorded_samples = self.recorded_samples + [self.last_recording_path]
        self.last_recording_path = None
        self._update_samples_list()
        self.action_shuffle_phrase()
        self.query_one("#btn-record", Button).disabled = False
        self.query_one("#btn-shuffle", Button).disabled = False
        
        target = config.TRAINING_CONFIG["TARGET_SAMPLES"]
        self.notify(f"Amostra salva automaticamente! ({len(self.recorded_samples)}/{target})", severity="information")
    
    def _run_comparison_async(self) -> None:
        def compare():
            try:
                expected_phrase = config.TRAINING_PHRASES[self.current_phrase_index]
                transcribed = self.audio_comparator.transcribe_audio(self.last_recording_path)
                
                if not transcribed:
                    self.call_from_thread(self.notify, "Erro na transcricao!", severity="error")
                    return
                
                comparisons, score = self.audio_comparator.compare_texts(expected_phrase, transcribed)
                formatted = self.audio_comparator.format_comparison(comparisons, score)
                
                phrase_display = self.query_one("#phrase-display", PhraseDisplay)
                self.call_from_thread(setattr, phrase_display, "comparison_result", formatted)
                
            except Exception as e:
                logger.error(f"Comparison error: {e}", exc_info=True)
                self.call_from_thread(self.notify, f"Erro na comparacao: {e}", severity="error")
        
        thread = threading.Thread(target=compare, daemon=True)
        thread.start()

    def _animate_progress(self) -> None:
        progress = self.query_one("#recording-progress", ProgressBar)
        
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
        samples_list = self.query_one("#samples-list", Static)
        counter = self.query_one("#sample-counter", SampleCounter)
        
        counter.recorded = len(self.recorded_samples)
        
        if self.recorded_samples:
            items = [f"  {i+1}. {Path(s).name}" for i, s in enumerate(self.recorded_samples)]
            samples_list.update("\n".join(items[-5:]))
            if len(self.recorded_samples) > 5:
                samples_list.update(f"  ... e mais {len(self.recorded_samples) - 5}")
        else:
            samples_list.update("Nenhuma amostra ainda")
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "btn-shuffle":
            self.action_shuffle_phrase()
        
        elif button_id == "btn-record":
            self._start_recording()
        
        elif button_id == "btn-play":
            if self.last_recording_path:
                logger.info(f"Reproduzindo: {self.last_recording_path}")
                self.notify("Reproduzindo amostra...", severity="information")
                
                def play():
                    try:
                        self.audio_recorder.play_audio(self.last_recording_path)
                    except Exception as e:
                        logger.error(f"Playback error: {e}")
                        self.call_from_thread(self.notify, f"Erro ao reproduzir: {e}", severity="error")
                
                thread = threading.Thread(target=play, daemon=True)
                thread.start()
        
        elif button_id == "btn-discard":
            if self.last_recording_path and os.path.exists(self.last_recording_path):
                os.remove(self.last_recording_path)
            self.last_recording_path = None
            self.query_one("#btn-play", Button).disabled = True
            self.query_one("#btn-discard", Button).disabled = True
            self.query_one("#btn-save-sample", Button).disabled = True
            self.notify("Amostra descartada.", severity="warning")
        
        elif button_id == "btn-save-sample":
            if self.last_recording_path and os.path.exists(self.last_recording_path):
                self.recorded_samples = self.recorded_samples + [self.last_recording_path]
                self.last_recording_path = None
                self._update_samples_list()
                self.action_shuffle_phrase()
                self.query_one("#btn-play", Button).disabled = True
                self.query_one("#btn-discard", Button).disabled = True
                self.query_one("#btn-save-sample", Button).disabled = True
                target = config.TRAINING_CONFIG["TARGET_SAMPLES"]
                self.notify(f"Amostra salva! ({len(self.recorded_samples)}/{target})", severity="information")
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        button_id = event.pressed.id
        
        if button_id == "level-baseline":
            self.current_level = "baseline"
        elif button_id == "level-low":
            self.current_level = "low"
        elif button_id == "level-medium":
            self.current_level = "medium"
        elif button_id == "level-high":
            self.current_level = "high"
        
        self.current_phrase_index = -1
        self.action_shuffle_phrase()
        self.notify(f"Nivel alterado para: {self.current_level}", severity="information")


if __name__ == "__main__":
    app = VoiceTrainerApp()
    app.run()
