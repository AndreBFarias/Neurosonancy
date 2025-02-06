# -*- coding: utf-8 -*-

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.screen import ModalScreen

if TYPE_CHECKING:
    from src.modules.clone_voice.ui.panel import CloneVoicePanel
from textual.widgets import Static, Button, Label

from src.core.theme import COLORS

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
            logger.warning("Erro ao carregar config: %s", e)
    return {}


def save_config(config: Dict[str, Any]) -> None:
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Erro ao salvar config: %s", e)


class StatusPanel(Static):
    status: reactive[str] = reactive("idle")

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

    #clone-modal-container {
        width: 50;
        height: 14;
        background: #1a1d2e;
        border: solid #22c55e;
        padding: 1 2;
    }

    #clone-modal-title {
        text-align: center;
        text-style: bold;
        color: #22c55e;
        margin-bottom: 1;
    }

    #clone-modal-message {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1;
    }

    #clone-modal-path {
        text-align: center;
        color: #22d3ee;
        text-style: italic;
        margin-bottom: 1;
    }

    #clone-modal-close {
        width: 100%;
        background: #22c55e;
        color: #0d0f18;
        text-style: bold;
    }
    """

    def __init__(self, output_path: Path, model_type: str) -> None:
        super().__init__()
        self.output_path = output_path
        self.model_type = model_type

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        with Vertical(id="clone-modal-container"):
            yield Label("TREINAMENTO CONCLUIDO", id="clone-modal-title")
            yield Label(f"Modelo {self.model_type.upper()} salvo com sucesso", id="clone-modal-message")
            yield Label(str(self.output_path), id="clone-modal-path")
            yield Button("OK", id="clone-modal-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clone-modal-close":
            self.dismiss()


CLONE_CSS = f"""
CloneVoicePanel {{
    height: 100%;
    width: 100%;
}}

CloneVoicePanel #clone-main-container {{
    layout: horizontal;
    height: 100%;
    padding: 0 1;
}}

CloneVoicePanel #clone-left-column {{
    width: 30%;
    height: 100%;
    padding: 0;
}}

CloneVoicePanel #clone-right-column {{
    width: 70%;
    height: 100%;
    padding: 0 0 0 1;
}}

CloneVoicePanel #clone-config-panel {{
    height: 100%;
    background: {COLORS['bg_elevated']};
    border: solid {COLORS['neon_pink']};
    border-title-color: {COLORS['neon_pink']};
    padding: 1 2;
    scrollbar-gutter: stable;
}}

CloneVoicePanel #clone-config-panel:focus-within {{
    border: solid {COLORS['accent_tertiary']};
    scrollbar-color: {COLORS['neon_pink']};
}}

CloneVoicePanel #clone-right-top {{
    layout: horizontal;
    height: 55%;
}}

CloneVoicePanel #clone-generator-panel {{
    width: 1fr;
    height: 100%;
    background: {COLORS['bg_elevated']};
    border: solid {COLORS['accent_primary']};
    border-title-color: {COLORS['accent_primary']};
    padding: 1 2;
}}

CloneVoicePanel #clone-training-panel {{
    width: 1fr;
    height: 100%;
    background: {COLORS['bg_elevated']};
    border: solid {COLORS['neon_cyan']};
    border-title-color: {COLORS['neon_cyan']};
    padding: 1 2;
    margin-left: 1;
    overflow-y: auto;
}}

CloneVoicePanel #clone-test-panel {{
    layout: horizontal;
    height: auto;
    min-height: 14;
    background: {COLORS['bg_elevated']};
    border: solid {COLORS['warning']};
    border-title-color: {COLORS['warning']};
    padding: 2 3;
    margin-top: 1;
}}

CloneVoicePanel #clone-test-left {{
    width: 1fr;
    height: auto;
    padding: 0 3 0 0;
    border-right: solid {COLORS['border_normal']};
}}

CloneVoicePanel #clone-test-right {{
    width: 2fr;
    height: auto;
    padding: 0 0 0 3;
}}

CloneVoicePanel #clone-model-selector-test {{
    width: 100%;
    margin-bottom: 1;
}}

CloneVoicePanel #clone-input-test-text {{
    width: 100%;
    height: 5;
    min-height: 5;
    margin-bottom: 1;
    background: {COLORS['bg_dark']};
    border: solid {COLORS['border_normal']};
}}

CloneVoicePanel #clone-btn-play {{
    width: 100%;
    background: {COLORS['warning']};
    color: {COLORS['bg_darkest']};
    text-style: bold;
    margin-top: 1;
}}

CloneVoicePanel #clone-btn-play:hover {{ background: #fbbf24; }}
CloneVoicePanel #clone-btn-play:disabled {{ background: {COLORS['border_dim']}; color: {COLORS['text_muted']}; }}

CloneVoicePanel #clone-btn-refresh-models {{
    width: 100%;
    background: {COLORS['accent_secondary']};
    color: {COLORS['bg_darkest']};
    margin-top: 1;
}}

CloneVoicePanel Select {{
    width: 100%;
}}

CloneVoicePanel .clone-panel-title {{
    text-style: bold;
    text-align: center;
    padding: 0;
    margin-bottom: 1;
}}

CloneVoicePanel #clone-config-panel .clone-panel-title {{ color: {COLORS['neon_pink']}; }}
CloneVoicePanel #clone-generator-panel .clone-panel-title {{ color: {COLORS['accent_primary']}; }}
CloneVoicePanel #clone-training-panel .clone-panel-title {{ color: {COLORS['neon_cyan']}; }}
CloneVoicePanel #clone-test-panel .clone-panel-title {{ color: {COLORS['warning']}; }}

CloneVoicePanel .hidden-log {{
    display: none;
}}

CloneVoicePanel .clone-setting-label {{
    color: {COLORS['text_secondary']};
    padding: 0;
    margin-top: 0;
    height: 1;
}}

CloneVoicePanel Input {{
    height: 3;
    padding: 0 1;
    margin: 0;
}}

CloneVoicePanel #clone-training-panel Input,
CloneVoicePanel #clone-generator-panel Input {{
    height: 3;
    margin-bottom: 1;
}}

CloneVoicePanel Button {{
    height: 3;
    min-width: 8;
    margin: 0;
}}

CloneVoicePanel #clone-btn-generate {{
    width: 100%;
    background: {COLORS['success']};
    color: {COLORS['bg_darkest']};
    text-style: bold;
    margin-top: 1;
}}

CloneVoicePanel #clone-btn-generate:hover {{ background: {COLORS['neon_green']}; }}
CloneVoicePanel #clone-btn-generate:disabled {{ background: {COLORS['border_dim']}; color: {COLORS['text_muted']}; }}

CloneVoicePanel #clone-btn-stop {{
    width: 100%;
    background: {COLORS['error']};
    color: {COLORS['bg_darkest']};
    text-style: bold;
    margin-top: 1;
}}

CloneVoicePanel #clone-btn-stop:disabled {{ background: {COLORS['border_dim']}; color: {COLORS['text_muted']}; }}

CloneVoicePanel #clone-status-indicator {{
    text-align: center;
    padding: 0;
    height: 1;
}}

CloneVoicePanel .clone-counter-display {{
    text-align: center;
    color: {COLORS['success']};
    text-style: bold;
    padding: 0;
    height: 1;
}}

CloneVoicePanel .clone-current-phrase {{
    text-align: center;
    color: {COLORS['text_secondary']};
    padding: 0;
    height: 1;
}}

CloneVoicePanel #clone-output-log {{
    height: 100%;
    background: {COLORS['bg_dark']};
    border: solid {COLORS['border_dim']};
}}

CloneVoicePanel RadioSet {{
    background: {COLORS['bg_dark']};
    border: solid {COLORS['border_dim']};
    height: auto;
    padding: 0;
}}

CloneVoicePanel RadioButton {{
    height: 1;
    padding: 0;
}}

CloneVoicePanel #clone-btn-browse {{
    width: 100%;
    height: 3;
    margin-top: 1;
    background: {COLORS['accent_primary']};
    color: {COLORS['bg_darkest']};
}}

CloneVoicePanel #clone-btn-browse:hover {{ background: {COLORS['accent_tertiary']}; }}

CloneVoicePanel .clone-file-hint {{
    color: {COLORS['neon_green']};
    text-style: italic;
    margin-top: 0;
    height: 1;
}}

CloneVoicePanel .clone-train-buttons {{
    layout: horizontal;
    height: auto;
    margin-top: 1;
}}

CloneVoicePanel .clone-train-buttons Button {{
    width: 1fr;
}}

CloneVoicePanel #clone-btn-refresh-datasets {{
    width: 100%;
    background: {COLORS['accent_secondary']};
    color: {COLORS['bg_darkest']};
    margin-top: 1;
}}

CloneVoicePanel Checkbox {{
    height: 3;
    margin: 0;
    padding: 0 1;
    width: auto;
}}

CloneVoicePanel .clone-use-top-row {{
    layout: horizontal;
    height: 3;
    margin: 0;
    align: left middle;
}}

CloneVoicePanel #clone-input-top-n {{
    width: 6;
    margin-left: 1;
}}

CloneVoicePanel .clone-model-hint {{
    color: {COLORS['text_muted']};
    text-style: italic;
    height: 1;
    padding: 0;
    margin: 0;
}}

CloneVoicePanel #clone-btn-train-chatterbox {{
    background: #f97316;
    color: {COLORS['bg_darkest']};
    text-style: bold;
    margin-right: 1;
}}

CloneVoicePanel #clone-btn-train-chatterbox:hover {{ background: #fb923c; }}

CloneVoicePanel #clone-btn-train-coqui {{
    background: #10b981;
    color: {COLORS['bg_darkest']};
    text-style: bold;
}}

CloneVoicePanel #clone-btn-train-coqui:hover {{ background: #34d399; }}

CloneVoicePanel .clone-training-status {{
    text-align: center;
    padding: 0;
    height: 1;
}}

CloneVoicePanel ProgressBar {{
    height: 1;
    margin: 0;
}}

CloneVoicePanel .clone-section-separator {{
    height: 1;
    margin: 1 0;
    border-top: solid {COLORS['border_dim']};
}}

CloneVoicePanel #clone-save-indicator {{
    text-align: right;
    color: {COLORS['text_muted']};
    height: 1;
    padding: 0;
    margin: 0;
}}
"""


def run_chatterbox_inference(model_path: Path, text: str, output_file: Path, root_dir: Path) -> None:
    python_exec = str(root_dir / "venv_chatterbox" / "bin" / "python")
    reference_audio = model_path / "reference.wav"
    if not reference_audio.exists():
        raise FileNotFoundError(f"Audio de referencia nao encontrado: {reference_audio}")
    script = (
        f'import torch\nimport torchaudio\n'
        f'from chatterbox.mtl_tts import ChatterboxMultilingualTTS\n'
        f'model = ChatterboxMultilingualTTS.from_pretrained('
        f'device="cuda" if torch.cuda.is_available() else "cpu")\n'
        f'audio = model.generate(text="{text}", audio_prompt_path="{reference_audio}", language_id="pt")\n'
        f'audio_cpu = audio.cpu()\n'
        f'if audio_cpu.dim() == 3: audio_cpu = audio_cpu.squeeze(0)\n'
        f'if audio_cpu.dim() == 1: audio_cpu = audio_cpu.unsqueeze(0)\n'
        f'torchaudio.save("{output_file}", audio_cpu, 24000)\n'
        f'print("OK")\n'
    )
    result = subprocess.run([python_exec, "-c", script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def run_coqui_inference(model_path: Path, text: str, output_file: Path, root_dir: Path) -> None:
    python_exec = str(root_dir / "venv_coqui" / "bin" / "python")
    reference_audio = model_path / "reference_speaker.wav"
    if not reference_audio.exists():
        raise FileNotFoundError(f"Audio de referencia nao encontrado: {reference_audio}")
    script = (
        f'from TTS.api import TTS\n'
        f'tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")\n'
        f'tts.tts_to_file(text="{text}", file_path="{output_file}", '
        f'speaker_wav="{reference_audio}", language="pt")\n'
        f'print("OK")\n'
    )
    result = subprocess.run([python_exec, "-c", script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def run_generation_thread(
    panel: "CloneVoicePanel",
    api_key: str,
    voice_id: str,
    dataset_name: str,
    model_id: str,
    phrases_file: Path,
    output_dir: Path,
) -> None:
    from src.core.theme import COLORS
    from datetime import datetime
    try:
        from src.modules.clone_voice.core.generator import DatasetManager, DatasetConfig
        cfg = DatasetConfig(
            name=dataset_name,
            voice_id=voice_id,
            model_id=model_id,
            total_phrases=panel.target_phrases,
            output_dir=output_dir,
            phrases_file=phrases_file if phrases_file.exists() else None,
        )
        manager = DatasetManager(cfg, api_key=api_key)
        manager.on_progress = lambda c, t, p: panel.call_from_thread(
            panel._on_generation_progress, c, t, p
        )
        manager.on_phrase_generated = lambda i, p, a: panel.call_from_thread(
            panel._on_phrase_generated, i, p, a
        )
        manager.on_error = lambda e: panel.call_from_thread(panel._on_generation_error, e)

        panel.call_from_thread(panel._set_dataset_manager, manager)
        panel.call_from_thread(panel._log, f"[{COLORS['success']}]Gerador OK[/]")
        panel.call_from_thread(panel._set_status, "generating")

        if not manager.initialize():
            panel.call_from_thread(panel._on_generation_failed, "Falha ao inicializar ElevenLabs")
            return

        stats = manager.generate_dataset()
        if manager._should_stop:
            panel.call_from_thread(panel._on_generation_stopped, stats)
        else:
            panel.call_from_thread(panel._on_generation_complete, stats)

    except ImportError as e:
        panel.call_from_thread(panel._on_generation_failed, f"Pacote nao instalado: {e}")
    except Exception as e:
        panel.call_from_thread(panel._on_generation_failed, str(e))


def run_training_thread(
    panel: "CloneVoicePanel",
    model_type: str,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    use_top: bool,
    top_n: int,
) -> None:
    from src.core.theme import COLORS
    try:
        from src.modules.clone_voice.core.training import TrainingConfig, ChatterboxTrainer, CoquiTrainer
        from src.modules.clone_voice.core import AudioQualityAnalyzer

        training_dataset_dir = dataset_dir
        unified_audio_path = None

        if use_top:
            panel.call_from_thread(panel._log, f"[{COLORS['info']}]Selecionando TOP {top_n} audios...[/]")
            analyzer = AudioQualityAnalyzer()
            result = analyzer.analyze_dataset(dataset_dir)
            if result.analyzed_files == 0:
                panel.call_from_thread(panel._on_training_error, "Nenhum audio no dataset")
                return
            selection = analyzer.get_diverse_selection(top_n)
            if not selection:
                panel.call_from_thread(panel._on_training_error, "Falha ao selecionar audios")
                return
            selection_dir = dataset_dir / f"top_{top_n}_selection"
            analyzer.export_selection(selection, selection_dir)
            panel.call_from_thread(
                panel._log,
                f"[{COLORS['info']}]Concatenando {len(selection)} audios em arquivo unico...[/]"
            )
            unified_audio_path = analyzer.create_unified_reference(selection, selection_dir)
            if not (unified_audio_path and unified_audio_path.exists()):
                panel.call_from_thread(panel._on_training_error, "Falha ao criar audio unificado")
                return
            panel.call_from_thread(
                panel._log, f"[{COLORS['success']}]Audio unificado: {unified_audio_path.name}[/]"
            )
            import shutil
            wavs_dir = selection_dir / "wavs"
            wavs_dir.mkdir(parents=True, exist_ok=True)
            for m in selection:
                dst = wavs_dir / m.file_name
                if not dst.exists():
                    shutil.copy(m.file_path, dst)
            training_dataset_dir = selection_dir
            panel.call_from_thread(
                panel._log, f"[{COLORS['success']}]TOP {top_n}: {len(selection)} audios prontos[/]"
            )

        base_name = re.sub(r'_\d{8}_\d{6}$', '', dataset_dir.name)
        cfg = TrainingConfig(
            dataset_dir=training_dataset_dir,
            output_dir=output_dir,
            model_name=base_name,
            epochs=epochs,
            unified_reference_path=unified_audio_path,
        )
        trainer = ChatterboxTrainer(cfg) if model_type == "chatterbox" else CoquiTrainer(cfg)
        panel.call_from_thread(panel._set_trainer, trainer)
        trainer.on_progress = lambda step, total, loss: panel.call_from_thread(
            panel._on_training_progress, step, total, loss
        )
        trainer.on_epoch_complete = lambda epoch, loss: panel.call_from_thread(
            panel._on_epoch_complete, epoch, loss
        )
        trainer.on_log = lambda msg: panel.call_from_thread(panel._log, msg)
        trainer.on_complete = lambda stats: panel.call_from_thread(panel._on_training_complete, stats)
        trainer.on_error = lambda err: panel.call_from_thread(panel._on_training_error, err)

        panel.call_from_thread(panel._update_training_status, "VALIDANDO")
        if not trainer.validate_dataset():
            panel.call_from_thread(panel._on_training_error, "Dataset invalido")
            return

        panel.call_from_thread(panel._update_training_status, "INICIALIZANDO")
        if not trainer.initialize():
            panel.call_from_thread(panel._on_training_error, "Falha ao inicializar")
            return

        panel.call_from_thread(panel._update_training_status, "TREINANDO")
        trainer.train()

    except ImportError as e:
        panel.call_from_thread(panel._on_training_error, f"Dependencia: {e}")
    except Exception as e:
        panel.call_from_thread(panel._on_training_error, str(e))


# "Divide et impera — não por poder, mas por clareza." — princípio de engenharia
