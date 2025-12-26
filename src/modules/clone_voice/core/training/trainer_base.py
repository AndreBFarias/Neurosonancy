# -*- coding: utf-8 -*-

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent
VENV_CHATTERBOX = ROOT_DIR / "venv_chatterbox"
VENV_COQUI = ROOT_DIR / "venv_coqui"


@dataclass
class TrainingConfig:
    dataset_dir: Path
    output_dir: Path
    model_name: str = "custom_voice"
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-5
    use_gpu: bool = True
    save_every_n_epochs: int = 25
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 50
    max_audio_length_seconds: float = 15.0
    unified_reference_path: Optional[Path] = None


@dataclass
class TrainingStats:
    model_type: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_epochs: int = 0
    current_epoch: int = 0
    total_steps: int = 0
    current_step: int = 0
    loss: float = 0.0
    best_loss: float = float('inf')
    output_path: Optional[Path] = None
    checkpoints: List[Path] = field(default_factory=list)
    is_completed: bool = False
    error: Optional[str] = None


class TrainerBase(ABC):

    VENV_PATH: Optional[Path] = None

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.stats = TrainingStats()
        self._is_training = False
        self._should_stop = False

        self.on_progress: Optional[Callable[[int, int, float], None]] = None
        self.on_epoch_complete: Optional[Callable[[int, float], None]] = None
        self.on_checkpoint_saved: Optional[Callable[[Path], None]] = None
        self.on_complete: Optional[Callable[[TrainingStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def _get_python_executable(self) -> str:
        if self.VENV_PATH and self.VENV_PATH.exists():
            return str(self.VENV_PATH / "bin" / "python")
        return "python3"

    def _check_venv(self) -> bool:
        if not self.VENV_PATH:
            return True

        if not self.VENV_PATH.exists():
            self._log(f"ERRO: venv nao encontrado: {self.VENV_PATH}")
            self._log("Execute ./install.sh para criar os ambientes")
            return False

        python_path = self.VENV_PATH / "bin" / "python"
        if not python_path.exists():
            self._log(f"ERRO: Python nao encontrado no venv: {python_path}")
            return False

        return True

    @abstractmethod
    def validate_dataset(self) -> bool:
        pass

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def train(self) -> TrainingStats:
        pass

    def stop(self) -> None:
        self._should_stop = True
        self._log("Solicitada parada do treinamento")

    @property
    def is_training(self) -> bool:
        return self._is_training

    def get_dataset_info(self) -> Dict[str, Any]:
        dataset_dir = self.config.dataset_dir

        info = {
            "path": str(dataset_dir),
            "exists": dataset_dir.exists(),
            "wav_count": 0,
            "total_duration_seconds": 0,
            "has_metadata": False,
        }

        if not dataset_dir.exists():
            return info

        wavs_dir = dataset_dir / "wavs"
        if wavs_dir.exists():
            audio_files = list(wavs_dir.glob("*.wav")) + list(wavs_dir.glob("*.mp3"))
            info["wav_count"] = len(audio_files)

        metadata_csv = dataset_dir / "metadata.csv"
        info["has_metadata"] = metadata_csv.exists() and metadata_csv.stat().st_size > 0

        return info
