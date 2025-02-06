# -*- coding: utf-8 -*-

import gc
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
VENV_CHATTERBOX = ROOT_DIR / "venv_chatterbox"
VENV_COQUI = ROOT_DIR / "venv_coqui"


class ModelCategory(str, Enum):
    TTS = "tts"
    STT = "stt"
    LLM = "llm"


class VenvTarget(str, Enum):
    MAIN = "main"
    CHATTERBOX = "chatterbox"
    COQUI = "coqui"


@dataclass
class ModelSpec:
    category: ModelCategory
    engine: str
    variant: str
    venv: VenvTarget
    display_name: str


class BaseModel(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    async def load(self) -> bool:
        pass

    @abstractmethod
    async def unload(self) -> None:
        pass


class SubprocessModel(BaseModel):
    def _get_python(self) -> str:
        venv_map = {
            VenvTarget.CHATTERBOX: VENV_CHATTERBOX,
            VenvTarget.COQUI: VENV_COQUI,
        }
        venv_path = venv_map.get(self.spec.venv)
        if venv_path:
            python_bin = venv_path / "bin" / "python"
            if python_bin.exists():
                return str(python_bin)
            logger.warning("venv %s nao encontrado, usando python3 do sistema", venv_path)
        return "python3"

    async def _run_in_venv(self, script: str, timeout: float = 30.0) -> tuple[int, str, str]:
        python = self._get_python()
        proc = await asyncio.create_subprocess_exec(
            python, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", "timeout"


class ChatterboxTTSModel(SubprocessModel):
    async def load(self) -> bool:
        probe = "import chatterbox; print('ok')"
        code, out, err = await self._run_in_venv(probe, timeout=15.0)
        if code == 0 and "ok" in out:
            self._is_loaded = True
            logger.info("ChatterboxTTS carregado")
            return True
        logger.warning("ChatterboxTTS probe falhou: %s", err[:200])
        return False

    async def unload(self) -> None:
        self._is_loaded = False


class CoquiXTTSModel(SubprocessModel):
    async def load(self) -> bool:
        probe = "from TTS.api import TTS; print('ok')"
        code, out, err = await self._run_in_venv(probe, timeout=15.0)
        if code == 0 and "ok" in out:
            self._is_loaded = True
            logger.info("CoquiXTTS carregado")
            return True
        logger.warning("CoquiXTTS probe falhou: %s", err[:200])
        return False

    async def unload(self) -> None:
        self._is_loaded = False


class WhisperSTTModel(BaseModel):
    _model_instance = None

    async def load(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model_instance = WhisperModel("base", device="cpu", compute_type="int8")
            self._is_loaded = True
            logger.info("WhisperSTT carregado")
            return True
        except ImportError:
            logger.warning("faster_whisper nao instalado")
            return False
        except Exception as e:
            logger.error("Erro ao carregar WhisperSTT: %s", e)
            return False

    async def unload(self) -> None:
        self._model_instance = None
        self._is_loaded = False
        gc.collect()


class OllamaLLMModel(BaseModel):
    async def load(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    self._is_loaded = True
                    logger.info("OllamaLLM disponivel")
                    return True
        except Exception as e:
            logger.warning("Ollama nao disponivel: %s", e)
        return False

    async def unload(self) -> None:
        self._is_loaded = False


class ModelFactory:
    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, key: str, model_class: type[BaseModel]) -> None:
        cls._registry[key] = model_class

    @classmethod
    def create(cls, spec: ModelSpec) -> BaseModel:
        key = f"{spec.engine}:{spec.variant}"
        model_class = cls._registry.get(key, SubprocessModel)
        return model_class(spec)


ModelFactory.register("chatterbox:multilingual", ChatterboxTTSModel)
ModelFactory.register("coqui:xtts", CoquiXTTSModel)
ModelFactory.register("whisper:base", WhisperSTTModel)
ModelFactory.register("ollama:default", OllamaLLMModel)


AVAILABLE_MODELS: list[ModelSpec] = [
    ModelSpec(
        category=ModelCategory.TTS,
        engine="chatterbox",
        variant="multilingual",
        venv=VenvTarget.CHATTERBOX,
        display_name="TTS: Chatterbox Multilingual",
    ),
    ModelSpec(
        category=ModelCategory.TTS,
        engine="coqui",
        variant="xtts",
        venv=VenvTarget.COQUI,
        display_name="TTS: Coqui XTTS",
    ),
    ModelSpec(
        category=ModelCategory.STT,
        engine="whisper",
        variant="base",
        venv=VenvTarget.MAIN,
        display_name="STT: Whisper Base",
    ),
    ModelSpec(
        category=ModelCategory.LLM,
        engine="ollama",
        variant="default",
        venv=VenvTarget.MAIN,
        display_name="LLM: Ollama Local",
    ),
]


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_init(self) -> None:
        if not self._initialized:
            self._current: dict[ModelCategory, BaseModel | None] = {c: None for c in ModelCategory}
            self._lock: asyncio.Lock = asyncio.Lock()
            self._initialized = True

    async def switch(self, category: ModelCategory, engine: str, variant: str) -> bool:
        self._ensure_init()
        async with self._lock:
            current = self._current.get(category)
            if current and current.is_loaded:
                await current.unload()

            spec = self._find_spec(category, engine, variant)
            if spec is None:
                logger.warning("ModelSpec nao encontrado: %s/%s/%s", category, engine, variant)
                return False

            model = ModelFactory.create(spec)
            success = await model.load()
            if success:
                self._current[category] = model
                logger.info("Modelo ativo: %s", spec.display_name)
            return success

    def _find_spec(self, category: ModelCategory, engine: str, variant: str) -> ModelSpec | None:
        for spec in AVAILABLE_MODELS:
            if spec.category == category and spec.engine == engine and spec.variant == variant:
                return spec
        return None

    def get_current(self, category: ModelCategory) -> BaseModel | None:
        self._ensure_init()
        return self._current.get(category)

    def get_display_names(self) -> list[str]:
        return [s.display_name for s in AVAILABLE_MODELS]

    def get_spec_by_display_name(self, display_name: str) -> ModelSpec | None:
        for spec in AVAILABLE_MODELS:
            if spec.display_name == display_name:
                return spec
        return None


# "O homem sábio não teme a mudança — ele a dirige." — Marco Aurélio
