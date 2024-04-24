"""
Gerenciador de Configurações do Sistema
Carrega, valida e persiste configurações em arquivo .env
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv, set_key, find_dotenv


class Colors:
    """Cores ANSI para terminal (definidas localmente para evitar ciclo de import)."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"



class Settings:
    """Classe singleton para gerenciar configurações do sistema."""
    
    _instance: Optional['Settings'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.project_root = Path(__file__).parent.parent
            self.config_dir = self.project_root / "config"
            self.env_file = self.config_dir / ".env"
            self.env_example = self.config_dir / ".env.example"
            
            # Criar .env se não existir
            if not self.env_file.exists():
                self._create_default_env()
            
            # Carregar variáveis de ambiente
            load_dotenv(self.env_file)
            
            # Carregar todas as configurações
            self._load_settings()
            # Debug para verificar carregamento
            # print(f"DEBUG: Loaded settings. VISUALIZATION_MODE={self.visualization_mode}")
            Settings._initialized = True
    
    def _create_default_env(self):
        """Cria arquivo .env a partir do .env.example."""
        if self.env_example.exists():
            import shutil
            shutil.copy(self.env_example, self.env_file)
            print(f"✓ Arquivo .env criado em {self.env_file}")
        else:
            # Criar .env com valores padrão mínimos
            self.env_file.touch()
            print(f"✓ Arquivo .env vazio criado em {self.env_file}")
    
    def _load_settings(self):
        """Carrega todas as configurações do .env."""
        
        # === MODELO DE TRANSCRIÇÃO ===
        self.transcription_model = os.getenv("TRANSCRIPTION_MODEL", "whisper")
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        
        # === CONFIGURAÇÕES DE ÁUDIO ===
        self.audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
        self.audio_device_id = int(os.getenv("AUDIO_DEVICE_ID", "10"))
        self.audio_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
        self.audio_chunk_size = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
        
        # === VAD (Voice Activity Detection) ===
        self.vad_silence_duration = float(os.getenv("VAD_SILENCE_DURATION", "1.0"))
        self.vad_energy_threshold = int(os.getenv("VAD_ENERGY_THRESHOLD", "300"))
        self.vad_strategy = os.getenv("VAD_STRATEGY", "webrtc")
        
        # === PERFORMANCE E GPU ===
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.gpu_device = int(os.getenv("GPU_DEVICE", "0"))
        self.latency_mode = os.getenv("LATENCY_MODE", "balanced")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        
        # === ARMAZENAMENTO ===
        self.save_audio = os.getenv("SAVE_AUDIO", "true").lower() == "true"
        self.save_transcriptions = os.getenv("SAVE_TRANSCRIPTIONS", "true").lower() == "true"
        self.database_path = Path(os.getenv("DATABASE_PATH", "./data/transcriptions.db"))
        self.audio_storage_path = Path(os.getenv("AUDIO_STORAGE_PATH", "./data/audio/"))
        self.audio_retention_days = int(os.getenv("AUDIO_RETENTION_DAYS", "30"))
        
        # === VISUALIZAÇÃO ===
        self.enable_visualization = os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true"
        self.visualization_mode = os.getenv("VISUALIZATION_MODE", "gui")
        self.visualization_fps = int(os.getenv("VISUALIZATION_FPS", "30"))
        
        # === LOGGING ===
        self.log_level = os.getenv("LOG_LEVEL", "DEBUG")
        self.log_file = Path(os.getenv("LOG_FILE", "./logs/transcription.log"))
        self.log_max_size_mb = int(os.getenv("LOG_MAX_SIZE_MB", "10"))
        self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
        # === ADVANCED ===
        self.audio_buffer_duration = float(os.getenv("AUDIO_BUFFER_DURATION", "10.0"))
        self.processing_timeout = float(os.getenv("PROCESSING_TIMEOUT", "30.0"))
        
        # === UX ===
        self.auto_clipboard = os.getenv("AUTO_CLIPBOARD", "true").lower() == "true"
        self.show_progress = os.getenv("SHOW_PROGRESS", "true").lower() == "true"
        
        # Garantir que diretórios existem
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Garante que todos os diretórios necessários existem."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_storage_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def update(self, key: str, value: Any) -> bool:
        """
        Atualiza uma configuração e persiste no .env.
        
        Args:
            key: Nome da variável de ambiente
            value: Novo valor
            
        Returns:
            True se atualizado com sucesso
        """
        try:
            # Converter boolean para string
            if isinstance(value, bool):
                value = "true" if value else "false"
            else:
                value = str(value)
            
            # Atualizar no .env
            set_key(self.env_file, key, value)
            
            # Recarregar settings
            load_dotenv(self.env_file, override=True)
            self._load_settings()
            
            return True
        except Exception as e:
            print(f"✗ Erro ao atualizar configuração {key}: {e}")
            return False
    
    def save_config_to_env(self, config_dict: Dict[str, Any]) -> bool:
        """
        Salva múltiplas configurações no .env de uma vez.
        
        Args:
            config_dict: Dicionário com as configurações
            
        Returns:
            True se salvou com sucesso
        """
        try:
            for key, value in config_dict.items():
                # Converter boolean para string
                if isinstance(value, bool):
                    value = "true" if value else "false"
                else:
                    value = str(value)
                
                # Salvar no .env (em maiúsculas)
                env_key = key.upper()
                set_key(self.env_file, env_key, value)
            
            # Recarregar settings
            load_dotenv(self.env_file, override=True)
            self._load_settings()
            
            return True
        except Exception as e:
            print(f"✗ Erro ao salvar configurações: {e}")
            return False

    
    def get_all(self) -> Dict[str, Any]:
        """Retorna todas as configurações como dicionário."""
        return {
            # Modelo
            "transcription_model": self.transcription_model,
            "whisper_model_size": self.whisper_model_size,
            
            # Áudio
            "audio_sample_rate": self.audio_sample_rate,
            "audio_device_id": self.audio_device_id,
            "audio_channels": self.audio_channels,
            "audio_chunk_size": self.audio_chunk_size,
            
            # VAD
            "vad_silence_duration": self.vad_silence_duration,
            "vad_energy_threshold": self.vad_energy_threshold,
            "vad_strategy": self.vad_strategy,
            
            # Performance
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
            "latency_mode": self.latency_mode,
            "whisper_compute_type": self.whisper_compute_type,
            
            # Storage
            "save_audio": self.save_audio,
            "save_transcriptions": self.save_transcriptions,
            "database_path": str(self.database_path),
            "audio_storage_path": str(self.audio_storage_path),
            "audio_retention_days": self.audio_retention_days,
            
            # Visualização
            "enable_visualization": self.enable_visualization,
            "visualization_mode": self.visualization_mode,
            "visualization_fps": self.visualization_fps,
            
            # Logging
            "log_level": self.log_level,
            "log_file": str(self.log_file),
            "log_max_size_mb": self.log_max_size_mb,
            "log_backup_count": self.log_backup_count,
            
            # Advanced
            "audio_buffer_duration": self.audio_buffer_duration,
            "processing_timeout": self.processing_timeout,
        }
    
    def get_summary_text(self) -> str:
        """Retorna resumo das configurações como string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CONFIGURAÇÕES DO SISTEMA DE TRANSCRIÇÃO")
        lines.append("=" * 60)
        
        lines.append(f"\nMODELO:")
        lines.append(f"  • Tipo: {self.transcription_model}")
        if self.transcription_model == "whisper":
            lines.append(f"  • Tamanho: {self.whisper_model_size}")
        
        lines.append(f"\nÁUDIO:")
        lines.append(f"  • Sample Rate: {self.audio_sample_rate} Hz")
        lines.append(f"  • Device ID: {self.audio_device_id}")
        lines.append(f"  • Canais: {self.audio_channels}")
        
        lines.append(f"\nVAD:")
        lines.append(f"  • Silêncio: {self.vad_silence_duration}s")
        lines.append(f"  • Threshold: {self.vad_energy_threshold}")
        lines.append(f"  • Estratégia: {self.vad_strategy}")
        
        lines.append(f"\nPERFORMANCE:")
        lines.append(f"  • GPU: {'Ativada' if self.use_gpu else 'Desativada'}")
        lines.append(f"  • Latência: {self.latency_mode}")
        
        lines.append(f"\nARMAZENAMENTO:")
        lines.append(f"  • Salvar áudio: {'Sim' if self.save_audio else 'Não'}")
        lines.append(f"  • Salvar transcrições: {'Sim' if self.save_transcriptions else 'Não'}")
        
        lines.append(f"\nVISUALIZAÇÃO:")
        lines.append(f"  • Status: {'Ativada' if self.enable_visualization else 'Desativada'}")
        lines.append(f"  • Modo: {self.visualization_mode}")
        
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def print_summary(self):
        """Imprime resumo das configurações atuais."""
        print(self.get_summary_text())


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retorna a instância singleton de Settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Para facilitar imports
__all__ = ["Settings", "get_settings"]
