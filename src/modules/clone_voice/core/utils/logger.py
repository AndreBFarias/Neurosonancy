"""
Sistema de Logging Ultra-Detalhado
Fornece logs estruturados, coloridos e com contexto completo
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime
import traceback


# Cores ANSI para terminal
class Colors:
    """Cores para formatação de logs no terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Cores normais
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Cores brilhantes
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Formatter customizado com cores para terminal."""
    
    # Mapeamento de níveis para cores
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_CYAN,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }
    
    # Símbolos para cada nível
    LEVEL_SYMBOLS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "💥",
    }
    
    def format(self, record):
        """Formata a mensagem de log com cores."""
        # Obter cor e símbolo do nível
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        level_symbol = self.LEVEL_SYMBOLS.get(record.levelno, "  ")
        
        # Adicionar cor ao nome do nível
        record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"
        
        # Adicionar símbolo
        record.levelname = f"{level_symbol} {record.levelname}"
        
        # Colorir nome do módulo
        record.name = f"{Colors.BRIGHT_BLUE}{record.name}{Colors.RESET}"
        
        # Colorir localização (arquivo:linha)
        location = f"{record.filename}:{record.lineno}"
        record.location = f"{Colors.BRIGHT_BLACK}({location}){Colors.RESET}"
        
        # Formatar timestamp
        record.timestamp = f"{Colors.BRIGHT_BLACK}{self.formatTime(record, self.datefmt)}{Colors.RESET}"
        
        return super().format(record)


class DetailedFormatter(logging.Formatter):
    """Formatter detalhado para arquivo de log (sem cores)."""
    
    def format(self, record):
        """Formata a mensagem de log com informações detalhadas."""
        # Adicionar informações extras
        record.location = f"{record.filename}:{record.lineno}"
        record.function_name = record.funcName
        
        # Se houver exceção, adicionar traceback completo
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        
        return super().format(record)


class TranscriptionLogger:
    """Logger customizado para o sistema de transcrição."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = None
            TranscriptionLogger._initialized = True
    
    def setup(
        self,
        log_level: str = "DEBUG",
        log_file: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        """
        Configura o sistema de logging.
        
        Args:
            log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Caminho para arquivo de log
            max_bytes: Tamanho máximo do arquivo de log em bytes
            backup_count: Número de arquivos de backup
        """
        # Criar logger root
        self.logger = logging.getLogger("transcription")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remover handlers existentes
        self.logger.handlers.clear()
        
        # === CONSOLE HANDLER (colorido) ===
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Usuário vê INFO+ (eventos importantes)
        
        console_format = (
            "%(message)s" # Clean format for console
        )
        
        console_formatter = ColoredFormatter(
            console_format,
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # === FILE HANDLER (detalhado) ===
        if log_file:
            # Garantir que o diretório existe
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            
            file_format = (
                "%(asctime)s | %(levelname)-8s | %(name)-25s | "
                "%(location)-30s | %(function_name)-20s\n"
                "%(message)s\n"
                "%(exc_text)s"
            )
            
            file_formatter = DetailedFormatter(
                file_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Não propagar para o root logger
        self.logger.propagate = False
        
        self.logger.debug(f"Nível de log: {log_level}")
        if log_file:
            self.logger.debug(f"Arquivo de log: {log_file}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Retorna um logger filho com o nome especificado.
        
        Args:
            name: Nome do módulo/componente
            
        Returns:
            Logger configurado
        """
        if self.logger is None:
            self.setup()
        
        return self.logger.getChild(name)


# Instância global
_transcription_logger = TranscriptionLogger()


def setup_logging(
    log_level: str = "DEBUG",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
):
    """
    Configura o sistema de logging global.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log
        max_bytes: Tamanho máximo do arquivo de log em bytes
        backup_count: Número de arquivos de backup
    """
    _transcription_logger.setup(log_level, log_file, max_bytes, backup_count)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado para o módulo.
    
    Args:
        name: Nome do módulo/componente
        
    Returns:
        Logger configurado
    """
    return _transcription_logger.get_logger(name)


# Para facilitar imports
__all__ = ["setup_logging", "get_logger"]
