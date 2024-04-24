#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

handler = RotatingFileHandler(
    LOGS_DIR / "neurosonancy.log",
    maxBytes=5_000_000,
    backupCount=3
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("neurosonancy")


def run_launcher():
    from src.gui.main_menu import NeurosonancyLauncher
    app = NeurosonancyLauncher()
    return app.run()


def run_ascii_control():
    from src.modules.ascii_control.ui.app import AsciiControlApp
    app = AsciiControlApp(return_to_menu=True)
    return app.run()


def run_voice_trainer():
    from src.modules.voice_trainer.main import VoiceTrainerApp
    app = VoiceTrainerApp(return_to_menu=True)
    return app.run()


def run_clone_voice():
    from src.modules.clone_voice.ui.app import CloneVoiceApp
    app = CloneVoiceApp(return_to_menu=True)
    return app.run()


def main():
    logger.info("Neurosonancy iniciado")

    while True:
        result = run_launcher()

        if result == "ascii":
            logger.info("Iniciando Monitor")
            module_result = run_ascii_control()
            if module_result == "quit":
                break
        elif result == "voice":
            logger.info("Iniciando Trainer")
            module_result = run_voice_trainer()
            if module_result == "quit":
                break
        elif result == "clone":
            logger.info("Iniciando Clone Voice")
            module_result = run_clone_voice()
            if module_result == "quit":
                break
        else:
            logger.info("Neurosonancy encerrado")
            break


if __name__ == "__main__":
    main()
