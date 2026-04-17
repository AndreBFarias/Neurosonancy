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
    LOGS_DIR / "neurosonancy.log", maxBytes=5_000_000, backupCount=3
)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("neurosonancy")


def main() -> None:
    logger.info("Neurosonancy iniciado")
    from src.unified_app import NeurosonancyUnifiedApp

    NeurosonancyUnifiedApp().run()
    logger.info("Neurosonancy encerrado")


if __name__ == "__main__":
    main()
