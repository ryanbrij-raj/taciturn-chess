"""
utils/logger.py — Consistent logging across all modules.
Logs to both console and a file.
"""

import os
import logging
from datetime import datetime
from config import LOG_DIR


_loggers = {}

def get_logger(name: str) -> logging.Logger:
    """Get or create a named logger."""
    if name in _loggers:
        return _loggers[name]

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"taciturn_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(f"taciturn.{name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler (INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(ch)

        # File handler (DEBUG and above)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger
