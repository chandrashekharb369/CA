"""
utils/logger.py — CA Intelligence Suite
Centralised logging configuration.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing 1,000 transactions...")
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_LOG_DIR  = "logs"
_LOG_FILE = os.path.join(_LOG_DIR, "ca_suite.log")
_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

os.makedirs(_LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Root logger setup (idempotent — configure once)
# ─────────────────────────────────────────────────────────────────────────────
def _configure_root_logger() -> None:
    """Configure the root logger with console + rotating file handlers."""
    root = logging.getLogger("ca_suite")
    if root.handlers:
        return  # Already configured — avoid adding duplicate handlers

    root.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))

    # Rotating file handler (5 MB per file, keep 3 backups)
    try:
        file_handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        root.addHandler(file_handler)
    except OSError:
        pass  # If log directory is read-only, skip file logging

    root.addHandler(console)


_configure_root_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'ca_suite' root.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Data loaded: %d rows", len(df))
    """
    return logging.getLogger(f"ca_suite.{name}")
