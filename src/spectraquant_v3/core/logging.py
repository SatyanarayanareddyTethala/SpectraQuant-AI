"""Logging setup for SpectraQuant-AI-V3.

Call ``setup_logging()`` once at process startup from the CLI entry point.
Individual modules obtain loggers via ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Configure the root logger with a console handler and optional file handler.

    This function is idempotent – calling it a second time with different
    parameters will add a new handler but not remove existing ones.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write structured log output.
        fmt: ``logging.Formatter`` format string.
    """
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(fmt)

    # Add console handler only if none exists yet
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Thin wrapper kept for import convenience."""
    return logging.getLogger(name)
