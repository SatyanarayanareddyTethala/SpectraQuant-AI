from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Deque, Iterable


_LOG_BUFFER: Deque[str] = deque(maxlen=300)
_CONFIGURED = False


class InMemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        _LOG_BUFFER.append(msg)


def configure_logger(
    name: str = "spectraquant.dashboard",
    log_path: Path | None = None,
) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    memory_handler = InMemoryLogHandler()
    memory_handler.setFormatter(formatter)
    logger.addHandler(memory_handler)

    if log_path is None:
        log_path = Path("reports") / "dashboard" / "dashboard.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Failed to configure dashboard log file at %s", log_path)

    _CONFIGURED = True
    return logger


def get_recent_logs(limit: int = 200) -> Iterable[str]:
    if limit <= 0:
        return []
    return list(_LOG_BUFFER)[-limit:]
