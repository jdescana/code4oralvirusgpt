from __future__ import annotations

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOGGER_NAME_ROOT: Final[str] = "oral_virus_gpt"


def configure_logging(level: int | str = logging.INFO, fmt: str = _DEFAULT_FORMAT) -> None:
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root = logging.getLogger(_LOGGER_NAME_ROOT)
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None or name == _LOGGER_NAME_ROOT:
        return logging.getLogger(_LOGGER_NAME_ROOT)
    if name.startswith(_LOGGER_NAME_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME_ROOT}.{name}")
