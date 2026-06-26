"""
Logging utilities.

Sets up a root logger that writes to both stdout and a rotating file,
with a clean format that includes timestamp, level, and module name.

Usage::

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting training ...")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call.

    Args:
        name:    logger name (pass __name__ from calling module)
        log_dir: optional directory to write a ``stmad.log`` file;
                 only used on the *first* call to this function
    """
    global _configured

    if not _configured:
        _configure_root(log_dir)
        _configured = True

    return logging.getLogger(name)


def _configure_root(log_dir: str | Path | None = None) -> None:
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if log_dir is not None:
        log_path = Path(log_dir) / "stmad.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
