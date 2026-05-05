"""Loguru-based logging setup."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Initialise the global loguru sink with a sensible default format."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level:<7}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "<level>{message}</level>"
        ),
    )


__all__ = ["configure_logging", "logger"]
