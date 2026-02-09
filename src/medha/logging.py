"""Centralized logging configuration for Medha.

Provides a convenience function to configure the ``medha`` logger hierarchy.
All modules use ``logging.getLogger(__name__)`` so they inherit from the
top-level ``medha`` logger.

Usage::

    from medha.logging import setup_logging

    # Quick setup — DEBUG to console
    setup_logging(level="DEBUG")

    # Production — INFO to file, WARNING to console
    setup_logging(level="INFO", log_file="medha.log", console_level="WARNING")
"""

import logging
import sys
from typing import Optional


LIBRARY_LOGGER_NAME = "medha"

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_level: Optional[str] = None,
    fmt: Optional[str] = None,
    date_fmt: Optional[str] = None,
) -> logging.Logger:
    """Configure the ``medha`` logger hierarchy.

    Args:
        level: Root level for the ``medha`` logger
            (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: If provided, add a FileHandler writing at *level*.
        console_level: Separate level for the console (stderr) handler.
            Defaults to *level* when not set.
        fmt: Log format string. Uses a sensible default if omitted.
        date_fmt: Date format string. Uses ISO-style default if omitted.

    Returns:
        The configured ``medha`` root logger.
    """
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=fmt or _DEFAULT_FORMAT,
        datefmt=date_fmt or _DEFAULT_DATE_FORMAT,
    )

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    effective_console_level = console_level or level
    console.setLevel(getattr(logging, effective_console_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate output
    logger.propagate = False

    return logger
