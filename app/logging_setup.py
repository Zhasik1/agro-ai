"""Centralised logging configuration.

Provides :func:`configure_logging` which sets up application-level logging
using ``structlog`` when available, falling back to :mod:`logging.basicConfig`
otherwise.  Import and call this function once at application start-up.
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging"]


def configure_logging(level: str) -> None:
    """Configure application logging.

    Uses ``structlog`` for structured JSON logging if the package is installed;
    falls back to :func:`logging.basicConfig` with a human-readable format
    otherwise.

    Args:
        level: Log level string, e.g. ``"INFO"``, ``"DEBUG"``, ``"WARNING"``.
    """
    log_level = level.upper()
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    try:
        import structlog

        logging.basicConfig(level=log_level, format=fmt)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        )
    except ImportError:
        logging.basicConfig(level=log_level, format=fmt)
