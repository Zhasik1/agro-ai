"""Declarative base re-export for ergonomic imports.

Provides a single import target for :class:`Base` so downstream code does not
need to know that it lives inside :mod:`app.database.models`.
"""

from __future__ import annotations

from app.database.models import Base

__all__ = ["Base"]
