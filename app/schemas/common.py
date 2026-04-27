"""Shared enum types used across the MalChain application.

Re-exports :class:`Species` and :class:`AnimalStatus` from the ORM models so
that business-logic code can import them from a schema-level location without
pulling in SQLAlchemy.
"""

from __future__ import annotations

from app.database.models import AnimalStatus, Species

__all__ = ["AnimalStatus", "Species"]
