"""FastAPI dependency providers.

Centralises the construction of shared resources (DB sessions, AI pipeline,
vector store) so routers can consume them via ``Depends(...)`` without
importing implementation modules directly.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.database.session import SessionLocal

__all__ = ["get_db", "get_pipeline", "get_vector_manager"]


def get_db() -> Iterator[Session]:
    """Yield a request-scoped SQLAlchemy session.

    Ensures the session is closed in a ``finally`` block regardless of
    whether the request succeeded or raised an exception.

    Yields:
        An open :class:`~sqlalchemy.orm.Session`.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_pipeline():  # noqa: ANN201
    """Return the AI identification pipeline module.

    Returns:
        The :mod:`app.ai.pipeline` module exposing ``identify_animal`` and
        ``register_animal``.
    """
    from app.ai import pipeline as _pipeline

    return _pipeline


def get_vector_manager():  # noqa: ANN201
    """Return the singleton :class:`~app.database.vector_db.VectorDBManager`.

    Returns:
        The process-wide :class:`~app.database.vector_db.VectorDBManager`
        instance, initialised lazily on first access.
    """
    from app.database.vector_db import get_vector_db

    return get_vector_db()
