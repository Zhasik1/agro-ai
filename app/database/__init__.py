"""Database package: models, sessions, vector store, repository."""

from __future__ import annotations

from app.database.base import Base
from app.database.models import Animal, AnimalStatus, Owner, Species
from app.database.repository import (
    count_by_species,
    create_animal,
    create_owner,
    delete_animal,
    get_animal,
    get_owner,
    list_animals_by_owner,
    list_recent,
)
from app.database.session import SessionLocal, get_db, init_db

__all__ = [
    "Animal",
    "AnimalStatus",
    "Base",
    "Owner",
    "SessionLocal",
    "Species",
    "count_by_species",
    "create_animal",
    "create_owner",
    "delete_animal",
    "get_animal",
    "get_db",
    "get_owner",
    "init_db",
    "list_animals_by_owner",
    "list_recent",
]
