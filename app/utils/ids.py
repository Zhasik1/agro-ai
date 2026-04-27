"""Animal-ID generator."""

from __future__ import annotations

import uuid

from app.database.models import Species

__all__ = ["SPECIES_PREFIX", "generate_animal_id"]


SPECIES_PREFIX: dict[Species, str] = {
    Species.CATTLE: "COW",
    Species.SHEEP: "SHP",
    Species.HORSE: "HRS",
}


def generate_animal_id(species: Species) -> str:
    """Generate a unique animal id like ``COW-A7F8B2``.

    The 6-char suffix is the upper-cased first 6 hex chars of a uuid4.
    """
    prefix = SPECIES_PREFIX[species]
    suffix = uuid.uuid4().hex[:6].upper()
    return f"{prefix}-{suffix}"
