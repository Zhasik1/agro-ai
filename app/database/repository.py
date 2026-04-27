"""Plain SQLAlchemy CRUD helpers for Animal and Owner models.

These functions contain no business logic — they are thin wrappers around
SQLAlchemy queries intended to be called from API endpoints and tests.
All functions accept an explicit :class:`~sqlalchemy.orm.Session` so
they remain easy to unit-test with any session fixture.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Animal, AnimalStatus, Owner, Species

__all__ = [
    "count_by_species",
    "create_animal",
    "create_owner",
    "delete_animal",
    "get_animal",
    "get_owner",
    "list_animals_by_owner",
    "list_recent",
]


def create_owner(
    session: Session,
    iin: str,
    full_name: str | None = None,
    phone: str | None = None,
    region: str | None = None,
) -> Owner:
    """Create and flush a new :class:`Owner` record.

    Args:
        session: Active SQLAlchemy session.
        iin: 12-digit Kazakh IIN (primary key).
        full_name: Optional owner display name.
        phone: Optional contact phone number.
        region: Optional region/oblast name.

    Returns:
        The newly created :class:`Owner` instance.
    """
    owner = Owner(iin=iin, full_name=full_name, phone=phone, region=region)
    session.add(owner)
    session.flush()
    return owner


def get_owner(session: Session, iin: str) -> Owner | None:
    """Return the :class:`Owner` with the given IIN, or ``None``.

    Args:
        session: Active SQLAlchemy session.
        iin: 12-digit Kazakh IIN.

    Returns:
        Matching :class:`Owner` or ``None`` if not found.
    """
    return session.get(Owner, iin)


def create_animal(
    session: Session,
    animal_id: str,
    species: Species,
    owner_iin: str,
    breed: str | None,
    age_years: int | None,
    weight_kg: float | None,
    photo_sha256: str,
) -> Animal:
    """Create and flush a new :class:`Animal` record.

    Args:
        session: Active SQLAlchemy session.
        animal_id: Pre-generated unique ID (e.g. ``COW-A7F8B2``).
        species: Animal species enum value.
        owner_iin: IIN of the owning farmer (FK).
        breed: Optional breed string.
        age_years: Optional age in years.
        weight_kg: Optional weight in kilograms.
        photo_sha256: SHA-256 hex digest of the registration photo.

    Returns:
        The newly created :class:`Animal` instance.
    """
    animal = Animal(
        id=animal_id,
        species=species,
        owner_iin=owner_iin,
        breed=breed,
        age_years=age_years,
        weight_kg=weight_kg,
        photo_hash=photo_sha256,
        status=AnimalStatus.ACTIVE,
    )
    session.add(animal)
    session.flush()
    return animal


def get_animal(session: Session, animal_id: str) -> Animal | None:
    """Return the :class:`Animal` with the given ID, or ``None``.

    Args:
        session: Active SQLAlchemy session.
        animal_id: Unique animal identifier (e.g. ``COW-A7F8B2``).

    Returns:
        Matching :class:`Animal` or ``None`` if not found.
    """
    return session.get(Animal, animal_id)


def list_animals_by_owner(session: Session, iin: str) -> list[Animal]:
    """Return all animals registered to the given owner IIN.

    Args:
        session: Active SQLAlchemy session.
        iin: 12-digit Kazakh IIN.

    Returns:
        List of :class:`Animal` instances (may be empty).
    """
    return list(session.execute(select(Animal).where(Animal.owner_iin == iin)).scalars().all())


def delete_animal(session: Session, animal_id: str) -> bool:
    """Delete an animal by ID and flush the change.

    Args:
        session: Active SQLAlchemy session.
        animal_id: Unique animal identifier.

    Returns:
        ``True`` if the animal was found and deleted, ``False`` if not found.
    """
    animal = session.get(Animal, animal_id)
    if animal is None:
        return False
    session.delete(animal)
    session.flush()
    return True


def count_by_species(session: Session) -> dict[Species, int]:
    """Return a mapping of each :class:`Species` to its animal count.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        Dict with all three species keys, values default to ``0``.
    """
    result: dict[Species, int] = dict.fromkeys(Species, 0)
    rows = session.execute(
        select(Animal.species, func.count(Animal.id)).group_by(Animal.species)
    ).all()
    for species_value, count in rows:
        result[species_value] = int(count)
    return result


def list_recent(session: Session, limit: int = 10) -> list[Animal]:
    """Return the most recently registered animals.

    Args:
        session: Active SQLAlchemy session.
        limit: Maximum number of records to return (default 10).

    Returns:
        List of :class:`Animal` instances ordered by ``registered_at`` desc.
    """
    return list(
        session.execute(select(Animal).order_by(Animal.registered_at.desc()).limit(limit))
        .scalars()
        .all()
    )
