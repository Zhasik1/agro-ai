"""SQLAlchemy ORM models."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ["Base", "Species", "AnimalStatus", "Animal", "Owner"]


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Species(str, enum.Enum):
    """Supported livestock species."""

    CATTLE = "cattle"
    SHEEP = "sheep"
    HORSE = "horse"


class AnimalStatus(str, enum.Enum):
    """Lifecycle status of a registered animal."""

    ACTIVE = "active"
    SOLD = "sold"
    DECEASED = "deceased"


def _utcnow() -> datetime:
    """Return naive UTC timestamp (SQLite friendly)."""
    return datetime.utcnow()


class Owner(Base):
    """Farmer / owner of one or more animals (keyed by IIN)."""

    __tablename__ = "owners"

    iin: Mapped[str] = mapped_column(String(12), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    animals: Mapped[list[Animal]] = relationship(
        "Animal", back_populates="owner", cascade="all, delete-orphan"
    )


class Animal(Base):
    """Single registered animal with biometric identity."""

    __tablename__ = "animals"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    species: Mapped[Species] = mapped_column(Enum(Species), index=True, nullable=False)
    breed: Mapped[str | None] = mapped_column(String(128), nullable=True)
    age_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    owner_iin: Mapped[str] = mapped_column(
        String(12), ForeignKey("owners.iin"), index=True, nullable=False
    )
    photo_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    status: Mapped[AnimalStatus] = mapped_column(
        Enum(AnimalStatus), default=AnimalStatus.ACTIVE, nullable=False
    )
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    owner: Mapped[Owner] = relationship("Owner", back_populates="animals")


@event.listens_for(Animal, "before_update")
def _animal_before_update(mapper, connection, target: Animal) -> None:  # noqa: ANN001
    """Refresh ``updated_at`` on any row update."""
    target.updated_at = _utcnow()
