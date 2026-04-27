"""Pydantic request / response schemas."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.database.models import AnimalStatus, Species

__all__ = [
    "IIN_PATTERN",
    "OwnerResponse",
    "AnimalRegisterRequest",
    "AnimalResponse",
    "Candidate",
    "IdentificationResult",
    "RegistrationResult",
    "SpeciesCount",
    "RecentRegistration",
    "StatsResponse",
]


IIN_PATTERN = re.compile(r"^\d{12}$")


def _validate_iin(value: str) -> str:
    """Validate Kazakh IIN: exactly 12 digits."""
    if not IIN_PATTERN.fullmatch(value):
        raise ValueError("ЖСН 12 саннан тұруы керек / IIN must be 12 digits")
    return value


class OwnerResponse(BaseModel):
    """Owner information returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    iin: str
    full_name: str | None = None
    phone: str | None = None
    region: str | None = None
    registered_at: datetime


class AnimalRegisterRequest(BaseModel):
    """Form payload for ``POST /api/animals/register`` (without the photo)."""

    owner_iin: str = Field(..., description="12-digit Kazakh IIN")
    age_years: int | None = Field(None, ge=0, le=40)
    weight_kg: float | None = Field(None, ge=0, le=2000)
    breed: str | None = Field(None, max_length=128)
    notes: str | None = Field(None, max_length=500)

    @field_validator("owner_iin")
    @classmethod
    def _check_iin(cls, v: str) -> str:
        return _validate_iin(v)


class AnimalResponse(BaseModel):
    """Animal data returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    species: Species
    breed: str | None = None
    age_years: int | None = None
    weight_kg: float | None = None
    owner_iin: str
    photo_hash: str
    status: AnimalStatus
    registered_at: datetime
    updated_at: datetime
    notes: str | None = None


class Candidate(BaseModel):
    """One match candidate from the vector search."""

    animal_id: str
    similarity: float = Field(..., ge=-1.0, le=1.0)


class IdentificationResult(BaseModel):
    """Result of running the identification pipeline."""

    status: Literal["matched", "suspect", "new", "error"]
    species: Species | None = None
    detection_confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    best_match: Candidate | None = None
    message: str | None = None


class RegistrationResult(BaseModel):
    """Successful registration outcome."""

    animal: AnimalResponse
    detection_confidence: float
    message: str = "Тіркеу сәтті аяқталды"


class SpeciesCount(BaseModel):
    """Per-species count entry."""

    species: Species
    count: int


class RecentRegistration(BaseModel):
    """Trimmed animal entry for dashboard listings."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    species: Species
    owner_iin: str
    registered_at: datetime


class StatsResponse(BaseModel):
    """Aggregate statistics for the dashboard."""

    total: int
    per_species: list[SpeciesCount]
    recent: list[RecentRegistration]
