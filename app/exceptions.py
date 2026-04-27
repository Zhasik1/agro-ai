"""Domain-specific exceptions."""

from __future__ import annotations

__all__ = [
    "MalChainError",
    "AnimalNotDetectedError",
    "InvalidImageError",
    "DuplicateAnimalError",
    "AnimalNotFoundError",
    "OwnerNotFoundError",
]


class MalChainError(Exception):
    """Base class for all MalChain domain errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class AnimalNotDetectedError(MalChainError):
    """Raised when no supported animal is detected on the photo."""

    status_code = 422
    code = "animal_not_detected"


class InvalidImageError(MalChainError):
    """Raised when uploaded file fails validation (size / mime / corrupt)."""

    status_code = 400
    code = "invalid_image"


class DuplicateAnimalError(MalChainError):
    """Raised when registering an animal that already exists in the registry."""

    status_code = 409
    code = "duplicate_animal"

    def __init__(self, message: str, *, existing_id: str, similarity: float) -> None:
        super().__init__(message)
        self.existing_id = existing_id
        self.similarity = similarity


class AnimalNotFoundError(MalChainError):
    """Raised when an animal_id has no record in the database."""

    status_code = 404
    code = "animal_not_found"


class OwnerNotFoundError(MalChainError):
    """Raised when no owner is found for given IIN."""

    status_code = 404
    code = "owner_not_found"
