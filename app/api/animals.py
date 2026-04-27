"""Animal-related endpoints: register, identify, lookup."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database.models import Animal, AnimalStatus, Owner, Species
from app.database.session import get_db
from app.exceptions import AnimalNotFoundError, OwnerNotFoundError
from app.schemas.animal import (
    AnimalRegisterRequest,
    AnimalResponse,
    IdentificationResult,
    RegistrationResult,
)
from app.utils.image_utils import decode_image, sha256_bytes, validate_image_bytes

__all__ = ["router"]

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/animals", tags=["animals"])


@router.post(
    "/identify",
    response_model=IdentificationResult,
    summary="Жануарды тану / Identify an animal",
)
async def identify_endpoint(photo: UploadFile = File(...)) -> IdentificationResult:
    """Run the identification pipeline on an uploaded photo."""
    from app.ai import pipeline  # lazy import — heavy ML deps

    data = await photo.read()
    validate_image_bytes(data, photo.content_type)
    image = decode_image(data)
    outcome = pipeline.identify_animal(image)
    return outcome.result


@router.post(
    "/register",
    response_model=RegistrationResult,
    status_code=status.HTTP_201_CREATED,
    summary="Жаңа жануарды тіркеу / Register a new animal",
)
async def register_endpoint(
    photo: Annotated[UploadFile, File(...)],
    owner_iin: Annotated[str, Form(...)],
    age_years: Annotated[int | None, Form()] = None,
    weight_kg: Annotated[float | None, Form()] = None,
    breed: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
) -> RegistrationResult:
    """Register a new animal. Duplicates raise HTTP 409."""
    try:
        payload = AnimalRegisterRequest(
            owner_iin=owner_iin,
            age_years=age_years,
            weight_kg=weight_kg,
            breed=breed,
            notes=notes,
        )
    except ValidationError as exc:
        # Avoid leaking non-serializable `ctx.error` objects into JSON responses.
        first_error = exc.errors()[0] if exc.errors() else {}
        detail = first_error.get("msg", "Validation error")
        raise HTTPException(status_code=422, detail=detail) from exc

    from app.ai import pipeline  # lazy import

    data = await photo.read()
    validate_image_bytes(data, photo.content_type)
    photo_hash = sha256_bytes(data)

    # Owner: upsert minimal record so registration never silently fails on FK
    owner = db.get(Owner, payload.owner_iin)
    if owner is None:
        owner = Owner(iin=payload.owner_iin)
        db.add(owner)
        db.flush()

    image = decode_image(data)
    outcome = pipeline.register_animal(image)

    animal = Animal(
        id=outcome.animal_id,
        species=outcome.species,
        breed=payload.breed,
        age_years=payload.age_years,
        weight_kg=payload.weight_kg,
        owner_iin=payload.owner_iin,
        photo_hash=photo_hash,
        status=AnimalStatus.ACTIVE,
        notes=payload.notes,
    )
    db.add(animal)
    db.commit()
    db.refresh(animal)

    return RegistrationResult(
        animal=AnimalResponse.model_validate(animal),
        detection_confidence=outcome.detection.confidence,
    )


@router.get(
    "/by-owner/{iin}",
    response_model=list[AnimalResponse],
    summary="Иесінің малдары / List animals by owner",
)
def list_by_owner(iin: str, db: Session = Depends(get_db)) -> list[AnimalResponse]:
    """List every animal registered to an owner IIN."""
    if len(iin) != 12 or not iin.isdigit():
        raise HTTPException(status_code=422, detail="ЖСН 12 саннан тұруы керек")
    owner = db.get(Owner, iin)
    if owner is None:
        raise OwnerNotFoundError(f"Иесі табылмады: {iin}")
    return [AnimalResponse.model_validate(a) for a in owner.animals]


@router.get(
    "/{animal_id}",
    response_model=AnimalResponse,
    summary="Жануар туралы ақпарат / Animal details",
)
def get_animal(animal_id: str, db: Session = Depends(get_db)) -> AnimalResponse:
    """Return a single animal by id."""
    animal = db.get(Animal, animal_id)
    if animal is None:
        raise AnimalNotFoundError(f"Жануар табылмады: {animal_id}")
    return AnimalResponse.model_validate(animal)


@router.delete(
    "/{animal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Жануарды жою / Delete animal",
)
def delete_animal(animal_id: str, db: Session = Depends(get_db)) -> Response:
    """Delete an animal record (also removes from FAISS index)."""
    from app.database.vector_db import get_vector_db

    animal = db.get(Animal, animal_id)
    if animal is None:
        raise AnimalNotFoundError(f"Жануар табылмады: {animal_id}")
    species: Species = animal.species
    db.delete(animal)
    db.commit()
    get_vector_db().remove(species, animal_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
