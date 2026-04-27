"""Identification + registration pipeline orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.ai.extractors.feature_extractor import get_feature_extractor
from app.ai.species_classifier import SpeciesDetection, get_species_classifier
from app.ai.thresholds import classify_match_status
from app.config import Settings, get_settings
from app.database.models import Species
from app.database.vector_db import VectorDBManager, VectorMatch, get_vector_db
from app.exceptions import AnimalNotDetectedError, DuplicateAnimalError
from app.schemas.animal import Candidate, IdentificationResult
from app.utils.ids import generate_animal_id

__all__ = [
    "IdentificationOutcome",
    "RegistrationOutcome",
    "identify_animal",
    "register_animal",
]

logger = logging.getLogger(__name__)


@dataclass
class IdentificationOutcome:
    """Internal pipeline result with the embedding kept for re-use."""

    result: IdentificationResult
    embedding: np.ndarray | None
    detection: SpeciesDetection | None


@dataclass
class RegistrationOutcome:
    """Internal result of a successful registration."""

    animal_id: str
    species: Species
    embedding: np.ndarray
    detection: SpeciesDetection


def _sub_detect(detection: SpeciesDetection) -> np.ndarray:
    """Apply the species-appropriate sub-detector to the animal crop.

    Routes the bounding-box crop through the geometric ROI heuristic that
    focuses on the discriminative region (muzzle for cattle, face for
    sheep/horse) before passing it to the feature extractor.

    Args:
        detection: Result from the species classifier containing the crop.

    Returns:
        Sub-cropped BGR ndarray (or the original crop if too small).
    """
    from app.ai.detectors.face_detector import detect_face
    from app.ai.detectors.muzzle_detector import detect_muzzle

    if detection.species == Species.CATTLE:
        return detect_muzzle(detection.cropped_image)
    if detection.species == Species.SHEEP:
        return detect_face(detection.cropped_image, species="sheep")
    if detection.species == Species.HORSE:
        return detect_face(detection.cropped_image, species="horse")
    return detection.cropped_image


def _classify_status(matches: list[VectorMatch], settings: Settings) -> str:
    """Decide pipeline status from the top match similarity.

    Delegates to :func:`app.ai.thresholds.classify_match_status` so threshold
    logic lives in a single place.  The ``settings`` parameter is kept for
    backward compatibility with existing callers and tests.
    """
    if not matches:
        return "new"
    return classify_match_status(matches[0].similarity)


def identify_animal(
    image_bgr: np.ndarray,
    *,
    classifier=None,  # noqa: ANN001 (lazy injectable for tests)
    extractor=None,  # noqa: ANN001
    vector_db: VectorDBManager | None = None,
    settings: Settings | None = None,
    top_k: int = 3,
) -> IdentificationOutcome:
    """Run the full identification pipeline for a single image.

    Steps:
        1. Detect species & crop the animal (YOLOv8).
        2. Extract a 512-dim embedding (ResNet50 + projection).
        3. Search the per-species FAISS index.
        4. Apply thresholds to derive ``status`` (matched / suspect / new).
    """
    classifier = classifier or get_species_classifier()
    extractor = extractor or get_feature_extractor()
    vector_db = vector_db or get_vector_db()
    settings = settings or get_settings()

    try:
        detection = classifier.detect(image_bgr)
    except AnimalNotDetectedError:
        raise

    sub_crop = _sub_detect(detection)
    embedding = extractor.extract(sub_crop)
    matches = vector_db.search(detection.species, embedding, top_k=top_k)
    status = _classify_status(matches, settings)

    candidates = [Candidate(animal_id=m.animal_id, similarity=m.similarity) for m in matches]
    best = candidates[0] if candidates else None

    result = IdentificationResult(
        status=status,  # type: ignore[arg-type]
        species=detection.species,
        detection_confidence=detection.confidence,
        bbox=detection.bbox,
        candidates=candidates,
        best_match=best,
    )
    logger.info(
        "identify status=%s species=%s top_sim=%s",
        status,
        detection.species.value,
        best.similarity if best else None,
    )
    return IdentificationOutcome(result=result, embedding=embedding, detection=detection)


def register_animal(
    image_bgr: np.ndarray,
    *,
    classifier=None,  # noqa: ANN001
    extractor=None,  # noqa: ANN001
    vector_db: VectorDBManager | None = None,
    settings: Settings | None = None,
) -> RegistrationOutcome:
    """Register a new animal — duplicates are rejected with 409.

    Raises:
        AnimalNotDetectedError: forwarded from the classifier.
        DuplicateAnimalError: when a matching animal already exists.
    """
    outcome = identify_animal(
        image_bgr,
        classifier=classifier,
        extractor=extractor,
        vector_db=vector_db,
        settings=settings,
    )
    vector_db = vector_db or get_vector_db()

    if outcome.result.status == "matched" and outcome.result.best_match is not None:
        existing = outcome.result.best_match
        raise DuplicateAnimalError(
            "Бұл жануар тіркелген / Animal already registered",
            existing_id=existing.animal_id,
            similarity=existing.similarity,
        )

    assert outcome.embedding is not None and outcome.detection is not None
    species = outcome.detection.species
    animal_id = generate_animal_id(species)
    vector_db.add(species, animal_id, outcome.embedding)
    logger.info("registered animal_id=%s species=%s", animal_id, species.value)
    return RegistrationOutcome(
        animal_id=animal_id,
        species=species,
        embedding=outcome.embedding,
        detection=outcome.detection,
    )
