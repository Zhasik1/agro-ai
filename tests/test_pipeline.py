"""Pipeline threshold + exception logic with mocked components."""

from __future__ import annotations

import numpy as np
import pytest

from app.ai import pipeline as pipeline_mod
from app.ai.pipeline import _classify_status, identify_animal, register_animal
from app.ai.species_classifier import SpeciesDetection
from app.config import get_settings
from app.database.models import Species
from app.database.vector_db import VectorMatch
from app.exceptions import AnimalNotDetectedError, DuplicateAnimalError


def test_classify_status_thresholds() -> None:
    settings = get_settings()
    assert _classify_status([], settings) == "new"
    assert (
        _classify_status([VectorMatch("X", settings.MATCH_THRESHOLD + 0.01)], settings) == "matched"
    )
    mid = (settings.MATCH_THRESHOLD + settings.SUSPECT_THRESHOLD) / 2
    assert _classify_status([VectorMatch("X", mid)], settings) == "suspect"
    assert (
        _classify_status([VectorMatch("X", settings.SUSPECT_THRESHOLD - 0.01)], settings) == "new"
    )


class _FakeClassifier:
    def __init__(self, raise_exc: bool = False) -> None:
        self._raise = raise_exc

    def detect(self, image):  # noqa: ANN001
        if self._raise:
            raise AnimalNotDetectedError("nope")
        return SpeciesDetection(
            species=Species.SHEEP,
            confidence=0.95,
            bbox=(0, 0, 10, 10),
            cropped_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )


class _FakeExtractor:
    def extract(self, crop):  # noqa: ANN001
        v = np.ones(512, dtype=np.float32)
        return v / np.linalg.norm(v)


class _FakeVecDB:
    def __init__(self, matches: list[VectorMatch]) -> None:
        self._matches = matches
        self.added: list[tuple[Species, str]] = []

    def search(self, species, query, top_k=3):  # noqa: ANN001
        return self._matches

    def add(self, species, animal_id, embedding) -> None:  # noqa: ANN001
        self.added.append((species, animal_id))


def test_identify_propagates_no_animal() -> None:
    with pytest.raises(AnimalNotDetectedError):
        identify_animal(
            np.zeros((10, 10, 3), dtype=np.uint8),
            classifier=_FakeClassifier(raise_exc=True),
            extractor=_FakeExtractor(),
            vector_db=_FakeVecDB([]),
        )


def test_register_rejects_duplicate() -> None:
    with pytest.raises(DuplicateAnimalError) as exc:
        register_animal(
            np.zeros((10, 10, 3), dtype=np.uint8),
            classifier=_FakeClassifier(),
            extractor=_FakeExtractor(),
            vector_db=_FakeVecDB([VectorMatch("SHP-AAAAAA", 0.99)]),
        )
    assert exc.value.existing_id == "SHP-AAAAAA"


def test_register_writes_to_vector_db() -> None:
    db = _FakeVecDB([])
    outcome = register_animal(
        np.zeros((10, 10, 3), dtype=np.uint8),
        classifier=_FakeClassifier(),
        extractor=_FakeExtractor(),
        vector_db=db,
    )
    assert outcome.species == Species.SHEEP
    assert outcome.animal_id.startswith("SHP-")
    assert db.added and db.added[0][0] == Species.SHEEP
