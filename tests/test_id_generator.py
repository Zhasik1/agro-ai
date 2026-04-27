"""Tests for the animal ID generator."""

from __future__ import annotations

import re

from app.database.models import Species
from app.utils.ids import SPECIES_PREFIX, generate_animal_id

_ID_PATTERN = re.compile(r"^(COW|SHP|HRS)-[A-F0-9]{6}$")


def test_id_format_cattle() -> None:
    animal_id = generate_animal_id(Species.CATTLE)
    assert _ID_PATTERN.match(animal_id), f"Invalid format: {animal_id}"


def test_id_format_sheep() -> None:
    animal_id = generate_animal_id(Species.SHEEP)
    assert _ID_PATTERN.match(animal_id), f"Invalid format: {animal_id}"


def test_id_format_horse() -> None:
    animal_id = generate_animal_id(Species.HORSE)
    assert _ID_PATTERN.match(animal_id), f"Invalid format: {animal_id}"


def test_species_prefix_mapping() -> None:
    """SPECIES_PREFIX must map each species to the correct prefix."""
    assert SPECIES_PREFIX[Species.CATTLE] == "COW"
    assert SPECIES_PREFIX[Species.SHEEP] == "SHP"
    assert SPECIES_PREFIX[Species.HORSE] == "HRS"


def test_prefix_in_generated_id() -> None:
    """Generated IDs must start with the species prefix."""
    for species in Species:
        animal_id = generate_animal_id(species)
        prefix = animal_id.split("-")[0]
        assert prefix == SPECIES_PREFIX[species]


def test_uniqueness_1000() -> None:
    """1000 generated cattle IDs must all be distinct."""
    ids = {generate_animal_id(Species.CATTLE) for _ in range(1000)}
    assert len(ids) == 1000
