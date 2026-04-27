"""Tests for the SQLAlchemy CRUD repository helpers."""

from __future__ import annotations

from app.database.models import Species
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


class TestOwnerCRUD:
    def test_create_and_fetch(self, db_session) -> None:
        owner = create_owner(db_session, "880101300123", full_name="Test Owner")
        assert owner.iin == "880101300123"
        assert owner.full_name == "Test Owner"

        fetched = get_owner(db_session, "880101300123")
        assert fetched is not None
        assert fetched.iin == "880101300123"

    def test_get_missing_returns_none(self, db_session) -> None:
        assert get_owner(db_session, "000000000000") is None

    def test_optional_fields_default_none(self, db_session) -> None:
        owner = create_owner(db_session, "880101300124")
        assert owner.phone is None
        assert owner.region is None


class TestAnimalCRUD:
    def test_create_and_fetch(self, db_session) -> None:
        create_owner(db_session, "880101300123")
        animal = create_animal(
            db_session,
            "COW-AAAAAA",
            Species.CATTLE,
            "880101300123",
            breed="Holstein",
            age_years=3,
            weight_kg=500.0,
            photo_sha256="abc123",
        )
        assert animal.id == "COW-AAAAAA"
        assert animal.species == Species.CATTLE

        fetched = get_animal(db_session, "COW-AAAAAA")
        assert fetched is not None
        assert fetched.breed == "Holstein"

    def test_get_missing_returns_none(self, db_session) -> None:
        assert get_animal(db_session, "COW-DEADBE") is None


class TestListAnimals:
    def test_list_by_owner_returns_only_that_owners_animals(self, db_session) -> None:
        create_owner(db_session, "880101300123")
        create_owner(db_session, "880101300456")
        create_animal(
            db_session,
            "COW-OWN001",
            Species.CATTLE,
            "880101300123",
            breed=None,
            age_years=None,
            weight_kg=None,
            photo_sha256="h1",
        )
        create_animal(
            db_session,
            "COW-OWN002",
            Species.CATTLE,
            "880101300456",
            breed=None,
            age_years=None,
            weight_kg=None,
            photo_sha256="h2",
        )
        animals = list_animals_by_owner(db_session, "880101300123")
        assert len(animals) == 1
        assert animals[0].id == "COW-OWN001"

    def test_list_by_owner_returns_empty_for_unknown(self, db_session) -> None:
        result = list_animals_by_owner(db_session, "000000000000")
        assert result == []


class TestDeleteAnimal:
    def test_delete_existing_returns_true(self, db_session) -> None:
        create_owner(db_session, "880101300123")
        create_animal(
            db_session,
            "COW-DEL001",
            Species.CATTLE,
            "880101300123",
            breed=None,
            age_years=None,
            weight_kg=None,
            photo_sha256="hd1",
        )
        result = delete_animal(db_session, "COW-DEL001")
        assert result is True
        assert get_animal(db_session, "COW-DEL001") is None

    def test_delete_missing_returns_false(self, db_session) -> None:
        result = delete_animal(db_session, "COW-NOPE01")
        assert result is False


class TestCountAndRecent:
    def test_count_by_species(self, db_session) -> None:
        create_owner(db_session, "880101300123")
        create_animal(
            db_session,
            "COW-CNT01",
            Species.CATTLE,
            "880101300123",
            breed=None,
            age_years=None,
            weight_kg=None,
            photo_sha256="hc1",
        )
        create_animal(
            db_session,
            "SHP-CNT01",
            Species.SHEEP,
            "880101300123",
            breed=None,
            age_years=None,
            weight_kg=None,
            photo_sha256="hc2",
        )
        counts = count_by_species(db_session)
        assert counts[Species.CATTLE] == 1
        assert counts[Species.SHEEP] == 1
        assert counts[Species.HORSE] == 0

    def test_list_recent_respects_limit(self, db_session) -> None:
        create_owner(db_session, "880101300123")
        for i in range(5):
            create_animal(
                db_session,
                f"COW-REC0{i}",
                Species.CATTLE,
                "880101300123",
                breed=None,
                age_years=None,
                weight_kg=None,
                photo_sha256=f"hr{i}",
            )
        recent = list_recent(db_session, limit=3)
        assert len(recent) == 3
