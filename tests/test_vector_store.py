"""Tests for the FAISS-backed VectorDBManager."""

from __future__ import annotations

import numpy as np

from app.database.models import Species
from app.database.vector_db import VectorDBManager


def _unit_vec(dim: int, seed: int) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestAddSearchRoundTrip:
    def test_exact_match_similarity(self, tmp_vector_store: VectorDBManager) -> None:
        """Querying with the same vector that was added must return similarity ≈ 1."""
        v = _unit_vec(768, seed=0)
        tmp_vector_store.add(Species.CATTLE, "COW-000001", v)
        results = tmp_vector_store.search(Species.CATTLE, v, top_k=1)
        assert len(results) == 1
        assert abs(results[0].similarity - 1.0) < 1e-5

    def test_animal_id_preserved(self, tmp_vector_store: VectorDBManager) -> None:
        v = _unit_vec(768, seed=1)
        tmp_vector_store.add(Species.CATTLE, "COW-ABCDEF", v)
        results = tmp_vector_store.search(Species.CATTLE, v, top_k=1)
        assert results[0].animal_id == "COW-ABCDEF"

    def test_empty_store_returns_no_results(self, tmp_vector_store: VectorDBManager) -> None:
        q = _unit_vec(768, seed=2)
        results = tmp_vector_store.search(Species.CATTLE, q, top_k=5)
        assert results == []


class TestPersistence:
    def test_entries_survive_reload(self, tmp_path) -> None:
        """Entries added to a store must still be searchable after reload."""
        storage = tmp_path / "persist_vec"
        v = _unit_vec(768, seed=10)

        store1 = VectorDBManager(storage_dir=storage, dim=768)
        store1.add(Species.CATTLE, "COW-PERSIST", v)

        store2 = VectorDBManager(storage_dir=storage, dim=768)
        results = store2.search(Species.CATTLE, v, top_k=1)
        assert len(results) == 1
        assert results[0].animal_id == "COW-PERSIST"
        assert abs(results[0].similarity - 1.0) < 1e-5


class TestSpeciesIsolation:
    def test_cattle_entry_not_in_sheep(self, tmp_vector_store: VectorDBManager) -> None:
        """An embedding added to cattle must NOT appear in a sheep search."""
        v = _unit_vec(768, seed=20)
        tmp_vector_store.add(Species.CATTLE, "COW-ISOL01", v)
        sheep_results = tmp_vector_store.search(Species.SHEEP, v, top_k=5)
        assert sheep_results == []

    def test_sheep_entry_not_in_horse(self, tmp_vector_store: VectorDBManager) -> None:
        v = _unit_vec(768, seed=21)
        tmp_vector_store.add(Species.SHEEP, "SHP-ISOL01", v)
        horse_results = tmp_vector_store.search(Species.HORSE, v, top_k=5)
        assert horse_results == []


class TestTopK:
    def test_top1_is_query_itself(self, tmp_vector_store: VectorDBManager) -> None:
        """When query equals one of the stored vectors, top-1 similarity must be ≈ 1."""
        vectors = [_unit_vec(768, seed=i) for i in range(5)]
        for i, v in enumerate(vectors):
            tmp_vector_store.add(Species.SHEEP, f"SHP-{i:06d}", v)

        results = tmp_vector_store.search(Species.SHEEP, vectors[2], top_k=5)
        assert results[0].animal_id == "SHP-000002"
        assert abs(results[0].similarity - 1.0) < 1e-5

    def test_top_k_respects_limit(self, tmp_vector_store: VectorDBManager) -> None:
        for i in range(10):
            v = _unit_vec(768, seed=100 + i)
            tmp_vector_store.add(Species.HORSE, f"HRS-{i:06d}", v)
        results = tmp_vector_store.search(Species.HORSE, _unit_vec(768, seed=0), top_k=3)
        assert len(results) <= 3
