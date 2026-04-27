"""FAISS-based vector store with one index per species.

Each species gets an :class:`faiss.IndexFlatIP` (inner product = cosine on
L2-normalised vectors). Indices and the parallel id list are persisted to::

    {VECTOR_DB_PATH}/{species}/index.faiss
    {VECTOR_DB_PATH}/{species}/ids.pkl

The store is process-safe via a per-species :class:`threading.Lock`.
"""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings
from app.database.models import Species

__all__ = ["VectorMatch", "VectorDBManager", "get_vector_db"]

logger = logging.getLogger(__name__)


@dataclass
class VectorMatch:
    """One nearest-neighbour result."""

    animal_id: str
    similarity: float


class _SpeciesIndex:
    """FAISS IndexFlatIP + parallel id list for a single species."""

    def __init__(self, species: Species, dim: int, storage_dir: Path) -> None:
        self.species = species
        self.dim = dim
        self.dir = storage_dir / species.value
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.ids_path = self.dir / "ids.pkl"
        self.lock = threading.Lock()
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.ids_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with self.ids_path.open("rb") as fh:
                    self.ids = pickle.load(fh)
                logger.info("Loaded FAISS index for %s (n=%d)", self.species.value, len(self.ids))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed loading index for %s: %s", self.species.value, exc)
                self.index = faiss.IndexFlatIP(self.dim)
                self.ids = []

    def persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.ids_path.open("wb") as fh:
            pickle.dump(self.ids, fh)

    def add(self, animal_id: str, embedding: np.ndarray) -> None:
        with self.lock:
            vec = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(vec)
            self.ids.append(animal_id)
            self.persist()

    def search(self, query: np.ndarray, top_k: int = 5) -> list[VectorMatch]:
        with self.lock:
            if self.index.ntotal == 0:
                return []
            q = query.astype(np.float32).reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(q, k)
        out: list[VectorMatch] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            out.append(VectorMatch(animal_id=self.ids[int(idx)], similarity=float(score)))
        return out

    def remove(self, animal_id: str) -> bool:
        with self.lock:
            if animal_id not in self.ids:
                return False
            keep_indices = [i for i, aid in enumerate(self.ids) if aid != animal_id]
            new_ids = [self.ids[i] for i in keep_indices]
            new_index = faiss.IndexFlatIP(self.dim)
            if keep_indices:
                # IndexFlatIP exposes reconstruct
                vectors = np.vstack([self.index.reconstruct(int(i)) for i in keep_indices]).astype(
                    np.float32
                )
                new_index.add(vectors)
            self.index = new_index
            self.ids = new_ids
            self.persist()
            return True

    def count(self) -> int:
        return int(self.index.ntotal)


class VectorDBManager:
    """Manages one :class:`_SpeciesIndex` per supported species."""

    def __init__(self, storage_dir: Path | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.VECTOR_DB_PATH)
        self.dim = dim or settings.EMBEDDING_DIM
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[Species, _SpeciesIndex] = {}
        self._global_lock = threading.Lock()

    def _get_index(self, species: Species) -> _SpeciesIndex:
        with self._global_lock:
            if species not in self._indices:
                self._indices[species] = _SpeciesIndex(species, self.dim, self.storage_dir)
            return self._indices[species]

    def add(self, species: Species, animal_id: str, embedding: np.ndarray) -> None:
        """Insert a new (id, embedding) pair into the species index."""
        self._get_index(species).add(animal_id, embedding)

    def search(self, species: Species, query: np.ndarray, top_k: int = 5) -> list[VectorMatch]:
        """Return up to ``top_k`` nearest matches in the species index."""
        return self._get_index(species).search(query, top_k=top_k)

    def remove(self, species: Species, animal_id: str) -> bool:
        """Remove an animal from its species index. Returns ``True`` if removed."""
        return self._get_index(species).remove(animal_id)

    def stats(self) -> dict[str, int]:
        """Return per-species record counts plus a ``total``."""
        out: dict[str, int] = {}
        total = 0
        for species in Species:
            count = self._get_index(species).count()
            out[species.value] = count
            total += count
        out["total"] = total
        return out

    def reset(self) -> None:
        """Wipe all indices (used by tests)."""
        for species in Species:
            idx = self._get_index(species)
            with idx.lock:
                idx.index = faiss.IndexFlatIP(self.dim)
                idx.ids = []
                idx.persist()


_lock = threading.Lock()
_manager: VectorDBManager | None = None


def get_vector_db() -> VectorDBManager:
    """Return the lazy-initialised global :class:`VectorDBManager`."""
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = VectorDBManager()
    return _manager
