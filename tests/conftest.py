"""Pytest fixtures: tmp DB, FAISS, mocked ML pipeline."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect DB + FAISS + model paths to a fresh temp dir for every test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "vec"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "ml"))

    # Reset cached settings + module-level singletons
    from app import config as cfg

    cfg.get_settings.cache_clear()

    # Reset DB engine to the new URL
    import importlib

    from app.database import session as session_mod

    importlib.reload(session_mod)

    # Reset vector DB singleton
    from app.database import vector_db as vec_mod

    vec_mod._manager = None  # type: ignore[attr-defined]

    yield

    cfg.get_settings.cache_clear()


@pytest.fixture
def fake_image_bytes() -> bytes:
    """Return a tiny valid PNG file in-memory."""
    img = Image.new("RGB", (32, 32), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Replace YOLO + ResNet50 with deterministic fakes.

    Returns a small dict letting tests tweak the fake species / confidence /
    embedding for individual cases.
    """
    from app.ai import pipeline as pipeline_mod
    from app.ai.species_classifier import SpeciesDetection
    from app.database.models import Species

    state: dict[str, object] = {
        "species": Species.CATTLE,
        "confidence": 0.9,
        "embedding": None,  # set per test
    }

    class FakeClassifier:
        def detect(self, image):  # noqa: ANN001
            return SpeciesDetection(
                species=state["species"],  # type: ignore[arg-type]
                confidence=float(state["confidence"]),  # type: ignore[arg-type]
                bbox=(0, 0, 10, 10),
                cropped_image=np.zeros((10, 10, 3), dtype=np.uint8),
            )

    class FakeExtractor:
        def extract(self, crop):  # noqa: ANN001
            emb = state["embedding"]
            if emb is None:
                # Deterministic per-call vector
                rng = np.random.default_rng(42)
                emb = rng.standard_normal(512).astype(np.float32)
            emb = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(emb))
            return emb / n if n > 0 else emb

    monkeypatch.setattr(pipeline_mod, "get_species_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(pipeline_mod, "get_feature_extractor", lambda: FakeExtractor())
    return state


@pytest.fixture
def client(fake_pipeline):  # noqa: ARG001 - ensures ML stays mocked
    """FastAPI TestClient with isolated DB + mocked ML."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    """Yield a request-scoped SQLAlchemy session, rolling back after the test.

    Depends on the ``_isolated_env`` autouse fixture to point the engine at a
    fresh temp database.
    """
    from app.database.session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def tmp_vector_store(tmp_path: Path):
    """Return a fresh :class:`~app.database.vector_db.VectorDBManager`.

    Backed by a temporary directory so tests are fully isolated.
    Uses 768-dimensional vectors to exercise non-default dimensionality.
    """
    from app.database.vector_db import VectorDBManager

    return VectorDBManager(storage_dir=tmp_path / "testvec", dim=768)
