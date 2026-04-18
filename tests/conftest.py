"""Shared pytest fixtures.

* Swap the real DB engine for an in-memory SQLite DB per test.
* Swap the real face engine for a :class:`ScriptedFaceEngine` we can program.
* Expose a ready-to-use ``TestClient``.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Iterator

# Make sure we boot in "stub" mode with an in-memory DB **before** importing
# the app — `get_settings()` is cached on first import.
os.environ.setdefault("FACE_ENGINE", "stub")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import database as db_module  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import face_engine as fe_module  # noqa: E402
from app.services.face_engine import DetectedFace, FaceEngine  # noqa: E402


# --------- Programmable face engine for deterministic tests ---------


class ScriptedFaceEngine(FaceEngine):
    """Return pre-programmed faces keyed by decoded-image hash.

    Tests call :meth:`program` to say "when the pipeline decodes this image,
    pretend you found these faces". Embeddings are plain numpy arrays —
    identical arrays for two different images will cluster together, distinct
    arrays will not.
    """

    def __init__(self) -> None:
        self._script: dict[str, list[DetectedFace]] = {}
        self.default: list[DetectedFace] = []

    @staticmethod
    def _key(image_bgr: np.ndarray) -> str:
        return hashlib.sha256(image_bgr.tobytes()).hexdigest()

    def program_for_array(
        self, image_bgr: np.ndarray, faces: list[DetectedFace]
    ) -> None:
        self._script[self._key(image_bgr)] = faces

    def program_for_bytes(self, data: bytes, faces: list[DetectedFace]) -> None:
        """Program by file bytes — we decode exactly like the real pipeline."""
        from app.services.storage import decode_image

        image_bgr = decode_image(data)
        self.program_for_array(image_bgr, faces)

    def detect_and_embed(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        return list(self._script.get(self._key(image_bgr), self.default))


# --------- Fixtures ---------


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch):
    """Fresh on-disk SQLite DB per test (in-memory doesn't share between threads)."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, connect_args={"check_same_thread": False}, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    # Point the app at this engine.
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", SessionLocal)

    # (Re)create schema.
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield SessionLocal

    engine.dispose()


@pytest.fixture
def face_engine(monkeypatch) -> ScriptedFaceEngine:
    engine = ScriptedFaceEngine()
    monkeypatch.setattr(fe_module, "_engine_singleton", engine)
    return engine


@pytest.fixture
def client(test_db, face_engine) -> Iterator[TestClient]:
    """TestClient wired to the test DB + scripted face engine."""

    def _get_test_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def db_session(test_db):
    """Raw SQLAlchemy Session for service-level unit tests."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


# --------- Image / embedding helpers ---------


def make_image_bytes(color: tuple[int, int, int], size: int = 64) -> bytes:
    """Synthesize a solid-colour PNG. Each colour → a unique content hash."""
    img = PILImage.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_embedding(seed: int, dim: int = 128) -> np.ndarray:
    """Deterministic L2-normalised 128-d vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_face(
    embedding: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] = (0, 0, 64, 64),
    confidence: float = 0.99,
) -> DetectedFace:
    return DetectedFace(
        bbox=bbox, detection_confidence=confidence, embedding=embedding
    )
