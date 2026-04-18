"""Unit tests for the face engine utility layer."""

from __future__ import annotations

import numpy as np

from app.services.face_engine import (
    StubFaceEngine,
    _l2_normalise,
    cosine_similarity,
)
from tests.conftest import make_image_bytes
from app.services.storage import decode_image


def test_l2_normalise_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = _l2_normalise(v)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_l2_normalise_handles_zero_vector():
    v = np.zeros(4, dtype=np.float32)
    out = _l2_normalise(v)
    assert np.allclose(out, 0.0)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


def test_stub_engine_same_image_same_embedding():
    engine = StubFaceEngine()
    img = decode_image(make_image_bytes((120, 40, 200)))
    faces_a = engine.detect_and_embed(img)
    faces_b = engine.detect_and_embed(img)
    assert len(faces_a) == 1 and len(faces_b) == 1
    assert np.allclose(faces_a[0].embedding, faces_b[0].embedding)


def test_stub_engine_different_images_different_embeddings():
    engine = StubFaceEngine()
    img1 = decode_image(make_image_bytes((10, 10, 10)))
    img2 = decode_image(make_image_bytes((250, 250, 250)))
    e1 = engine.detect_and_embed(img1)[0].embedding
    e2 = engine.detect_and_embed(img2)[0].embedding
    assert not np.allclose(e1, e2)
