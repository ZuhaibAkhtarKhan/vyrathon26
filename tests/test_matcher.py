"""Unit tests for the identity-clustering matcher."""

from __future__ import annotations

import numpy as np

from app.models import Grab
from app.services.matcher import assign_or_create_grab, find_best_grab
from tests.conftest import make_embedding


def test_first_face_creates_new_grab(db_session):
    emb = make_embedding(seed=1)
    result = assign_or_create_grab(db_session, emb)
    db_session.commit()

    assert result.is_new is True
    assert result.similarity == 1.0
    stored = db_session.query(Grab).all()
    assert len(stored) == 1
    assert stored[0].face_count == 1
    assert len(stored[0].centroid) == 128


def test_same_embedding_joins_existing_grab(db_session):
    emb = make_embedding(seed=42)
    first = assign_or_create_grab(db_session, emb)
    db_session.commit()

    second = assign_or_create_grab(db_session, emb)
    db_session.commit()

    assert second.is_new is False
    assert second.grab_id == first.grab_id
    assert second.similarity > 0.99

    stored = db_session.query(Grab).all()
    assert len(stored) == 1
    assert stored[0].face_count == 2


def test_dissimilar_embedding_creates_distinct_grab(db_session):
    a = make_embedding(seed=1)
    b = make_embedding(seed=99)  # very likely orthogonal-ish

    first = assign_or_create_grab(db_session, a)
    second = assign_or_create_grab(db_session, b)
    db_session.commit()

    assert first.grab_id != second.grab_id
    assert db_session.query(Grab).count() == 2


def test_centroid_is_running_mean(db_session):
    base = make_embedding(seed=7)
    noisy = base + 0.01 * make_embedding(seed=8)
    noisy = (noisy / np.linalg.norm(noisy)).astype(np.float32)

    assign_or_create_grab(db_session, base, threshold=0.5)
    assign_or_create_grab(db_session, noisy, threshold=0.5)
    db_session.commit()

    grabs = db_session.query(Grab).all()
    assert len(grabs) == 1
    centroid = np.asarray(grabs[0].centroid, dtype=np.float32)
    # Near unit length.
    assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-4


def test_find_best_grab_returns_none_when_empty(db_session):
    assert find_best_grab(db_session, make_embedding(seed=0)) is None


def test_find_best_grab_respects_threshold(db_session):
    emb = make_embedding(seed=10)
    assign_or_create_grab(db_session, emb)
    db_session.commit()

    # Very similar probe → hit.
    hit = find_best_grab(db_session, emb, threshold=0.5)
    assert hit is not None and hit.similarity > 0.9

    # Completely different probe → miss.
    miss = find_best_grab(
        db_session, make_embedding(seed=999), threshold=0.9
    )
    assert miss is None
