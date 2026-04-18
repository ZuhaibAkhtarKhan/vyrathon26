"""Identity matching: compare face embeddings against known grab centroids.

Performs two closely-related jobs:

* :func:`assign_or_create_grab` — used during ingestion. Given a new face
  embedding, find the best-matching existing grab or create a fresh one,
  and keep the centroid up to date via a running mean.
* :func:`find_best_grab` — used by selfie authentication. Pure read; returns
  the best match above the configured threshold or ``None``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Grab
from app.services.face_engine import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    grab_id: str
    similarity: float
    is_new: bool


@dataclass
class LookupResult:
    grab_id: str
    similarity: float


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def _best_match(embedding: np.ndarray, grabs: list[Grab]) -> tuple[Grab | None, float]:
    """Return the single highest-similarity grab and its cosine score."""
    best: Grab | None = None
    best_score = -1.0
    emb = _l2_normalise(np.asarray(embedding, dtype=np.float32))
    for grab in grabs:
        centroid = np.asarray(grab.centroid, dtype=np.float32)
        score = cosine_similarity(emb, centroid)
        if score > best_score:
            best = grab
            best_score = score
    return best, best_score


def assign_or_create_grab(
    db: Session,
    embedding: np.ndarray,
    *,
    threshold: float | None = None,
) -> MatchResult:
    """Online single-pass clustering against grab centroids.

    * If the best existing grab's cosine similarity ≥ threshold, assign to it
      and incrementally update the centroid (running mean, re-normalised).
    * Otherwise create a new grab whose centroid is this embedding.
    """
    settings = get_settings()
    threshold = settings.face_match_threshold if threshold is None else threshold

    grabs = db.query(Grab).all()
    best, score = _best_match(embedding, grabs)

    if best is not None and score >= threshold:
        _update_centroid(best, embedding)
        best.face_count += 1
        best.updated_at = datetime.now(timezone.utc)
        db.flush()
        return MatchResult(grab_id=best.id, similarity=score, is_new=False)

    emb = _l2_normalise(np.asarray(embedding, dtype=np.float32))
    new_grab = Grab(
        centroid=emb.tolist(),
        face_count=1,
    )
    db.add(new_grab)
    db.flush()  # populate `id`
    logger.info("Created new grab %s (best prior score %.3f)", new_grab.id, score)
    return MatchResult(grab_id=new_grab.id, similarity=1.0, is_new=True)


def _update_centroid(grab: Grab, embedding: np.ndarray) -> None:
    """Running-mean update, re-normalised to unit length."""
    old = np.asarray(grab.centroid, dtype=np.float32)
    n = grab.face_count
    new_emb = _l2_normalise(np.asarray(embedding, dtype=np.float32))
    updated = (old * n + new_emb) / (n + 1)
    grab.centroid = _l2_normalise(updated).tolist()


def find_best_grab(
    db: Session,
    embedding: np.ndarray,
    *,
    threshold: float | None = None,
) -> LookupResult | None:
    """Read-only 1:N lookup used by selfie authentication."""
    settings = get_settings()
    threshold = settings.face_match_threshold if threshold is None else threshold

    grabs = db.query(Grab).all()
    if not grabs:
        return None
    best, score = _best_match(embedding, grabs)
    if best is None or score < threshold:
        return None
    return LookupResult(grab_id=best.id, similarity=score)


def similarity_to_confidence(score: float, threshold: float) -> str:
    """Bucket a cosine-similarity score to a human-readable confidence."""
    if score >= min(1.0, threshold + 0.25):
        return "high"
    if score >= threshold + 0.10:
        return "medium"
    return "low"
