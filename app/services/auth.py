"""Selfie authentication service."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.face_engine import FaceEngine, get_face_engine
from app.services.matcher import (
    LookupResult,
    find_best_grab,
    similarity_to_confidence,
)
from app.services.storage import decode_image
from app.utils.errors import NoFaceDetectedError, NoMatchError


@dataclass
class AuthOutcome:
    grab_id: str
    similarity: float
    confidence: str
    faces_in_selfie: int


class AuthService:
    """1:N selfie → grab_id lookup."""

    def __init__(self, face_engine: FaceEngine | None = None) -> None:
        self.face_engine = face_engine or get_face_engine()

    def authenticate_selfie(self, db: Session, *, data: bytes) -> AuthOutcome:
        image = decode_image(data)
        detected = self.face_engine.detect_and_embed(image)
        if not detected:
            raise NoFaceDetectedError(
                "No face detected in the uploaded selfie. Please retake with "
                "good lighting and the face fully visible."
            )

        # Pick the largest face (most likely the "selfie subject") with a
        # tiebreaker on detector confidence.
        primary = max(
            detected, key=lambda f: (f.area, f.detection_confidence)
        )

        settings = get_settings()
        lookup: LookupResult | None = find_best_grab(db, primary.embedding)
        if lookup is None:
            raise NoMatchError(
                "No matching identity found. You may not have been "
                "photographed at this event yet, or image quality is too low.",
                details={"faces_in_selfie": len(detected)},
            )

        return AuthOutcome(
            grab_id=lookup.grab_id,
            similarity=lookup.similarity,
            confidence=similarity_to_confidence(
                lookup.similarity, settings.face_match_threshold
            ),
            faces_in_selfie=len(detected),
        )
