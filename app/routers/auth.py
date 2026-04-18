"""Selfie authentication endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import read_image_upload
from app.schemas import AuthSelfieResponse
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/selfie",
    response_model=AuthSelfieResponse,
    summary="Authenticate a user via a selfie",
    description=(
        "Upload a selfie. The service detects your face, embeds it, and "
        "looks it up against every known `grab_id`. On a successful match "
        "the returned `grab_id` acts as the authorisation token for "
        "`GET /grabs/{grab_id}/images`."
    ),
)
async def selfie(
    file: UploadFile = File(..., description="A selfie image of the user."),
    db: Session = Depends(get_db),
) -> AuthSelfieResponse:
    data, _ = await read_image_upload(file)
    service = AuthService()
    outcome = service.authenticate_selfie(db, data=data)
    return AuthSelfieResponse(
        grab_id=outcome.grab_id,
        similarity=outcome.similarity,
        confidence=outcome.confidence,
        faces_in_selfie=outcome.faces_in_selfie,
    )
