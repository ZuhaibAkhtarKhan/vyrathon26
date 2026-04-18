"""Liveness / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.config import get_settings
from app.database import get_db
from app.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(db: Session = Depends(get_db)) -> HealthResponse:
    settings = get_settings()
    try:
        db.execute(text("SELECT 1"))
        db_state = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        db_state = f"error: {exc}"
    return HealthResponse(
        status="ok",
        version=__version__,
        face_engine=settings.face_engine,
        database=db_state,
    )
