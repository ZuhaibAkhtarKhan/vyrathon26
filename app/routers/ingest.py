"""Ingestion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.deps import read_image_upload
from app.schemas import (
    IngestImageResponse,
    IngestResult,
    IngestScanRequest,
    IngestScanResponse,
)
from app.services.ingestion import IngestionService
from app.utils.errors import NotFoundError

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _outcome_to_schema(outcome) -> IngestResult:
    return IngestResult(
        image_id=outcome.image_id,
        filename=outcome.filename,
        faces_detected=outcome.faces_detected,
        grab_ids=outcome.grab_ids,
        is_new_image=outcome.is_new_image,
        new_grabs_created=outcome.new_grabs_created,
    )


@router.post(
    "/scan",
    response_model=IngestScanResponse,
    summary="Crawl a directory and ingest every image in it",
)
def scan(
    payload: IngestScanRequest = Body(default_factory=IngestScanRequest),
    db: Session = Depends(get_db),
) -> IngestScanResponse:
    settings = get_settings()
    target = Path(payload.directory) if payload.directory else settings.storage_dir
    if not target.exists():
        raise NotFoundError(
            f"Directory {target} does not exist.",
            details={"directory": str(target)},
        )
    service = IngestionService()
    summary = service.scan_directory(db, target, recursive=payload.recursive)

    return IngestScanResponse(
        directory=summary.directory,
        scanned=summary.scanned,
        ingested=summary.ingested,
        skipped_existing=summary.skipped_existing,
        skipped_no_faces=summary.skipped_no_faces,
        failed=summary.failed,
        total_faces=summary.total_faces,
        new_grabs_created=summary.new_grabs_created,
        results=[_outcome_to_schema(r) for r in summary.results],
    )


@router.post(
    "/image",
    response_model=IngestImageResponse,
    summary="Ingest a single uploaded image",
)
async def ingest_image(
    file: UploadFile = File(..., description="A single image file."),
    db: Session = Depends(get_db),
) -> IngestImageResponse:
    data, filename = await read_image_upload(file)
    service = IngestionService()
    outcome = service.ingest_bytes(db, data=data, filename=filename)
    db.commit()
    return IngestImageResponse(result=_outcome_to_schema(outcome))
