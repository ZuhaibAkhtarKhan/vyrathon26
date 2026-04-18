"""Pydantic request / response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------- Common ----------


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorPayload


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    face_engine: str
    database: str


# ---------- Images ----------


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceOut(BaseModel):
    id: str
    grab_id: str | None
    bbox: BBox
    detection_confidence: float

    model_config = ConfigDict(from_attributes=True)


class ImageOut(BaseModel):
    id: str
    filename: str
    storage_path: str
    content_hash: str
    width: int
    height: int
    face_count: int
    created_at: datetime
    grab_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ImageWithFacesOut(ImageOut):
    faces: list[FaceOut] = Field(default_factory=list)


# ---------- Grabs ----------


class GrabOut(BaseModel):
    id: str
    label: str | None
    face_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GrabImagesOut(BaseModel):
    grab_id: str
    count: int
    images: list[ImageOut]


# ---------- Ingestion ----------


class IngestScanRequest(BaseModel):
    directory: str | None = Field(
        default=None,
        description=(
            "Absolute or relative path to crawl. Defaults to the configured "
            "STORAGE_DIR if omitted."
        ),
    )
    recursive: bool = True


class IngestResult(BaseModel):
    image_id: str
    filename: str
    faces_detected: int
    grab_ids: list[str]
    is_new_image: bool
    new_grabs_created: int


class IngestScanResponse(BaseModel):
    directory: str
    scanned: int
    ingested: int
    skipped_existing: int
    skipped_no_faces: int
    failed: int
    total_faces: int
    new_grabs_created: int
    results: list[IngestResult]


class IngestImageResponse(BaseModel):
    result: IngestResult


# ---------- Authentication ----------


class AuthSelfieResponse(BaseModel):
    grab_id: str
    similarity: float = Field(description="Cosine similarity to the matched grab centroid (-1..1).")
    confidence: str = Field(description="'low' | 'medium' | 'high'.")
    faces_in_selfie: int
    message: str = (
        "Authenticated. Use `grab_id` to retrieve your images at "
        "GET /grabs/{grab_id}/images."
    )
