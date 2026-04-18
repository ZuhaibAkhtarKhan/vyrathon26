"""Image endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import BBox, FaceOut, ImageOut, ImageWithFacesOut
from app.services import retrieval
from app.utils.errors import NotFoundError

router = APIRouter(prefix="/images", tags=["images"])


def _image_to_schema(img) -> ImageOut:
    return ImageOut(
        id=img.id,
        filename=img.filename,
        storage_path=img.storage_path,
        content_hash=img.content_hash,
        width=img.width,
        height=img.height,
        face_count=img.face_count,
        created_at=img.created_at,
        grab_ids=[link.grab_id for link in img.grab_links],
    )


@router.get("", response_model=list[ImageOut], summary="List ingested images")
def list_images(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[ImageOut]:
    return [_image_to_schema(i) for i in retrieval.list_images(db, limit=limit, offset=offset)]


@router.get(
    "/{image_id}",
    response_model=ImageWithFacesOut,
    summary="Fetch image metadata including the faces detected in it",
)
def get_image(image_id: str, db: Session = Depends(get_db)) -> ImageWithFacesOut:
    img = retrieval.get_image_with_faces(db, image_id)
    base = _image_to_schema(img)
    return ImageWithFacesOut(
        **base.model_dump(),
        faces=[
            FaceOut(
                id=f.id,
                grab_id=f.grab_id,
                bbox=BBox(x=f.bbox_x, y=f.bbox_y, w=f.bbox_w, h=f.bbox_h),
                detection_confidence=f.detection_confidence,
            )
            for f in img.faces
        ],
    )


@router.get(
    "/{image_id}/download",
    summary="Stream the original image bytes from storage",
    responses={
        200: {"content": {"image/*": {}}},
        404: {"description": "Image not found."},
    },
)
def download_image(image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    img = retrieval.get_image(db, image_id)
    path = Path(img.storage_path)
    if not path.exists():
        raise NotFoundError(
            "Image record exists but the underlying file is missing on disk.",
            details={"image_id": image_id, "storage_path": img.storage_path},
        )
    return FileResponse(path=str(path), filename=img.filename)
