"""Grab (identity) endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import BBox, FaceOut, GrabImagesOut, GrabOut, ImageOut
from app.services import retrieval

router = APIRouter(prefix="/grabs", tags=["grabs"])


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


@router.get(
    "",
    response_model=list[GrabOut],
    summary="List all known grab identities",
)
def list_grabs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[GrabOut]:
    return [
        GrabOut.model_validate(g) for g in retrieval.list_grabs(db, limit=limit, offset=offset)
    ]


@router.get(
    "/{grab_id}",
    response_model=GrabOut,
    summary="Fetch a single grab by id",
)
def get_grab(grab_id: str, db: Session = Depends(get_db)) -> GrabOut:
    return GrabOut.model_validate(retrieval.get_grab(db, grab_id))


@router.get(
    "/{grab_id}/images",
    response_model=GrabImagesOut,
    summary="List every image in which this grab appears",
    description=(
        "Main data-extraction endpoint: given a `grab_id` (e.g. the one "
        "returned by `/auth/selfie`), return all images containing that face."
    ),
)
def grab_images(
    grab_id: str,
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> GrabImagesOut:
    images = retrieval.images_for_grab(db, grab_id, limit=limit, offset=offset)
    return GrabImagesOut(
        grab_id=grab_id,
        count=len(images),
        images=[_image_to_schema(i) for i in images],
    )


@router.get(
    "/{grab_id}/faces",
    response_model=list[FaceOut],
    summary="List all face instances belonging to a grab (debug)",
)
def grab_faces(grab_id: str, db: Session = Depends(get_db)) -> list[FaceOut]:
    grab = retrieval.get_grab(db, grab_id)
    return [
        FaceOut(
            id=f.id,
            grab_id=f.grab_id,
            bbox=BBox(x=f.bbox_x, y=f.bbox_y, w=f.bbox_w, h=f.bbox_h),
            detection_confidence=f.detection_confidence,
        )
        for f in grab.faces
    ]
