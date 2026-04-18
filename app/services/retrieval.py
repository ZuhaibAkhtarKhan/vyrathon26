"""Read-only queries for grabs, images, and their relationships."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Grab, Image, ImageGrab
from app.utils.errors import NotFoundError


def get_image(db: Session, image_id: str) -> Image:
    img = (
        db.query(Image)
        .options(selectinload(Image.grab_links))
        .filter(Image.id == image_id)
        .one_or_none()
    )
    if img is None:
        raise NotFoundError(f"Image {image_id} not found.")
    return img


def get_image_with_faces(db: Session, image_id: str) -> Image:
    img = (
        db.query(Image)
        .options(selectinload(Image.faces), selectinload(Image.grab_links))
        .filter(Image.id == image_id)
        .one_or_none()
    )
    if img is None:
        raise NotFoundError(f"Image {image_id} not found.")
    return img


def list_images(db: Session, *, limit: int = 100, offset: int = 0) -> list[Image]:
    return (
        db.query(Image)
        .options(selectinload(Image.grab_links))
        .order_by(Image.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_grab(db: Session, grab_id: str) -> Grab:
    g = db.query(Grab).filter(Grab.id == grab_id).one_or_none()
    if g is None:
        raise NotFoundError(f"Grab {grab_id} not found.")
    return g


def list_grabs(db: Session, *, limit: int = 100, offset: int = 0) -> list[Grab]:
    return (
        db.query(Grab)
        .order_by(Grab.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def images_for_grab(
    db: Session, grab_id: str, *, limit: int = 500, offset: int = 0
) -> list[Image]:
    """All images in which ``grab_id`` appears."""
    # Verify grab exists so we can 404 instead of silently returning [].
    get_grab(db, grab_id)

    stmt = (
        select(Image)
        .options(selectinload(Image.grab_links))
        .join(ImageGrab, ImageGrab.image_id == Image.id)
        .where(ImageGrab.grab_id == grab_id)
        .order_by(Image.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())
