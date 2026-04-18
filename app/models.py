"""ORM models. See `docs/schema.md` for the full schema description."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CHAR,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JSONEncodedVector(TypeDecorator):
    """Store a `list[float]` as a JSON string.

    Portable across Postgres and SQLite. On Postgres + pgvector this is the
    only class that needs to be swapped with `sqlalchemy_pgvector.Vector(128)`
    to enable native vector search.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "tolist"):  # numpy array
            value = value.tolist()
        return json.dumps(list(value))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return json.loads(value)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False, unique=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    faces: Mapped[list["Face"]] = relationship(
        "Face", back_populates="image", cascade="all, delete-orphan"
    )
    grab_links: Mapped[list["ImageGrab"]] = relationship(
        "ImageGrab", back_populates="image", cascade="all, delete-orphan"
    )


class Grab(Base):
    __tablename__ = "grabs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    centroid: Mapped[list[float]] = mapped_column(JSONEncodedVector, nullable=False)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
    )

    faces: Mapped[list["Face"]] = relationship("Face", back_populates="grab")
    image_links: Mapped[list["ImageGrab"]] = relationship(
        "ImageGrab", back_populates="grab", cascade="all, delete-orphan"
    )


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    grab_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("grabs.id", ondelete="SET NULL"), nullable=True, index=True
    )

    embedding: Mapped[list[float]] = mapped_column(JSONEncodedVector, nullable=False)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    image: Mapped[Image] = relationship("Image", back_populates="faces")
    grab: Mapped[Grab | None] = relationship("Grab", back_populates="faces")


class ImageGrab(Base):
    """Materialised many-to-many link between images and grabs.

    Redundant relative to `faces`, but gives us O(1)-indexed "which images
    does this grab appear in?" lookups and matches the schema described in
    the problem statement ("map one image to many grab_ids").
    """

    __tablename__ = "image_grabs"
    __table_args__ = (UniqueConstraint("image_id", "grab_id", name="uq_image_grab"),)

    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    grab_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("grabs.id", ondelete="CASCADE"), primary_key=True
    )
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    image: Mapped[Image] = relationship("Image", back_populates="grab_links")
    grab: Mapped[Grab] = relationship("Grab", back_populates="image_links")
