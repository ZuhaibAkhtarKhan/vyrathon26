"""Ingestion pipeline: crawl storage → detect → embed → cluster → persist."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Face, Image, ImageGrab
from app.services.face_engine import FaceEngine, get_face_engine
from app.services.matcher import assign_or_create_grab
from app.services.storage import decode_image, iter_image_files
from app.utils.errors import UnsupportedMediaError
from app.utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    image_id: str
    filename: str
    faces_detected: int
    grab_ids: list[str]
    is_new_image: bool
    new_grabs_created: int


@dataclass
class ScanSummary:
    directory: str
    scanned: int = 0
    ingested: int = 0
    skipped_existing: int = 0
    skipped_no_faces: int = 0
    failed: int = 0
    total_faces: int = 0
    new_grabs_created: int = 0
    results: list[IngestOutcome] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []


class IngestionService:
    """Stateless orchestrator around the face engine and DB."""

    def __init__(self, face_engine: FaceEngine | None = None) -> None:
        self.face_engine = face_engine or get_face_engine()

    # ---------- public API ----------

    def ingest_bytes(
        self,
        db: Session,
        *,
        data: bytes,
        filename: str,
        storage_path: str | None = None,
    ) -> IngestOutcome:
        """Ingest a single image provided as raw bytes."""
        if not data:
            raise UnsupportedMediaError("Empty image payload.")

        content_hash = sha256_bytes(data)

        existing = (
            db.query(Image).filter(Image.content_hash == content_hash).one_or_none()
        )
        if existing is not None:
            grab_ids = [link.grab_id for link in existing.grab_links]
            return IngestOutcome(
                image_id=existing.id,
                filename=existing.filename,
                faces_detected=existing.face_count,
                grab_ids=grab_ids,
                is_new_image=False,
                new_grabs_created=0,
            )

        image_bgr = decode_image(data)
        detected = self.face_engine.detect_and_embed(image_bgr)

        storage_path = storage_path or self._materialise_upload(
            data=data, filename=filename, content_hash=content_hash
        )

        height, width = image_bgr.shape[:2]
        img = Image(
            filename=filename,
            storage_path=storage_path,
            content_hash=content_hash,
            width=width,
            height=height,
            face_count=len(detected),
        )
        db.add(img)
        db.flush()  # populate img.id

        grab_counts: Counter[str] = Counter()
        new_grabs = 0

        for face in detected:
            match = assign_or_create_grab(db, face.embedding)
            if match.is_new:
                new_grabs += 1
            db.add(
                Face(
                    image_id=img.id,
                    grab_id=match.grab_id,
                    embedding=face.embedding.tolist(),
                    bbox_x=int(face.bbox[0]),
                    bbox_y=int(face.bbox[1]),
                    bbox_w=int(face.bbox[2]),
                    bbox_h=int(face.bbox[3]),
                    detection_confidence=float(face.detection_confidence),
                )
            )
            grab_counts[match.grab_id] += 1

        for grab_id, count in grab_counts.items():
            db.add(ImageGrab(image_id=img.id, grab_id=grab_id, face_count=count))

        db.flush()
        return IngestOutcome(
            image_id=img.id,
            filename=img.filename,
            faces_detected=len(detected),
            grab_ids=list(grab_counts.keys()),
            is_new_image=True,
            new_grabs_created=new_grabs,
        )

    def ingest_path(self, db: Session, path: Path) -> IngestOutcome:
        """Ingest an on-disk image file without copying it."""
        data = path.read_bytes()
        return self.ingest_bytes(
            db,
            data=data,
            filename=path.name,
            storage_path=str(path.resolve()),
        )

    def scan_directory(
        self,
        db: Session,
        directory: Path,
        *,
        recursive: bool = True,
    ) -> ScanSummary:
        """Crawl a directory, ingest each image, commit once at the end."""
        summary = ScanSummary(directory=str(directory))
        files = iter_image_files(directory, recursive=recursive)
        summary.scanned = len(files)

        for file in files:
            try:
                outcome = self.ingest_path(db, file)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to ingest %s: %s", file, exc)
                summary.failed += 1
                continue

            summary.results.append(outcome)
            summary.total_faces += outcome.faces_detected
            summary.new_grabs_created += outcome.new_grabs_created
            if not outcome.is_new_image:
                summary.skipped_existing += 1
            elif outcome.faces_detected == 0:
                summary.skipped_no_faces += 1
                summary.ingested += 1
            else:
                summary.ingested += 1

        db.commit()
        return summary

    # ---------- helpers ----------

    def _materialise_upload(
        self, *, data: bytes, filename: str, content_hash: str
    ) -> str:
        """Persist uploaded bytes under STORAGE_DIR using the content hash."""
        settings = get_settings()
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".bin"
        dst = settings.storage_dir / f"{content_hash}{suffix}"
        if not dst.exists():
            dst.write_bytes(data)
        return str(dst.resolve())
