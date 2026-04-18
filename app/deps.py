"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import UploadFile

from app.config import get_settings
from app.utils.errors import PayloadTooLargeError, UnsupportedMediaError


async def read_image_upload(upload: UploadFile) -> tuple[bytes, str]:
    """Validate and buffer an image upload.

    Returns the raw bytes and the original filename. Raises domain errors
    that the global handler converts to 413 / 415.
    """
    settings = get_settings()

    if upload.content_type and upload.content_type not in settings.allowed_mime_types:
        raise UnsupportedMediaError(
            f"Unsupported content type: {upload.content_type}. "
            f"Allowed: {', '.join(settings.allowed_mime_types)}",
            details={"content_type": upload.content_type},
        )

    data = await upload.read()
    if not data:
        raise UnsupportedMediaError("Empty file upload.")
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds {settings.max_upload_bytes} bytes.",
            details={"size": len(data), "limit": settings.max_upload_bytes},
        )
    return data, upload.filename or "upload.bin"
