"""Domain-level exceptions. Mapped to HTTP responses in `app.main`."""

from __future__ import annotations

from typing import Any


class GrabpicError(Exception):
    """Base class for all Grabpic domain errors."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(GrabpicError):
    code = "VALIDATION_ERROR"
    http_status = 400


class UnsupportedMediaError(GrabpicError):
    code = "UNSUPPORTED_MEDIA"
    http_status = 415


class PayloadTooLargeError(GrabpicError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class NotFoundError(GrabpicError):
    code = "NOT_FOUND"
    http_status = 404


class NoFaceDetectedError(GrabpicError):
    code = "NO_FACE"
    http_status = 400


class MultipleFacesInSelfieError(GrabpicError):
    code = "AMBIGUOUS_SELFIE"
    http_status = 400


class NoMatchError(GrabpicError):
    code = "NO_MATCH"
    http_status = 404


class StorageError(GrabpicError):
    code = "STORAGE_ERROR"
    http_status = 500
