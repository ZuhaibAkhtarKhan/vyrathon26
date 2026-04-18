"""Image decoding + crawling helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.utils.errors import UnsupportedMediaError

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR ``numpy`` array. Raises on invalid data."""
    if not data:
        raise UnsupportedMediaError("Empty file upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedMediaError(
            "File bytes could not be decoded as an image. "
            "Supported formats: JPEG, PNG, WEBP, BMP."
        )
    return img


def iter_image_files(root: Path, *, recursive: bool = True) -> list[Path]:
    """Enumerate image files under ``root``."""
    if not root.exists() or not root.is_dir():
        return []
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )
