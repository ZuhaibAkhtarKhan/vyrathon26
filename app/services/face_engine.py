"""Pluggable face detection + embedding engines.

Two concrete implementations are provided:

* :class:`OpenCVFaceEngine` — real engine using OpenCV's YuNet detector and
  SFace recogniser. Models auto-download from the OpenCV zoo on first use.
* :class:`StubFaceEngine` — deterministic, no-network fallback used by tests
  and as an escape hatch when the ONNX weights cannot be downloaded. It
  detects a "face" per image by hashing the pixel data and produces a stable
  128-d embedding derived from that hash, so two copies of the same image
  still cluster together but two different images never collide.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.utils.errors import StorageError

logger = logging.getLogger(__name__)


# -------- Value objects --------


@dataclass(frozen=True)
class DetectedFace:
    """A detected face along with its 128-d L2-normalised embedding."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    detection_confidence: float
    embedding: np.ndarray  # shape (128,), L2-normalised

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return int(w) * int(h)


# -------- Abstract base --------


class FaceEngine(ABC):
    """Detect faces and produce comparable embeddings."""

    embedding_dim: int = 128

    @abstractmethod
    def detect_and_embed(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Return every face found in the image, with embeddings."""


# -------- OpenCV YuNet + SFace --------


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dst)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
            while chunk := resp.read(64 * 1024):
                f.write(chunk)
        tmp.replace(dst)
    except Exception as exc:  # pragma: no cover - network path
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise StorageError(
            f"Failed to download model weights from {url}. "
            "Place the file manually at "
            f"{dst} or set FACE_ENGINE=stub.",
            details={"url": url, "dst": str(dst), "cause": str(exc)},
        ) from exc


class OpenCVFaceEngine(FaceEngine):
    """Detector: YuNet ONNX. Recogniser: SFace ONNX."""

    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_dir = settings.model_dir
        self.detection_threshold = settings.face_detection_score_threshold

        self._yunet_path = self.model_dir / "face_detection_yunet_2023mar.onnx"
        self._sface_path = self.model_dir / "face_recognition_sface_2021dec.onnx"

        self._ensure_models()

        # `cv2.FaceDetectorYN.create` expects input_size to be set per-image
        # via `setInputSize`; we'll update it before each detect call.
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self._yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=float(self.detection_threshold),
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            model=str(self._sface_path),
            config="",
        )

    def _ensure_models(self) -> None:
        if not self._yunet_path.exists():
            _download(YUNET_URL, self._yunet_path)
        if not self._sface_path.exists():
            _download(SFACE_URL, self._sface_path)

    def detect_and_embed(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]

        # OpenCV's detector / recogniser are not thread-safe — serialise.
        with self._lock:
            self._detector.setInputSize((w, h))
            _, results = self._detector.detect(image_bgr)
            if results is None:
                return []

            faces: list[DetectedFace] = []
            for row in results:
                # YuNet returns: [x, y, w, h, 5x (lx, ly), score]
                x, y, fw, fh = (int(round(v)) for v in row[:4])
                score = float(row[-1])

                # Clip into image bounds to avoid negative coords blowing up align.
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                fw = max(1, min(fw, w - x))
                fh = max(1, min(fh, h - y))

                try:
                    aligned = self._recognizer.alignCrop(image_bgr, row)
                    feat = self._recognizer.feature(aligned)
                except cv2.error as exc:
                    logger.warning("SFace align/feature failed, skipping face: %s", exc)
                    continue

                embedding = np.asarray(feat, dtype=np.float32).reshape(-1)
                embedding = _l2_normalise(embedding)

                faces.append(
                    DetectedFace(
                        bbox=(x, y, fw, fh),
                        detection_confidence=score,
                        embedding=embedding,
                    )
                )
            return faces


# -------- Deterministic stub (tests / offline) --------


class StubFaceEngine(FaceEngine):
    """A face engine with no model weights.

    * Detects exactly one "face" per decoded image (the whole frame).
    * Produces a 128-d embedding derived from the SHA-256 of the decoded pixel
      data, so identical images cluster and different images don't collide.
    * Intended for unit tests and for environments that can't reach the
      OpenCV zoo to download ONNX weights.
    """

    def detect_and_embed(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        digest = hashlib.sha256(image_bgr.tobytes()).digest()
        # Expand 32 bytes -> 128 floats deterministically.
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
        vec = rng.standard_normal(128).astype(np.float32)
        return [
            DetectedFace(
                bbox=(0, 0, w, h),
                detection_confidence=1.0,
                embedding=_l2_normalise(vec),
            )
        ]


# -------- Helpers --------


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised (or not) vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -------- Factory --------


_engine_singleton: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_face_engine(settings: Settings | None = None) -> FaceEngine:
    """Return (and lazily build) the configured face engine."""
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton
    with _engine_lock:
        if _engine_singleton is not None:
            return _engine_singleton
        settings = settings or get_settings()
        if settings.face_engine.lower() == "stub":
            logger.info("Using StubFaceEngine (no ONNX models).")
            _engine_singleton = StubFaceEngine()
        else:
            logger.info("Using OpenCVFaceEngine (YuNet + SFace).")
            _engine_singleton = OpenCVFaceEngine(settings)
        return _engine_singleton


def reset_face_engine() -> None:
    """For tests — drop the cached engine."""
    global _engine_singleton
    _engine_singleton = None
