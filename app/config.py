"""Application configuration loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Env vars override everything."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./grabpic.db",
        description="SQLAlchemy connection URL.",
    )

    # --- Storage ---
    storage_dir: Path = Field(default=Path("./data/images"))
    model_dir: Path = Field(default=Path("./models"))

    # --- Face recognition ---
    face_engine: str = Field(default="opencv")  # "opencv" | "stub"
    face_match_threshold: float = Field(default=0.363)
    face_detection_score_threshold: float = Field(default=0.6)

    # --- API ---
    api_prefix: str = Field(default="/api/v1")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024)

    # Allowed image MIME types for uploads.
    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    )

    def ensure_directories(self) -> None:
        """Create runtime directories if they don't yet exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
