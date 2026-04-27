"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """MalChain runtime configuration loaded from env / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "sqlite:///./malchain.db"

    # Storage
    VECTOR_DB_PATH: Path = Path("./data/vector_dbs")
    MODEL_PATH: Path = Path("./ml_models")

    # ML thresholds
    MATCH_THRESHOLD: float = 0.85
    SUSPECT_THRESHOLD: float = 0.70
    EMBEDDING_DIM: int = 512

    # Uploads
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")

    # API
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached :class:`Settings` instance."""
    return Settings()
