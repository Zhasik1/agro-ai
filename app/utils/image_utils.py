"""Image utilities: validation, loading, hashing."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import get_settings
from app.core import accel
from app.exceptions import InvalidImageError

__all__ = ["sha256_bytes", "validate_image_bytes", "decode_image", "to_rgb", "to_bgr"]

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest of ``data``."""
    return accel.sha256_hex(data)


def validate_image_bytes(data: bytes, content_type: str | None) -> None:
    """Validate uploaded image size and MIME type.

    Raises:
        InvalidImageError: if file is empty, too big, or has wrong MIME.
    """
    settings = get_settings()
    if not data:
        raise InvalidImageError("Бос файл / Empty file")

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise InvalidImageError(f"Файл өте үлкен (max {settings.MAX_IMAGE_SIZE_MB} MB)")

    if content_type and content_type.lower() not in settings.ALLOWED_IMAGE_TYPES:
        raise InvalidImageError(
            f"Қолдау көрсетілмейтін формат: {content_type}. "
            f"Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
        )


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array (OpenCV convention).

    Raises:
        InvalidImageError: if bytes cannot be parsed as an image.
    """
    try:
        rgb = accel.decode_image_rgb(data)
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Сурет декодталмады: {exc}") from exc

    return to_bgr(rgb)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR ndarray to RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB ndarray to BGR."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
