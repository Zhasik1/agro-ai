"""Species classifier wrapping a YOLOv8 model with default COCO weights.

COCO class indices used:
    * 19 -> cow   -> Species.CATTLE
    * 18 -> sheep -> Species.SHEEP
    * 17 -> horse -> Species.HORSE
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import get_settings
from app.core import accel
from app.database.models import Species
from app.exceptions import AnimalNotDetectedError

__all__ = [
    "SpeciesDetection",
    "SpeciesClassifier",
    "get_species_classifier",
    "classify_species",
]

logger = logging.getLogger(__name__)

# COCO indices
_COCO_TO_SPECIES: dict[int, Species] = {
    19: Species.CATTLE,
    18: Species.SHEEP,
    17: Species.HORSE,
}
_TARGET_CLASSES = list(_COCO_TO_SPECIES.keys())
_DEFAULT_CONF = 0.5
_YOLO_WEIGHTS = "yolov8n.pt"


@dataclass
class SpeciesDetection:
    """Result of a successful species detection."""

    species: Species
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    cropped_image: np.ndarray  # BGR


class SpeciesClassifier:
    """Singleton-style YOLOv8 wrapper for species classification."""

    def __init__(self, weights: str = _YOLO_WEIGHTS) -> None:
        # Lazy import — keeps unit tests fast when ML is mocked.
        from ultralytics import YOLO  # type: ignore

        settings = get_settings()
        weights_path = settings.MODEL_PATH / weights
        # Ultralytics auto-downloads if the file is missing.
        target = str(weights_path) if weights_path.exists() else weights
        logger.info("Loading YOLOv8 weights: %s", target)
        self._model = YOLO(target)

    def detect(self, image_bgr: np.ndarray, conf: float = _DEFAULT_CONF) -> SpeciesDetection:
        """Run detection and return the top supported animal.

        Raises:
            AnimalNotDetectedError: if no cattle / sheep / horse is detected.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise AnimalNotDetectedError("Бос сурет / Empty image")

        results = self._model.predict(
            image_bgr, conf=conf, classes=_TARGET_CLASSES, verbose=False
        )
        if not results:
            raise AnimalNotDetectedError("Жануар табылмады / No animal detected")

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            raise AnimalNotDetectedError("Жануар табылмады / No animal detected")

        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)

        if len(confidences) > 1:
            keep = accel.nms(xyxy, confidences.astype(np.float32), iou_threshold=0.5)
            if keep.size > 0:
                confidences = confidences[keep]
                classes = classes[keep]
                xyxy = xyxy[keep]

        best = int(np.argmax(confidences))
        cls_idx = int(classes[best])
        if cls_idx not in _COCO_TO_SPECIES:
            raise AnimalNotDetectedError("Қолдау көрсетілмейтін түр / Unsupported species")

        species = _COCO_TO_SPECIES[cls_idx]
        confidence = float(confidences[best])
        x1, y1, x2, y2 = (int(v) for v in xyxy[best])
        try:
            crop = accel.crop_bbox(image_bgr, (x1, y1, x2, y2))
        except ValueError as exc:
            raise AnimalNotDetectedError("Жануар табылмады / No animal detected") from exc
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        logger.info(
            "Detected species=%s confidence=%.3f bbox=(%d,%d,%d,%d)",
            species.value, confidence, x1, y1, x2, y2,
        )

        return SpeciesDetection(
            species=species,
            confidence=confidence,
            bbox=(x1, y1, x2, y2),
            cropped_image=crop,
        )


_lock = threading.Lock()
_instance: Optional[SpeciesClassifier] = None


def get_species_classifier() -> SpeciesClassifier:
    """Return the lazy singleton classifier."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SpeciesClassifier()
    return _instance


def classify_species(image_bgr: np.ndarray) -> SpeciesDetection:
    """Convenience wrapper around the singleton."""
    return get_species_classifier().detect(image_bgr)
