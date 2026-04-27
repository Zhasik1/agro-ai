"""Sheep / horse face detector — geometric ROI heuristic.

For sheep and horses the head occupies the upper portion of the animal body
bounding box (unlike cattle where the muzzle is in the lower half).  This
module implements a species-aware deterministic crop.

**Baseline heuristics (MVP)**::

    sheep: y1=int(H*0.05), y2=int(H*0.55), x1=int(W*0.20), x2=int(W*0.80)
    horse: y1=int(H*0.05), y2=int(H*0.65), x1=int(W*0.20), x2=int(W*0.80)

**Production replacement**
    Fine-tune a dedicated YOLO or landmark network on sheep/horse face
    datasets and substitute this function.  See ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_face"]

_MIN_SIDE = 64


def detect_face(animal_crop: np.ndarray, species: str = "sheep") -> np.ndarray:
    """Return the face ROI from a sheep or horse bounding-box crop.

    Uses a geometric heuristic: the head is in the upper portion of the body
    bounding box.  The exact vertical extent depends on ``species``.
    If the crop is too small to sub-crop meaningfully (min side < 64 px),
    the original crop is returned unchanged.

    Args:
        animal_crop: BGR ndarray of shape ``(H, W, 3)`` — the full animal
            bounding box returned by the species classifier.
        species: Either ``"sheep"`` (default) or ``"horse"``.  Horse uses a
            taller head region (y2 = 0.65 H vs 0.55 H for sheep).

    Returns:
        BGR ndarray cropped to the face region, or ``animal_crop`` unchanged
        when the image is too small.

    Note:
        The production replacement for this function is a fine-tuned YOLO /
        landmark network trained on dedicated face datasets.
    """
    h, w = animal_crop.shape[:2]
    if min(h, w) < _MIN_SIDE:
        return animal_crop

    y1 = max(0, int(h * 0.05))
    y2_frac = 0.65 if species == "horse" else 0.55
    y2 = min(h, int(h * y2_frac))
    x1 = max(0, int(w * 0.20))
    x2 = min(w, int(w * 0.80))

    return animal_crop[y1:y2, x1:x2]
