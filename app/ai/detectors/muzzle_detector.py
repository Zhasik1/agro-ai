"""Cattle muzzle detector — geometric ROI heuristic.

The muzzle of a cattle animal typically occupies the lower-front portion of
the body bounding box.  This module implements a deterministic geometric crop
that centres on that region, improving downstream embedding quality compared
to using the full body crop.

**Baseline heuristic (MVP)**::

    y1 = int(H * 0.45),  y2 = int(H * 0.95)   # lower-half emphasis
    x1 = int(W * 0.20),  x2 = int(W * 0.80)   # centred, exclude flanks

**Production replacement**
    Fine-tune a dedicated YOLO head on a cattle-muzzle dataset (e.g.
    BovineNetID, Roboflow muzzle sets) and substitute this function.
    See ``docs/ROADMAP.md`` for the planned timeline.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_muzzle"]

_MIN_SIDE = 64


def detect_muzzle(animal_crop: np.ndarray) -> np.ndarray:
    """Return the muzzle ROI from a cattle bounding-box crop.

    Uses a geometric heuristic: the cattle muzzle is in the lower-front 35 %
    of the body bounding box.  If the crop is too small to sub-crop
    meaningfully (min side < 64 px), the original crop is returned unchanged.

    Args:
        animal_crop: BGR ndarray of shape ``(H, W, 3)`` — the full cattle
            bounding box returned by the species classifier.

    Returns:
        BGR ndarray cropped to the muzzle region, or ``animal_crop`` unchanged
        when the image is too small.

    Note:
        The production replacement for this function is a fine-tuned YOLO head
        trained on dedicated muzzle datasets.
    """
    h, w = animal_crop.shape[:2]
    if min(h, w) < _MIN_SIDE:
        return animal_crop

    y1 = max(0, int(h * 0.45))
    y2 = min(h, int(h * 0.95))
    x1 = max(0, int(w * 0.20))
    x2 = min(w, int(w * 0.80))

    return animal_crop[y1:y2, x1:x2]
