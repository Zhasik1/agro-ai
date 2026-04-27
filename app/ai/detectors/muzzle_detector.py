"""Cattle muzzle detector (extension point for production).

In MVP, the muzzle ROI is approximated by the full animal bbox returned by
:mod:`app.ai.species_classifier`. A production version will fine-tune a
small YOLO model on the BovineNetID / Roboflow muzzle datasets and replace
:func:`detect_muzzle` below.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_muzzle"]


def detect_muzzle(animal_crop: np.ndarray) -> np.ndarray:
    """Return the muzzle ROI from a cattle crop.

    MVP: identity passthrough — returns the input crop unchanged.

    Args:
        animal_crop: BGR ndarray of the animal bounding box.

    Returns:
        ROI as a BGR ndarray (currently the same crop).
    """
    return animal_crop
