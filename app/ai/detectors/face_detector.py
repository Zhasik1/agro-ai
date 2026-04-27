"""Sheep / horse face detector (extension point for production).

Currently a passthrough — see module docstring of
:mod:`app.ai.detectors.muzzle_detector`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_face"]


def detect_face(animal_crop: np.ndarray) -> np.ndarray:
    """Return the face ROI from a sheep / horse crop.

    MVP: identity passthrough.
    """
    return animal_crop
