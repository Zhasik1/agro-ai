"""Object detectors (muzzle, face) — future fine-grained ROI extractors.

For the MVP, the YOLOv8 species classifier already returns a tight bbox around
the whole animal which we use directly as the ROI for the feature extractor.
These modules are stubs marking the extension points where dedicated detectors
(e.g. cattle muzzle YOLO, sheep/horse face landmark net) will plug in.
"""

from __future__ import annotations

__all__: list[str] = []
