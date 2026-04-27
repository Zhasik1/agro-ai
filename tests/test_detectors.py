"""Tests for the geometric ROI sub-detectors."""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.detectors.face_detector import detect_face
from app.ai.detectors.muzzle_detector import detect_muzzle


@pytest.fixture
def crop_200x300() -> np.ndarray:
    """Synthetic 200-high × 300-wide BGR crop."""
    return np.zeros((200, 300, 3), dtype=np.uint8)


class TestMuzzleDetector:
    def test_returns_smaller_region(self, crop_200x300: np.ndarray) -> None:
        """Muzzle ROI must be strictly smaller than the input."""
        roi = detect_muzzle(crop_200x300)
        assert roi.shape[0] < crop_200x300.shape[0] or roi.shape[1] < crop_200x300.shape[1]

    def test_expected_dimensions(self, crop_200x300: np.ndarray) -> None:
        """Verify the ROI dimensions match the documented heuristic."""
        h, w = 200, 300
        expected_h = int(h * 0.95) - int(h * 0.45)  # 100
        expected_w = int(w * 0.80) - int(w * 0.20)  # 180
        roi = detect_muzzle(crop_200x300)
        assert roi.shape == (expected_h, expected_w, 3)

    def test_small_crop_guard_returns_original(self) -> None:
        """Crops smaller than 64×64 must be returned unchanged (same object)."""
        small = np.zeros((32, 32, 3), dtype=np.uint8)
        result = detect_muzzle(small)
        assert result is small

    def test_64x64_boundary_is_cropped(self) -> None:
        """A 64×64 crop is large enough to sub-crop."""
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        roi = detect_muzzle(crop)
        # Must differ from input (heuristic produces a sub-region)
        assert roi.shape != crop.shape


class TestFaceDetector:
    def test_sheep_returns_smaller_region(self, crop_200x300: np.ndarray) -> None:
        roi = detect_face(crop_200x300, species="sheep")
        assert roi.shape[0] < crop_200x300.shape[0] or roi.shape[1] < crop_200x300.shape[1]

    def test_horse_returns_smaller_region(self, crop_200x300: np.ndarray) -> None:
        roi = detect_face(crop_200x300, species="horse")
        assert roi.shape[0] < crop_200x300.shape[0] or roi.shape[1] < crop_200x300.shape[1]

    def test_sheep_vs_horse_differ_in_height(self, crop_200x300: np.ndarray) -> None:
        """Horse ROI must be taller than sheep ROI (y2=0.65 vs y2=0.55)."""
        sheep_roi = detect_face(crop_200x300, species="sheep")
        horse_roi = detect_face(crop_200x300, species="horse")
        assert horse_roi.shape[0] > sheep_roi.shape[0]

    def test_sheep_expected_dimensions(self, crop_200x300: np.ndarray) -> None:
        h, w = 200, 300
        expected_h = int(h * 0.55) - int(h * 0.05)  # 100
        expected_w = int(w * 0.80) - int(w * 0.20)  # 180
        roi = detect_face(crop_200x300, species="sheep")
        assert roi.shape == (expected_h, expected_w, 3)

    def test_horse_expected_dimensions(self, crop_200x300: np.ndarray) -> None:
        h, w = 200, 300
        expected_h = int(h * 0.65) - int(h * 0.05)  # 120
        expected_w = int(w * 0.80) - int(w * 0.20)  # 180
        roi = detect_face(crop_200x300, species="horse")
        assert roi.shape == (expected_h, expected_w, 3)

    def test_small_crop_guard_returns_original(self) -> None:
        small = np.zeros((32, 32, 3), dtype=np.uint8)
        result = detect_face(small)
        assert result is small

    def test_default_species_is_sheep(self, crop_200x300: np.ndarray) -> None:
        """Calling detect_face without species= must behave like species='sheep'."""
        default_roi = detect_face(crop_200x300)
        sheep_roi = detect_face(crop_200x300, species="sheep")
        assert default_roi.shape == sheep_roi.shape
