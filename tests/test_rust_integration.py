"""Integration parity tests for Rust core vs Python fallbacks.

These tests are skipped automatically when `agroai_core` is not installed.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pytest

from app.core import accel

agroai_core = pytest.importorskip("agroai_core")


def _python_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        inter_x1 = max(float(a[0]), float(b[0]))
        inter_y1 = max(float(a[1]), float(b[1]))
        inter_x2 = min(float(a[2]), float(b[2]))
        inter_y2 = min(float(a[3]), float(b[3]))
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        return 0.0 if union <= 0.0 else inter / union

    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = np.array([iou(boxes[idx], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[overlaps < iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def test_sha256_hex_matches_python_hashlib() -> None:
    payload = b"malchain-rust-core"
    assert accel.sha256_hex(payload) == hashlib.sha256(payload).hexdigest()
    assert agroai_core.sha256_hex(payload) == hashlib.sha256(payload).hexdigest()


def test_resize_parity_with_opencv() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(61, 47, 3), dtype=np.uint8)

    rust_out = agroai_core.resize_bilinear(image, 128, 96)
    cv_out = cv2.resize(image, (96, 128), interpolation=cv2.INTER_LINEAR)

    diff = np.abs(rust_out.astype(np.int16) - cv_out.astype(np.int16))
    assert float(diff.max()) <= 2.0


def test_cosine_batch_matches_numpy_dot_for_normalized_vectors() -> None:
    rng = np.random.default_rng(11)
    query = rng.normal(size=(512,)).astype(np.float32)
    query /= np.linalg.norm(query)

    database = rng.normal(size=(256, 512)).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    rust_scores = agroai_core.cosine_similarity_batch(query, database)
    np_scores = database @ query

    np.testing.assert_allclose(rust_scores, np_scores, rtol=1e-5, atol=1e-5)


def test_nms_matches_python_reference() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)

    rust_keep = agroai_core.nms(boxes, scores, 0.5)
    py_keep = _python_nms(boxes, scores, 0.5)

    np.testing.assert_array_equal(rust_keep, py_keep)
