"""Acceleration facade with Rust-first and Python fallback implementations."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np

try:
    import agroai_core as _rust_core
except Exception:  # noqa: BLE001
    _rust_core = None

try:
    import blake3 as _blake3
except Exception:  # noqa: BLE001
    _blake3 = None

__all__ = [
    "backend",
    "has_rust",
    "decode_image_rgb",
    "resize_bilinear",
    "bgr_to_rgb",
    "normalize_imagenet",
    "crop_bbox",
    "sha256_hex",
    "blake3_hex",
    "phash",
    "bbox_iou",
    "bbox_area",
    "nms",
    "cosine_similarity",
    "cosine_similarity_batch",
    "top_k",
    "l2_normalize",
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def backend() -> str:
    return "rust" if _rust_core is not None else "python"


def has_rust() -> bool:
    return _rust_core is not None


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an image shaped (H, W, 3)")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _coerce_bbox(bbox: object) -> tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        return (
            float(bbox["x1"]),
            float(bbox["y1"]),
            float(bbox["x2"]),
            float(bbox["y2"]),
        )
    if hasattr(bbox, "x1") and hasattr(bbox, "y1") and hasattr(bbox, "x2") and hasattr(bbox, "y2"):
        return (
            float(getattr(bbox, "x1")),
            float(getattr(bbox, "y1")),
            float(getattr(bbox, "x2")),
            float(getattr(bbox, "y2")),
        )

    seq = tuple(bbox)  # type: ignore[arg-type]
    if len(seq) != 4:
        raise ValueError("bbox must have 4 elements")
    return float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])


def _rust_bbox_arg(bbox: object) -> SimpleNamespace:
    x1, y1, x2, y2 = _coerce_bbox(bbox)
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def decode_image_rgb(data: bytes) -> np.ndarray:
    if _rust_core is not None:
        return np.asarray(_rust_core.decode_image_rgb(data), dtype=np.uint8)

    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_bilinear(input: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    image = _as_uint8_rgb(input)
    if _rust_core is not None:
        return np.asarray(_rust_core.resize_bilinear(image, int(target_h), int(target_w)), dtype=np.uint8)
    return cv2.resize(image, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(input: np.ndarray) -> np.ndarray:
    image = _as_uint8_rgb(input)
    if _rust_core is not None:
        return np.asarray(_rust_core.bgr_to_rgb(image), dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_imagenet(input: np.ndarray) -> np.ndarray:
    image = _as_uint8_rgb(input)
    if _rust_core is not None:
        return np.asarray(_rust_core.normalize_imagenet(image), dtype=np.float32)

    x = image.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)


def crop_bbox(input: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    image = _as_uint8_rgb(input)
    if _rust_core is not None:
        return np.asarray(_rust_core.crop_bbox(image, tuple(int(v) for v in bbox)), dtype=np.uint8)

    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = image.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box is empty after clipping")
    return image[y1:y2, x1:x2].copy()


def sha256_hex(data: bytes) -> str:
    if _rust_core is not None:
        return str(_rust_core.sha256_hex(data))
    return hashlib.sha256(data).hexdigest()


def blake3_hex(data: bytes) -> str:
    if _rust_core is not None:
        return str(_rust_core.blake3_hex(data))
    if _blake3 is None:
        raise RuntimeError("Python fallback for blake3 requires the 'blake3' package")
    return _blake3.blake3(data).hexdigest()


def phash(input: np.ndarray) -> str:
    image = _as_uint8_rgb(input)
    if _rust_core is not None:
        return str(_rust_core.phash(image))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    flat = dct.reshape(-1)
    median = float(np.median(flat[1:]))
    bits = 0
    for i, value in enumerate(flat):
        if float(value) > median:
            bits |= 1 << i
    return f"{bits:016x}"


def bbox_iou(a: object, b: object) -> float:
    if _rust_core is not None:
        return float(_rust_core.bbox_iou(_rust_bbox_arg(a), _rust_bbox_arg(b)))

    ax1, ay1, ax2, ay2 = _coerce_bbox(a)
    bx1, by1, bx2, by2 = _coerce_bbox(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_area(b: object) -> float:
    if _rust_core is not None:
        return float(_rust_core.bbox_area(_rust_bbox_arg(b)))

    x1, y1, x2, y2 = _coerce_bbox(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if s.ndim != 1 or s.shape[0] != b.shape[0]:
        raise ValueError("scores must have shape (N,)")

    if _rust_core is not None:
        return np.asarray(_rust_core.nms(b, s, float(iou_threshold)), dtype=np.int64)

    order = np.argsort(-s)
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = np.array([bbox_iou(b[i], b[j]) for j in rest], dtype=np.float32)
        rest = rest[ious < float(iou_threshold)]
        order = rest

    return np.asarray(keep, dtype=np.int64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if _rust_core is not None:
        return float(_rust_core.cosine_similarity(va, vb))

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= np.finfo(np.float32).eps:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return float(np.dot(va, vb) / denom)


def cosine_similarity_batch(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    db = np.asarray(database, dtype=np.float32)
    if db.ndim != 2:
        raise ValueError("database must have shape (N, D)")

    if _rust_core is not None:
        return np.asarray(_rust_core.cosine_similarity_batch(q, db), dtype=np.float32)

    return (db @ q).astype(np.float32, copy=False)


def top_k(scores: np.ndarray | Iterable[float], k: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float32)
    if s.ndim != 1:
        raise ValueError("scores must be 1D")

    if _rust_core is not None:
        idx, vals = _rust_core.top_k(s, int(k))
        return np.asarray(idx, dtype=np.int64), np.asarray(vals, dtype=np.float32)

    if k <= 0 or s.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    k_eff = min(int(k), int(s.size))
    idx = np.argpartition(-s, k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-s[idx])]
    return idx.astype(np.int64), s[idx].astype(np.float32)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    if _rust_core is not None:
        return np.asarray(_rust_core.l2_normalize(vec), dtype=np.float32)

    norm = float(np.linalg.norm(vec))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("Cannot normalize zero vector")
    return (vec / norm).astype(np.float32)
