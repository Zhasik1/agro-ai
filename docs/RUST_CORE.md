# Rust Core Integration

This document explains how the Rust extension `agroai_core` is integrated into MalChain with a safe Python fallback path.

## Goals

- Speed up CPU-heavy image and vector math primitives.
- Keep the existing FastAPI + Streamlit flow unchanged.
- Preserve full behavior when Rust wheel is not installed.

## Components

- `rust_core/`: workspace root.
- `rust_core/crates/agroai_core`: PyO3 extension module.
- `app/core/accel.py`: runtime facade that auto-selects Rust or Python backend.

## Exported Rust API

Image ops:
- `decode_image_rgb(bytes)`
- `resize_bilinear(input, target_h, target_w)`
- `bgr_to_rgb(input)`
- `normalize_imagenet(input)`
- `crop_bbox(input, bbox)`

Hashing:
- `sha256_hex(bytes)`
- `blake3_hex(bytes)`
- `phash(input)`

BBox and similarity:
- `bbox_iou(a, b)`
- `bbox_area(b)`
- `nms(boxes, scores, iou_threshold)`
- `cosine_similarity(a, b)`
- `cosine_similarity_batch(query, database)`
- `top_k(scores, k)`
- `l2_normalize(v)`

## Python Fallback

`app/core/accel.py` loads `agroai_core` when available:

- Backend `rust`: uses native extension.
- Backend `python`: uses NumPy/OpenCV/hashlib fallbacks.

No call site needs to know which backend is active.

## Integrated Hot Paths

- `app/utils/image_utils.py`
  - `sha256_bytes` and `decode_image` are routed through `app.core.accel`.
- `app/ai/extractors/feature_extractor.py`
  - preprocess pipeline now uses accel for BGR->RGB, resize, and ImageNet normalization.
- `app/ai/species_classifier.py`
  - optional NMS and bbox crop through accel.
- `scripts/eval_reid_baseline.py`
  - batch similarity and L2 normalization through accel.
- `app/main.py`
  - startup log includes selected acceleration backend.

## Build and Test

```bash
make rust-build
make rust-install
make rust-test
make rust-bench
```

Or directly:

```bash
cd rust_core/crates/agroai_core
maturin develop --release
```

## CI

`.github/workflows/rust.yml` runs:

- Rust tests on Linux/macOS/Windows, Python 3.11/3.12.
- Python integration test for Rust extension.
- wheel build via `PyO3/maturin-action`.

## Notes

- YOLO, FAISS, DB, and web layers remain in Python.
- No CUDA assumptions in Rust core (CPU-only implementation).
- Performance numbers are hardware-dependent and must be generated locally with `cargo bench`.
