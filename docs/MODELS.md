# 🧠 Models — MalChain

> **Scope.** This document describes the models actually shipped in the MVP. It deliberately
> avoids quoting accuracy numbers we have not measured ourselves — see *Performance* below.

The pipeline has two layers:

```
input image ─▶  Layer A: detector + species classifier  ─▶  crop
                                                              │
                                                              ▼
                                                Layer B: embedding extractor ─▶ FAISS
```

---

## Layer A — Detector + species classifier

* **Architecture:** [Ultralytics YOLOv8n](https://docs.ultralytics.com/models/yolov8/)
  (`yolov8n.pt`, ~6.2 M params).
* **Weights:** zero-shot, pre-trained on COCO. We do **not** fine-tune in the MVP.
* **Species inference:** post-processing over COCO class IDs.
  * `17` → `horse`
  * `18` → `sheep`
  * `19` → `cow` (treated as `cattle`)
  * Any other class → species classification fails and the API returns a 4xx error.
* **Outputs:** highest-confidence bounding box + species label. Detector confidence is exposed via
  the API response (`detector_confidence`) but is **not** used as the matching threshold.

### Notes

* Performance of YOLOv8n on COCO is documented in the upstream
  [Ultralytics performance tables](https://docs.ultralytics.com/models/yolov8/#performance-metrics).
  We have **not** re-measured those numbers on our internal data — please consult the upstream
  table rather than quoting figures here.
* Multi-animal frames are reduced to a single crop (the highest-confidence box). Multi-subject
  identification is on the roadmap.

---

## Layer B — Embedding extractor

* **Architecture:** [`torchvision.models.resnet50`](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
  with `IMAGENET1K_V2` weights, final FC removed (2048-D pooled features).
* **Projection head:** a deterministic seeded random linear projection
  (`torch.manual_seed(42)`) maps 2048 → **512** dimensions, followed by L2 normalisation.
  This is *not* trained — it is a fixed reproducible map that lets us index a compact 512-D
  vector in FAISS. See `app/ai/extractors/feature_extractor.py`.
* **Distance metric:** cosine similarity, implemented through FAISS `IndexFlatIP` over
  L2-normalised vectors (one index per species).
* **Thresholds (configurable via `app/config.py`):**
  * `MATCH_THRESHOLD = 0.85` — the API returns `match`.
  * `SUSPECT_THRESHOLD = 0.70` — the API returns `suspect`.
  * Below `SUSPECT_THRESHOLD` — `no_match`.

### Why "zero-shot baseline"?

ResNet50 with ImageNet weights and a frozen random projection is a strong **baseline** for
similarity search on natural images, but it is **not** a livestock-specific identifier. We have
**not** measured per-species top-1 accuracy on the datasets listed in [DATASETS.md](DATASETS.md).
Anyone who quotes such a number must back it up with a reproducible evaluation script committed
to this repository.

### Optional fine-tuning

A starting-point notebook is provided at
[notebooks/finetune_baseline.ipynb](../notebooks/finetune_baseline.ipynb). It:

1. Loads the Cattely cattle-face dataset (with a synthetic fallback if it has not been
   downloaded yet).
2. Fine-tunes only the 2048 → 512 projection head with a triplet loss
   (`pytorch_metric_learning.losses.TripletMarginLoss`, `MultiSimilarityMiner`).
3. Saves the trained head to `ml_models/resnet50_cattle_finetuned.pt`.

The notebook is intentionally tiny (`EPOCHS = 1`, `BATCH_SIZE = 4`,
`NUM_INDIVIDUALS = 10`) so it runs end-to-end on a CPU. Scale these up when you have real data.

The MVP **does not load** `resnet50_cattle_finetuned.pt` automatically. Wiring the fine-tuned
head into `FeatureExtractor` is a follow-up task that should be gated behind a measured
improvement on a held-out evaluation set.

---

## Performance

* **No measured accuracy numbers are quoted here.** Earlier drafts of the brief proposed targets
  (e.g. ≥ 95 % top-1 on cattle), but the MVP has not been evaluated against those targets and we
  refuse to publish unverified figures.
* Latency on the demo machine (Apple Silicon CPU, 224×224 crop): typically **a few hundred
  milliseconds per request** end-to-end. This is an informal observation, not a benchmark.

To produce real numbers, build an evaluation script under `tests/eval/` that:

1. Splits a downloaded dataset (see [DATASETS.md](DATASETS.md)) into gallery / query.
2. Computes top-1 NN accuracy and ROC-AUC at the configured thresholds.
3. Saves the results to `docs/EVALUATION.md` with the commit hash and dataset version.

---

## Files of interest

* [app/ai/pipeline.py](../app/ai/pipeline.py) — orchestration.
* [app/ai/detectors/face_detector.py](../app/ai/detectors/face_detector.py) — YOLOv8n wrapper.
* [app/ai/extractors/feature_extractor.py](../app/ai/extractors/feature_extractor.py) — ResNet50 + projection.
* [app/database/vector_db.py](../app/database/vector_db.py) — FAISS index manager.
* [app/config.py](../app/config.py) — thresholds, paths, model names.
