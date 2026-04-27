# Next Steps (Operational Backlog)

This file tracks the immediate follow-up work after the dataset integration phase.

## Current Snapshot (2026-04-27)

- Auto-ready: `cattely`, `huggingface-cows-detection`, `huggingface-horse-id`, `pferd-pose`.
- Auto-partial: `cows-frontal-face-zenodo` (download in progress, expected 13.9 GB).
- Manual-pending: `pmc-cattle-muzzle`, `sheepface-107`, `roboflow-sheep-face-search`, `roboflow-horse-face`.
- Status commands:
	- `python scripts/dataset_status.py` (`--strict-auto` returns non-zero while auto datasets are incomplete).
	- `python scripts/validate_datasets.py` (returns non-zero while non-manual datasets are missing/partial/suspicious).

## 1) Dataset Completion (High Priority)

- [ ] Finish `cows-frontal-face-zenodo` full download and unpack under `data/datasets/cattle/cows-frontal-face-zenodo/`.
- [ ] Acquire `sheepface-107` files from the corresponding author and place them under `data/datasets/sheep/sheepface-107/`.
- [ ] Export a concrete Roboflow sheep dataset and place files under `data/datasets/sheep/roboflow-sheep-face-search/`.
- [ ] Export `roboflow-horse-face` via Roboflow account/API and place files under `data/datasets/horse/roboflow-horse-face/`.

## 2) Data Quality and Structure

- [ ] Standardize per-dataset folder layout (`images/`, `labels/`, `metadata/` where applicable).
- [x] Add a lightweight dataset installation/status check (`scripts/dataset_status.py`) with `partial` detection for large HTTPS archives.
- [x] Add baseline payload validation (`scripts/validate_datasets.py`) for missing/empty/partial/suspicious datasets.
- [x] Extend integrity checks with image readability probe, YOLO label-file parsing, and `data.yaml` split-path checks.
- [ ] Deepen schema checks further (full annotation-schema validation and strict split consistency policies per dataset type).
- [ ] Record final dataset licenses and accepted terms in `docs/DATASETS.md`.

## 3) Model and Evaluation

- [x] Build a repeatable re-identification evaluation script (gallery/query split + top-1 + ROC-AUC) — `scripts/eval_reid_baseline.py`.
- [ ] Fine-tune the embedding head on at least one complete species dataset and store reproducible config.
- [ ] Publish measured results in a dedicated report (`docs/EVALUATION.md`).

## 4) App Integration

- [ ] Add a switch to load fine-tuned embedding weights from `ml_models/`.
- [ ] Recompute FAISS indexes after any embedding-model change.
- [ ] Recalibrate matching thresholds from measured validation curves.

## 5) Documentation Cleanup

- [ ] Resolve `// TODO: confirm citation` placeholders in `docs/DATASETS.md` where sources are verified.
- [ ] Keep `data/datasets.yaml`, `scripts/download_datasets.py`, and docs synchronized after each dataset update.
