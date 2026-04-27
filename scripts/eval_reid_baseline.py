"""Baseline re-identification evaluation (gallery/query, top-1, ROC-AUC).

The script discovers identity folders under --data-root, extracts embeddings with
current app FeatureExtractor (ResNet50 + deterministic projection), then reports
retrieval quality metrics.

Expected layout:
    data_root/
      identity_001/
        img1.jpg
        img2.jpg
      identity_002/
        a.jpg
        b.jpg

Usage:
    python scripts/eval_reid_baseline.py \
            --data-root data/datasets/cattle/cattely

    python scripts/eval_reid_baseline.py \
            --data-root data/datasets/cattle/cattely \
      --max-identities 20 --max-images-per-identity 8 --output-json docs/eval_runs/cattle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ai.extractors.feature_extractor import get_feature_extractor  # noqa: E402
from app.core import accel  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    identity: str
    path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline ReID evaluation")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--species", default="unknown", help="Label used in the report")
    parser.add_argument("--query-per-id", type=int, default=1)
    parser.add_argument("--min-images-per-id", type=int, default=2)
    parser.add_argument("--max-identities", type=int, default=0)
    parser.add_argument("--max-images-per-identity", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def discover_identities(data_root: Path) -> dict[str, list[Path]]:
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise SystemExit(f"Data root must be a directory: {data_root}")

    by_identity: dict[str, list[Path]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = [p for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        if images:
            by_identity[child.name] = sorted(images)
    return by_identity


def sample_identities(
    by_identity: dict[str, list[Path]],
    min_images_per_id: int,
    max_identities: int,
    max_images_per_identity: int,
    rng: np.random.Generator,
) -> dict[str, list[Path]]:
    filtered: dict[str, list[Path]] = {
        identity: imgs
        for identity, imgs in by_identity.items()
        if len(imgs) >= min_images_per_id
    }

    if not filtered:
        raise SystemExit(
            "No identities matched filters. "
            f"Need at least {min_images_per_id} images per identity."
        )

    ids = list(filtered.keys())
    rng.shuffle(ids)
    if max_identities > 0:
        ids = ids[:max_identities]

    sampled: dict[str, list[Path]] = {}
    for identity in ids:
        imgs = list(filtered[identity])
        rng.shuffle(imgs)
        if max_images_per_identity > 0:
            imgs = imgs[:max_images_per_identity]
        if len(imgs) >= min_images_per_id:
            sampled[identity] = imgs

    if len(sampled) < 2:
        raise SystemExit("Need at least 2 identities to compute meaningful retrieval metrics")
    return sampled


def split_gallery_query(
    sampled: dict[str, list[Path]],
    query_per_id: int,
    rng: np.random.Generator,
) -> tuple[list[ImageSample], list[ImageSample]]:
    query: list[ImageSample] = []
    gallery: list[ImageSample] = []

    for identity, imgs in sampled.items():
        local = list(imgs)
        rng.shuffle(local)
        q_count = min(max(query_per_id, 1), len(local) - 1)
        if q_count <= 0:
            continue
        query += [ImageSample(identity=identity, path=p) for p in local[:q_count]]
        gallery += [ImageSample(identity=identity, path=p) for p in local[q_count:]]

    if not query or not gallery:
        raise SystemExit("Failed to build query/gallery split from provided data")
    return query, gallery


def extract_embeddings(samples: list[ImageSample]) -> tuple[list[ImageSample], np.ndarray, int]:
    extractor = get_feature_extractor()
    valid: list[ImageSample] = []
    vectors: list[np.ndarray] = []
    unreadable = 0

    for idx, sample in enumerate(samples, start=1):
        image = cv2.imread(str(sample.path))
        if image is None:
            unreadable += 1
            continue
        emb = extractor.extract(image).astype(np.float32)
        emb = accel.l2_normalize(emb)
        valid.append(sample)
        vectors.append(emb)

        if idx % 50 == 0:
            print(f"  extracted {idx}/{len(samples)}")

    if not vectors:
        raise SystemExit("No readable images for embedding extraction")

    matrix = np.vstack(vectors).astype(np.float32)
    return valid, matrix, unreadable


def roc_auc_mann_whitney(scores: np.ndarray, positives: np.ndarray) -> float:
    positives = positives.astype(bool)
    n_pos = int(positives.sum())
    n_neg = int(len(positives) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[positives].sum())
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def evaluate(
    query_samples: list[ImageSample],
    query_emb: np.ndarray,
    gallery_samples: list[ImageSample],
    gallery_emb: np.ndarray,
) -> dict[str, float | int]:
    gallery_ids = np.array([s.identity for s in gallery_samples], dtype=object)

    top1_hits = 0
    pair_scores: list[np.ndarray] = []
    pair_labels: list[np.ndarray] = []

    for i, query in enumerate(query_samples):
        sims = accel.cosine_similarity_batch(query_emb[i], gallery_emb)
        top_idx = int(np.argmax(sims))
        if gallery_ids[top_idx] == query.identity:
            top1_hits += 1

        pair_scores.append(sims.astype(np.float32))
        pair_labels.append((gallery_ids == query.identity))

    scores_all = np.concatenate(pair_scores)
    labels_all = np.concatenate(pair_labels).astype(bool)

    return {
        "query_count": len(query_samples),
        "gallery_count": len(gallery_samples),
        "pair_count": int(scores_all.shape[0]),
        "positive_pairs": int(labels_all.sum()),
        "negative_pairs": int((~labels_all).sum()),
        "top1": float(top1_hits / len(query_samples)),
        "roc_auc": float(roc_auc_mann_whitney(scores_all, labels_all)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.query_per_id < 1:
        raise SystemExit("--query-per-id must be >= 1")
    if args.min_images_per_id < 2:
        raise SystemExit("--min-images-per-id must be >= 2")

    rng = np.random.default_rng(args.seed)

    discovered = discover_identities(args.data_root)
    sampled = sample_identities(
        discovered,
        min_images_per_id=args.min_images_per_id,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        rng=rng,
    )

    query, gallery = split_gallery_query(sampled, query_per_id=args.query_per_id, rng=rng)

    print("ReID baseline evaluation")
    print("=" * 88)
    print(f"data_root: {args.data_root}")
    print(f"species:   {args.species}")
    print(f"identities discovered/sampled: {len(discovered)}/{len(sampled)}")
    print(f"query/gallery images: {len(query)}/{len(gallery)}")

    print("Extracting query embeddings...")
    query_ok, query_emb, query_unreadable = extract_embeddings(query)
    print("Extracting gallery embeddings...")
    gallery_ok, gallery_emb, gallery_unreadable = extract_embeddings(gallery)

    metrics = evaluate(query_ok, query_emb, gallery_ok, gallery_emb)
    metrics.update(
        {
            "species": args.species,
            "seed": args.seed,
            "unreadable_query_images": query_unreadable,
            "unreadable_gallery_images": gallery_unreadable,
            "sampled_identities": len(sampled),
        }
    )

    print("-" * 88)
    print(f"top-1:   {metrics['top1']:.4f}")
    print(f"roc-auc: {metrics['roc_auc']:.4f}")
    print(
        "pairs:   "
        f"{metrics['pair_count']} "
        f"(pos={metrics['positive_pairs']}, neg={metrics['negative_pairs']})"
    )
    print("-" * 88)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
