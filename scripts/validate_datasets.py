"""Validate local dataset payloads against data/datasets.yaml manifest.

Usage:
    python scripts/validate_datasets.py
    python scripts/validate_datasets.py --species horse
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "datasets.yaml"
DATA_ROOT = ROOT / "data" / "datasets"
SPECIES = ("cattle", "sheep", "horse")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
ANNOT_EXT = {".xml", ".json", ".csv", ".txt", ".yaml", ".yml"}
MAX_IMAGE_PROBE = 24
MAX_LABEL_PROBE = 200

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover
    Image = None
    UnidentifiedImageError = Exception


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install PyYAML") from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for sp in SPECIES:
        data.setdefault(sp, [])
    return data


def payload_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    out: list[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and p.name != "README_MANUAL.txt":
            out.append(p)
    return out


def sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    step = max(1, len(paths) // limit)
    return [paths[i] for i in range(0, len(paths), step)][:limit]


def https_target_file(path: Path, file_url: str | None) -> Path | None:
    if not file_url:
        return None
    parsed = urlparse(file_url)
    filename = unquote(Path(parsed.path).name)
    if not filename:
        return None
    return path / filename


def probe_images(images: list[Path]) -> tuple[int, int, list[str]]:
    if not images:
        return 0, 0, []
    if Image is None:
        return 0, 0, []

    checked = 0
    broken = 0
    examples: list[str] = []
    for img in sample_paths(images, MAX_IMAGE_PROBE):
        checked += 1
        try:
            with Image.open(img) as handle:
                handle.verify()
        except (UnidentifiedImageError, OSError):
            broken += 1
            if len(examples) < 3:
                examples.append(str(img.name))
    return checked, broken, examples


def parse_yolo_label_file(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            return False
        try:
            int(float(parts[0]))
            for token in parts[1:5]:
                float(token)
        except ValueError:
            return False
    return True


def yaml_split_path_issues(path: Path, files: list[Path]) -> list[str]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return []

    issues: list[str] = []
    split_keys = ("train", "val", "valid", "test")
    for config in files:
        if config.name not in {"data.yaml", "data.yml"}:
            continue
        try:
            payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        for key in split_keys:
            raw = payload.get(key)
            if not isinstance(raw, str):
                continue
            candidate = raw.strip()
            if not candidate or re.match(r"^[a-zA-Z]+://", candidate) or "*" in candidate:
                continue
            resolved = Path(candidate)
            if not resolved.is_absolute():
                resolved = (config.parent / resolved).resolve()
            if not resolved.exists():
                rel_cfg = config.relative_to(path)
                issues.append(f"{rel_cfg}:{key} -> missing '{candidate}'")

    return issues


def yolo_pair_issues(path: Path) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    pair_count = 0

    for images_dir in path.rglob("images"):
        if not images_dir.is_dir():
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            continue
        pair_count += 1

        image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT]
        label_files = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]

        if image_files and not label_files:
            errors.append(f"{labels_dir.relative_to(path)} has no .txt labels")
            continue
        if not label_files:
            continue

        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}

        orphan_labels = label_stems - image_stems
        if orphan_labels:
            errors.append(f"{labels_dir.relative_to(path)} has {len(orphan_labels)} orphan label files")

        if image_stems:
            missing_labels = image_stems - label_stems
            if missing_labels:
                warnings.append(
                    f"{images_dir.relative_to(path)} has {len(missing_labels)}/{len(image_stems)} images without labels"
                )

        malformed = 0
        for label in sample_paths(label_files, MAX_LABEL_PROBE):
            if not parse_yolo_label_file(label):
                malformed += 1
        if malformed:
            errors.append(f"{labels_dir.relative_to(path)} has malformed YOLO labels ({malformed} files)")

    return errors, warnings, pair_count


def format_validation(path: Path, files: list[Path]) -> tuple[str, str]:
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    annotations = [f for f in files if f.suffix.lower() in ANNOT_EXT]

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    checked, broken, examples = probe_images(images)
    if checked:
        info.append(f"imgprobe={checked}")
    if broken:
        sample = f" e.g. {', '.join(examples)}" if examples else ""
        errors.append(f"Unreadable images {broken}/{checked}.{sample}")

    split_issues = yaml_split_path_issues(path, files)
    if split_issues:
        warnings.append(split_issues[0])

    pair_errors, pair_warnings, pair_count = yolo_pair_issues(path)
    if pair_count:
        info.append(f"yolo_pairs={pair_count}")
    if pair_errors:
        errors.append(pair_errors[0])
    if pair_warnings:
        warnings.append(pair_warnings[0])

    if images and not annotations:
        warnings.append("Images are present without annotations (may be valid for re-id datasets)")

    if errors:
        return "fail", errors[0]
    if warnings:
        prefix = ", ".join(info) + "; " if info else ""
        return "warn", prefix + warnings[0]

    return "pass", ", ".join(info) if info else "format checks passed"


def classify(entry: dict[str, Any], path: Path) -> tuple[str, str]:
    mode = entry.get("mode", "?")
    files = payload_files(path)

    if not path.exists():
        return "missing", "Folder does not exist"

    if mode == "manual" and not files:
        return "manual-pending", "Manual source is not downloaded yet"

    if not files:
        return "empty", "No payload files found"

    if mode == "https":
        target = https_target_file(path, entry.get("file_url"))
        expected = entry.get("expected_bytes")
        if target and target.exists() and isinstance(expected, int):
            current = target.stat().st_size
            if current < expected:
                return "partial", f"Archive is partial ({current}/{expected} bytes)"

    n_images = sum(1 for f in files if f.suffix.lower() in IMAGE_EXT)
    n_archives = sum(1 for f in files if f.suffix.lower() in ARCHIVE_EXT)
    n_ann = sum(1 for f in files if f.suffix.lower() in ANNOT_EXT)

    if n_images == 0 and n_archives == 0 and n_ann == 0:
        return "suspicious", "No recognizable image/archive/annotation files"

    fmt_status, fmt_details = format_validation(path, files)
    metrics = f"images={n_images}, archives={n_archives}, ann={n_ann}, files={len(files)}"
    if fmt_status == "fail":
        return "suspicious", f"{metrics}; {fmt_details}"
    if fmt_status == "warn":
        return "ok", f"{metrics}; {fmt_details}"
    return "ok", f"{metrics}; {fmt_details}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate local dataset payloads")
    parser.add_argument("--species", choices=(*SPECIES, "all"), default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(MANIFEST)

    species_list = list(SPECIES) if args.species == "all" else [args.species]

    hard_failures = 0
    print("Dataset payload validation")
    print("=" * 116)
    print(f"{'species':8}  {'name':34}  {'mode':11}  {'result':14}  details")
    print("-" * 116)

    for sp in species_list:
        for entry in manifest.get(sp, []):
            name = entry["name"]
            mode = entry.get("mode", "?")
            path = DATA_ROOT / sp / name
            result, details = classify(entry, path)
            print(f"{sp:8}  {name:34}  {mode:11}  {result:14}  {details}")

            if mode != "manual" and result in {"missing", "empty", "partial", "suspicious"}:
                hard_failures += 1

    print("-" * 116)
    if hard_failures:
        print(f"Non-manual dataset validation failures: {hard_failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
