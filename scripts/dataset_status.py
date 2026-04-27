"""Report installation status for datasets declared in data/datasets.yaml.

Usage:
    python scripts/dataset_status.py
    python scripts/dataset_status.py --species horse
    python scripts/dataset_status.py --strict-auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "datasets.yaml"
DATA_ROOT = ROOT / "data" / "datasets"
VALID_SPECIES = ("cattle", "sheep", "horse")


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install PyYAML") from exc

    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for species in VALID_SPECIES:
        data.setdefault(species, [])
    return data


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def payload_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file() and p.name != "README_MANUAL.txt":
            count += 1
    return count


def human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def has_real_payload(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.name != "README_MANUAL.txt":
            return True
    return False


def https_target_file(path: Path, file_url: str | None) -> Path | None:
    if not file_url:
        return None
    parsed = urlparse(file_url)
    filename = unquote(Path(parsed.path).name)
    if not filename:
        return None
    return path / filename


def classify_status(entry: dict[str, Any], path: Path) -> str:
    mode = entry.get("mode", "?")
    if not path.exists():
        return "missing"

    if mode == "https":
        target = https_target_file(path, entry.get("file_url"))
        expected = entry.get("expected_bytes")
        if target and target.exists() and isinstance(expected, int):
            try:
                current = target.stat().st_size
                if current < expected:
                    return "partial"
            except OSError:
                pass

    if mode == "manual" and not has_real_payload(path):
        return "manual-pending"
    if has_real_payload(path):
        return "ready"
    return "empty"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show dataset installation status")
    parser.add_argument(
        "--species",
        choices=(*VALID_SPECIES, "all"),
        default="all",
        help="Filter by species",
    )
    parser.add_argument(
        "--strict-auto",
        action="store_true",
        help="Return exit code 1 if any non-manual dataset is not ready",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(MANIFEST_PATH)

    if args.species == "all":
        species_list = list(VALID_SPECIES)
    else:
        species_list = [args.species]

    print("Dataset installation status")
    print("=" * 96)
    print(f"{'species':8}  {'name':34}  {'mode':11}  {'status':14}  {'files':5}  {'size':10}")
    print("-" * 96)

    auto_failures = 0

    for species in species_list:
        for entry in manifest.get(species, []):
            name = entry["name"]
            mode = entry.get("mode", "?")
            path = DATA_ROOT / species / name
            status = classify_status(entry, path)
            files = payload_file_count(path)
            size = human_size(folder_size(path))

            if args.strict_auto and mode != "manual" and status != "ready":
                auto_failures += 1

            print(f"{species:8}  {name:34}  {mode:11}  {status:14}  {files:5d}  {size:10}")

    print("-" * 96)
    if args.strict_auto and auto_failures:
        print(f"Auto datasets not ready: {auto_failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
