"""Download / document verified livestock biometric datasets.

Reads ``data/datasets.yaml`` and either auto-downloads each entry or prints
clear manual instructions. Never bypasses authentication, never embeds API
keys, never invents URLs beyond what is in the YAML manifest.

Usage::

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --download cattle
    python scripts/download_datasets.py --download all
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("datasets")

DEFAULT_MANIFEST = ROOT / "data" / "datasets.yaml"
DEFAULT_TARGET = ROOT / "data" / "datasets"

SPECIES = ("cattle", "sheep", "horse")


# --------------------------------------------------------------------------- #
# Manifest loader
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse the YAML manifest into a dict of species -> entries."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required. Install it with: pip install PyYAML"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for species in SPECIES:
        data.setdefault(species, [])
    return data


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def target_dir(base: Path, species: str, name: str) -> Path:
    """Return the on-disk folder for a dataset entry."""
    return base / species / name


def human_size(num_bytes: int) -> str:
    """Format byte count for humans."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def folder_size(path: Path) -> int:
    """Return the recursive size in bytes of ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- #
# Mode handlers — each returns True on success, False on failure.
# --------------------------------------------------------------------------- #
def handle_huggingface(entry: dict[str, Any], dest: Path) -> bool:
    """Download via the HuggingFace ``datasets`` library."""
    hf_id = entry.get("hf_id")
    if not hf_id:
        logger.error("[%s] missing hf_id", entry["name"])
        return False
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "[%s] HuggingFace 'datasets' library not installed.\n"
            "  Install it: pip install datasets\n"
            "  Or download manually: %s",
            entry["name"], entry.get("url", f"https://huggingface.co/datasets/{hf_id}"),
        )
        return False

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] load_dataset(%r) -> %s", entry["name"], hf_id, dest)
    try:
        ds = load_dataset(hf_id)
        ds.save_to_disk(str(dest))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] load_dataset failed (%s). Trying snapshot_download fallback...",
            entry["name"], exc,
        )

    # Some public datasets still rely on legacy dataset scripts. In that case,
    # mirror repository files directly from the Hub as a practical fallback.
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "[%s] fallback requires huggingface_hub. Install with: pip install huggingface-hub",
            entry["name"],
        )
        return False

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] snapshot_download failed: %s", entry["name"], exc)
        return False
    return True


def handle_git(entry: dict[str, Any], dest: Path) -> bool:
    """Clone a git repository (shallow)."""
    if shutil.which("git") is None:
        logger.error("[%s] git executable not found in PATH", entry["name"])
        return False
    if dest.exists() and any(dest.iterdir()):
        logger.info("[%s] already present at %s — skipping", entry["name"], dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] git clone --depth 1 %s -> %s", entry["name"], entry["url"], dest)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", entry["url"], str(dest)],
        check=False,
    )
    if result.returncode != 0:
        logger.error("[%s] git clone failed (code %d)", entry["name"], result.returncode)
        return False
    return True


def handle_https(entry: dict[str, Any], dest: Path) -> bool:
    """Stream-download a single archive from ``file_url``."""
    file_url = entry.get("file_url")
    if not file_url:
        logger.error(
            "[%s] mode=https requires 'file_url' in datasets.yaml — got: %s",
            entry["name"], entry.get("url"),
        )
        return False

    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "[%s] 'requests' not installed. pip install requests", entry["name"]
        )
        return False

    dest.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(file_url)
    filename = unquote(Path(parsed.path).name) or "download.bin"
    out = dest / filename
    if out.exists():
        logger.info("[%s] already downloaded at %s — skipping", entry["name"], out)
        return True

    logger.info("[%s] HTTPS GET %s", entry["name"], file_url)
    try:
        with requests.get(file_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with out.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written * 100 / total
                        sys.stderr.write(
                            f"\r  {human_size(written)}/{human_size(total)} ({pct:5.1f}%)"
                        )
                        sys.stderr.flush()
            sys.stderr.write("\n")
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] download failed: %s", entry["name"], exc)
        return False
    return True


def handle_manual(entry: dict[str, Any], dest: Path) -> bool:
    """Print clear instructions; never auto-download manual entries."""
    name = entry["name"]
    url = entry.get("url", "<no url>")
    license_ = entry.get("license", "see-source")
    note = entry.get("note", "")

    msg = [
        "─" * 72,
        f"📋 [{name}] MANUAL — opens authentication, license, or paper page",
        f"   URL     : {url}",
        f"   License : {license_}",
        f"   Target  : {dest}",
    ]
    if note:
        msg.append(f"   Note    : {note}")

    msg += [
        "",
        "   Steps:",
        f"   1) Open the URL in a browser, accept the license.",
        f"   2) Download the dataset files into: {dest}",
        f"   3) (Optional) re-run this script — already-present entries are skipped.",
    ]

    if "roboflow" in url.lower():
        api_key_present = bool(os.environ.get("ROBOFLOW_API_KEY"))
        msg += [
            "",
            "   Roboflow API hint (set ROBOFLOW_API_KEY in your environment):",
            "     curl -L \"<dataset-export-url>?api_key=$ROBOFLOW_API_KEY\" "
            "-o roboflow.zip",
            f"   ROBOFLOW_API_KEY currently {'set' if api_key_present else 'NOT set'}.",
        ]
    elif "huggingface" in url.lower():
        msg += [
            "",
            "   HuggingFace hint (private datasets need HUGGINGFACE_HUB_TOKEN):",
            "     huggingface-cli login",
        ]
    elif "kaggle" in url.lower():
        msg += [
            "",
            "   Kaggle hint: place ~/.kaggle/kaggle.json then:",
            "     kaggle datasets download <slug> -p <dest>",
        ]

    msg.append("─" * 72)
    print("\n".join(msg))
    return True


HANDLERS = {
    "huggingface": handle_huggingface,
    "git": handle_git,
    "https": handle_https,
    "manual": handle_manual,
}


# --------------------------------------------------------------------------- #
# Top-level commands
# --------------------------------------------------------------------------- #
def cmd_list(manifest: dict[str, list[dict[str, Any]]]) -> int:
    """Print the manifest in a readable form."""
    print("📚 MalChain dataset manifest")
    print("=" * 72)
    for species in SPECIES:
        entries = manifest.get(species, [])
        print(f"\n🐾 {species.upper()}  ({len(entries)} entries)")
        for entry in entries:
            mode = entry.get("mode", "?")
            license_ = entry.get("license", "?")
            url = entry.get("url", "")
            print(f"  • {entry['name']:<32} mode={mode:<11} license={license_}")
            print(f"    {url}")
    print("\n💡 Roadmap (no hardcoded URLs): goat / camel / pig — see docs/DATASETS.md")
    return 0


def cmd_download(
    manifest: dict[str, list[dict[str, Any]]],
    species_arg: str,
    target_root: Path,
) -> int:
    """Download every entry under ``species_arg`` (or 'all')."""
    target_root.mkdir(parents=True, exist_ok=True)

    if species_arg == "all":
        species_list = list(SPECIES)
    elif species_arg in SPECIES:
        species_list = [species_arg]
    else:
        logger.error("Unknown species: %s. Allowed: %s, all", species_arg, SPECIES)
        return 1

    failures = 0
    for species in species_list:
        for entry in manifest.get(species, []):
            mode = entry.get("mode")
            handler = HANDLERS.get(mode or "")
            if handler is None:
                logger.error("[%s] unknown mode=%r", entry.get("name"), mode)
                failures += 1
                continue
            dest = target_dir(target_root, species, entry["name"])
            ok = handler(entry, dest)
            size = folder_size(dest)
            logger.info(
                "[%s] mode=%s status=%s on-disk=%s path=%s",
                entry["name"], mode, "ok" if ok else "fail", human_size(size), dest,
            )
            if not ok:
                failures += 1
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Path to datasets.yaml (default: data/datasets.yaml)",
    )
    parser.add_argument(
        "--target-dir", type=Path, default=DEFAULT_TARGET,
        help="Root folder for downloaded datasets (default: data/datasets/)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List manifest entries.")
    group.add_argument(
        "--download", choices=(*SPECIES, "all"),
        help="Download all entries for a species (or 'all').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 2
    manifest = load_manifest(args.manifest)
    if args.list:
        return cmd_list(manifest)
    return cmd_download(manifest, args.download, args.target_dir)


if __name__ == "__main__":
    raise SystemExit(main())
