"""Download and warm-cache the ML weights used by MalChain."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/download_models.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def download_yolo(model_dir: Path) -> Path:
    """Download YOLOv8n weights into ``model_dir``."""
    from ultralytics import YOLO  # type: ignore

    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "yolov8n.pt"
    print(f"[1/2] YOLOv8n -> {target}")
    # `YOLO("yolov8n.pt")` auto-downloads if missing; copy to model_dir.
    yolo = YOLO("yolov8n.pt")
    src = Path(yolo.ckpt_path) if hasattr(yolo, "ckpt_path") and yolo.ckpt_path else None
    if src and src.exists() and src != target:
        target.write_bytes(src.read_bytes())
    return target


def warm_resnet50() -> None:
    """Force torchvision to download the ResNet50 weights into its cache."""
    print("[2/2] ResNet50 (ImageNet) — warming torchvision cache...")
    from torchvision import models  # type: ignore

    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)


def main() -> None:
    settings = get_settings()
    yolo_path = download_yolo(settings.MODEL_PATH)
    if yolo_path.exists():
        print(f"  ✓ {yolo_path.name}: {_human_size(yolo_path.stat().st_size)}")
    warm_resnet50()
    print("✅ Барлық модельдер дайын / All models ready.")


if __name__ == "__main__":
    main()
