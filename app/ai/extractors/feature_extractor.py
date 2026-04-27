"""ResNet50 feature extractor producing L2-normalised 512-dim embeddings.

The MVP uses ImageNet-pretrained ResNet50 with the final classification head
removed (2048-dim avgpool features) followed by a *seeded random projection*
to 512 dims. This keeps the embedding compact for FAISS while remaining
deterministic between runs. The production version will replace the
projection with a triplet-loss-trained head on muzzle / face datasets.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision import models

from app.config import get_settings
from app.core import accel

__all__ = ["FeatureExtractor", "get_feature_extractor", "extract_embedding"]

logger = logging.getLogger(__name__)

_PROJECTION_SEED = 42


class FeatureExtractor:
    """ResNet50-based encoder producing L2-normalised vectors."""

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self.device = torch.device("cpu")

        logger.info("Loading ResNet50 (ImageNet weights)...")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the classifier; keep avgpool output (2048-dim).
        backbone.fc = nn.Identity()
        backbone.eval()
        self.backbone = backbone.to(self.device)

        # Seeded random projection 2048 -> embedding_dim for determinism.
        generator = torch.Generator().manual_seed(_PROJECTION_SEED)
        projection = nn.Linear(2048, embedding_dim, bias=False)
        with torch.no_grad():
            projection.weight.copy_(
                torch.empty(embedding_dim, 2048).normal_(
                    mean=0.0, std=1.0 / (2048**0.5), generator=generator
                )
            )
        projection.eval()
        self.projection = projection.to(self.device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised embedding for one BGR image.

        Args:
            image_bgr: BGR ndarray (e.g. an animal crop from YOLO).

        Returns:
            ``(embedding_dim,)`` float32 numpy array, L2-normalised.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to FeatureExtractor")

        rgb = accel.bgr_to_rgb(image_bgr)
        resized = accel.resize_bilinear(rgb, 224, 224)
        chw = accel.normalize_imagenet(resized)
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)
        feats = self.backbone(tensor)  # (1, 2048)
        proj = self.projection(feats)  # (1, embedding_dim)
        emb = proj.squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb


_lock = threading.Lock()
_instance: Optional[FeatureExtractor] = None


def get_feature_extractor() -> FeatureExtractor:
    """Return the lazy singleton :class:`FeatureExtractor`."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FeatureExtractor(embedding_dim=get_settings().EMBEDDING_DIM)
    return _instance


def extract_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Convenience wrapper around the singleton."""
    return get_feature_extractor().extract(image_bgr)
