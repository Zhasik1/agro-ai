"""Threshold classification for animal identification results.

Centralises the logic that converts a raw cosine-similarity score into a
human-readable status string.  Import :func:`classify_match_status` wherever
a threshold decision is needed instead of duplicating the numeric comparisons.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["classify_match_status"]


def classify_match_status(
    top_similarity: float,
) -> Literal["matched", "suspect", "new"]:
    """Classify a top cosine-similarity score into a match status.

    Thresholds are read from :class:`app.config.Settings` at call time so
    they can be overridden via environment variables without code changes.

    Args:
        top_similarity: The highest cosine-similarity score returned by the
            vector search (range −1 … 1, but in practice 0 … 1 for
            L2-normalised embeddings).

    Returns:
        ``"matched"`` if *top_similarity* ≥ ``MATCH_THRESHOLD``,
        ``"suspect"`` if *top_similarity* ≥ ``SUSPECT_THRESHOLD``,
        ``"new"`` otherwise.
    """
    from app.config import get_settings

    settings = get_settings()
    if top_similarity >= settings.MATCH_THRESHOLD:
        return "matched"
    if top_similarity >= settings.SUSPECT_THRESHOLD:
        return "suspect"
    return "new"
