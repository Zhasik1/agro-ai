"""Tests for the /health and / root endpoints."""

from __future__ import annotations


def test_health_returns_ok(client) -> None:
    """GET /health must return 200 with status=ok and a version string."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert isinstance(body["version"], str)


def test_root_returns_name(client) -> None:
    """GET / must return 200 with name=MalChain."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "MalChain"
    assert "version" in body
