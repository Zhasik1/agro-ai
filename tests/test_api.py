"""End-to-end API tests with a mocked ML pipeline."""

from __future__ import annotations


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_identify_without_photo_returns_422(client) -> None:
    response = client.post("/api/animals/identify")
    assert response.status_code == 422


def test_register_with_invalid_iin(client, fake_image_bytes) -> None:
    response = client.post(
        "/api/animals/register",
        files={"photo": ("a.png", fake_image_bytes, "image/png")},
        data={"owner_iin": "not-iin"},
    )
    assert response.status_code == 422


def test_register_then_identify_matches(client, fake_image_bytes) -> None:
    register = client.post(
        "/api/animals/register",
        files={"photo": ("a.png", fake_image_bytes, "image/png")},
        data={"owner_iin": "880101300123"},
    )
    assert register.status_code == 201, register.text
    animal_id = register.json()["animal"]["id"]
    assert animal_id.startswith("COW-")

    identify = client.post(
        "/api/animals/identify",
        files={"photo": ("a.png", fake_image_bytes, "image/png")},
    )
    assert identify.status_code == 200
    body = identify.json()
    assert body["status"] == "matched"
    assert body["best_match"]["animal_id"] == animal_id
    assert body["best_match"]["similarity"] >= 0.85


def test_double_register_returns_409(client, fake_image_bytes) -> None:
    payload = {
        "files": {"photo": ("a.png", fake_image_bytes, "image/png")},
        "data": {"owner_iin": "880101300123"},
    }
    first = client.post("/api/animals/register", **payload)
    assert first.status_code == 201
    second = client.post("/api/animals/register", **payload)
    assert second.status_code == 409
    assert second.json()["existing_id"] == first.json()["animal"]["id"]


def test_get_unknown_animal_returns_404(client) -> None:
    response = client.get("/api/animals/COW-DEADBE")
    assert response.status_code == 404


def test_stats_endpoint(client, fake_image_bytes) -> None:
    client.post(
        "/api/animals/register",
        files={"photo": ("a.png", fake_image_bytes, "image/png")},
        data={"owner_iin": "880101300123"},
    )
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert len(body["per_species"]) == 3
