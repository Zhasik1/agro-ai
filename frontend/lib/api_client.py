"""Typed HTTP helpers for communicating with the MalChain backend.

All helpers raise :class:`BackendError` on non-2xx responses so that
Streamlit pages can handle errors in a single ``except`` block.
The backend URL is read from the ``BACKEND_URL`` environment variable
(default: ``http://localhost:8000``).
"""

from __future__ import annotations

import os

import requests

__all__ = ["BackendError", "by_owner", "identify", "register", "stats"]

BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")


class BackendError(Exception):
    """Raised when the backend returns a non-2xx HTTP response.

    Attributes:
        status: HTTP status code returned by the backend.
        detail: Human-readable error message extracted from the response body.
        data: Full parsed JSON body (may be empty dict for non-JSON responses).
    """

    def __init__(self, status: int, detail: str, data: dict | None = None) -> None:
        self.status = status
        self.detail = detail
        self.data: dict = data if data is not None else {}
        super().__init__(f"Backend error {status}: {detail}")


def _check(resp: requests.Response) -> requests.Response:
    """Raise :class:`BackendError` for non-2xx responses.

    Args:
        resp: The :class:`~requests.Response` to check.

    Returns:
        The same response object if status is 2xx.

    Raises:
        BackendError: On any non-2xx status code.
    """
    if not resp.ok:
        data: dict = {}
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            pass
        detail = str(data.get("detail") or data.get("message") or resp.text)
        raise BackendError(resp.status_code, detail, data)
    return resp


def identify(photo_bytes: bytes) -> dict:
    """Run the identification pipeline on a photo.

    Args:
        photo_bytes: Raw image bytes (JPEG or PNG).

    Returns:
        Parsed JSON response containing ``status``, ``species``,
        ``detection_confidence``, and ``candidates``.

    Raises:
        BackendError: On non-2xx HTTP response.
        requests.RequestException: On network / timeout errors.
    """
    resp = requests.post(
        f"{BACKEND_URL}/api/animals/identify",
        files={"photo": ("photo.jpg", photo_bytes, "image/jpeg")},
        timeout=120,
    )
    return _check(resp).json()


def register(
    photo_bytes: bytes,
    owner_iin: str,
    age: int = 0,
    weight: float = 0.0,
    breed: str | None = None,
    notes: str | None = None,
) -> dict:
    """Register a new animal with the backend.

    Args:
        photo_bytes: Raw image bytes (JPEG or PNG).
        owner_iin: 12-digit Kazakh IIN of the owner.
        age: Age of the animal in years (0 = unknown).
        weight: Weight of the animal in kilograms (0 = unknown).
        breed: Optional breed string.
        notes: Optional free-text notes.

    Returns:
        Parsed JSON response containing the ``animal`` record and
        ``detection_confidence``.

    Raises:
        BackendError: On non-2xx HTTP response (including 409 duplicate).
        requests.RequestException: On network / timeout errors.
    """
    files = {"photo": ("photo.jpg", photo_bytes, "image/jpeg")}
    data: dict[str, str] = {"owner_iin": owner_iin}
    if age:
        data["age_years"] = str(int(age))
    if weight:
        data["weight_kg"] = str(weight)
    if breed:
        data["breed"] = breed
    if notes:
        data["notes"] = notes
    resp = requests.post(
        f"{BACKEND_URL}/api/animals/register",
        files=files,
        data=data,
        timeout=120,
    )
    return _check(resp).json()


def stats() -> dict:
    """Fetch per-species registration statistics.

    Returns:
        Parsed JSON response containing ``total``, ``per_species``, and
        ``recent`` fields.

    Raises:
        BackendError: On non-2xx HTTP response.
        requests.RequestException: On network / timeout errors.
    """
    resp = requests.get(f"{BACKEND_URL}/api/stats", timeout=10)
    return _check(resp).json()


def by_owner(iin: str) -> list[dict]:
    """Fetch all animals belonging to an owner.

    Args:
        iin: 12-digit Kazakh IIN.

    Returns:
        List of parsed animal JSON objects.

    Raises:
        BackendError: On non-2xx HTTP response.
        requests.RequestException: On network / timeout errors.
    """
    resp = requests.get(f"{BACKEND_URL}/api/animals/by-owner/{iin}", timeout=10)
    return _check(resp).json()
