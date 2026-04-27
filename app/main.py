"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import animals as animals_router
from app.api import stats as stats_router
from app.config import get_settings
from app.core import accel
from app.database.session import init_db
from app.exceptions import MalChainError

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)

    application = FastAPI(
        title="MalChain API",
        version=__version__,
        description=(
            "Multi-species livestock biometric identification platform. "
            "Әр мал — паспорт. Әр паспорт — мүмкіндік."
        ),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(animals_router.router)
    application.include_router(stats_router.router)

    @application.on_event("startup")
    def _on_startup() -> None:  # pragma: no cover - trivial
        settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        settings.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        init_db()
        logger.info("Acceleration backend: %s", accel.backend())
        logger.info("MalChain v%s started", __version__)

    @application.exception_handler(MalChainError)
    async def malchain_exception_handler(  # noqa: ANN001
        _request: Request, exc: MalChainError
    ) -> JSONResponse:
        payload: dict[str, object] = {
            "type": exc.code,
            "title": exc.__class__.__name__,
            "status": exc.status_code,
            "detail": exc.message,
        }
        # Surface domain-specific extras
        for attr in ("existing_id", "similarity"):
            if hasattr(exc, attr):
                payload[attr] = getattr(exc, attr)
        return JSONResponse(status_code=exc.status_code, content=payload)

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    @application.get("/", tags=["meta"], include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "MalChain", "version": __version__, "docs": "/docs"}

    return application


app = create_app()
