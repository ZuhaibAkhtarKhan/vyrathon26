"""Grabpic — Intelligent Identity & Retrieval Engine.

FastAPI application entrypoint. Wires routers, exception handlers, and
OpenAPI metadata.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import get_settings
from app.database import init_db
from app.routers import auth, grabs, health, images, ingest
from app.schemas import ErrorPayload, ErrorResponse
from app.utils.errors import GrabpicError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("grabpic")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Booting Grabpic v%s (face_engine=%s)", __version__, settings.face_engine)
    init_db()
    yield
    logger.info("Grabpic shutting down.")


def _error_response(status: int, code: str, message: str, details=None) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorPayload(code=code, message=message, details=details)
    )
    return JSONResponse(status_code=status, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GrabpicError)
    async def grabpic_error_handler(_: Request, exc: GrabpicError) -> JSONResponse:
        return _error_response(exc.http_status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status=422,
            code="VALIDATION_ERROR",
            message="Request payload failed validation.",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _error_response(
            status=exc.status_code,
            code="HTTP_ERROR",
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return _error_response(
            status=500,
            code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
        )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Grabpic — Intelligent Identity & Retrieval Engine",
        description=(
            "Facial-recognition-powered image indexing and retrieval.\n\n"
            "**Typical flow:**\n"
            "1. `POST /ingest/scan` — crawl storage and index photos.\n"
            "2. `POST /auth/selfie` — user authenticates with a selfie and "
            "receives a `grab_id`.\n"
            "3. `GET /grabs/{grab_id}/images` — fetch all the photos that "
            "person appears in.\n"
        ),
        version=__version__,
        lifespan=lifespan,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
    )

    register_exception_handlers(app)

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(ingest.router, prefix=settings.api_prefix)
    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(grabs.router, prefix=settings.api_prefix)
    app.include_router(images.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "Grabpic",
            "version": __version__,
            "docs": f"{settings.api_prefix}/docs",
            "redoc": f"{settings.api_prefix}/redoc",
            "openapi": f"{settings.api_prefix}/openapi.json",
        }

    return app


app = create_app()
