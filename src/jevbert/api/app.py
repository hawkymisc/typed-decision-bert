"""Application factory and lifespan (POC_DESIGN 3.1, 4.6)."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from jevbert import __version__
from jevbert.api.errors import (
    JevBERTError,
    http_exception_handler,
    jevbert_error_handler,
    request_validation_handler,
)
from jevbert.api.middleware import RequestContextMiddleware
from jevbert.api.routes import router
from jevbert.config import Settings
from jevbert.inference.engine import InferenceEngine
from jevbert.inference.registry import ModelRegistry

logger = logging.getLogger("jevbert")


def create_app(
    settings: Settings,
    registry: ModelRegistry,
    *,
    engine: InferenceEngine | None = None,
    load_on_startup: bool = True,
) -> FastAPI:
    """Build the ASGI application.

    Bundles load in the background so that ``/readyz`` answers 503 until warmup has
    finished rather than blocking startup (spec 16.1). ``load_on_startup=False`` lets a
    test drive readiness itself.
    """
    if not settings.api_keys:
        raise ValueError("create_app requires at least one API key; there is no open mode")

    engine = engine or InferenceEngine(
        max_pending_requests=settings.serving.max_pending_requests,
        request_deadline_seconds=settings.serving.request_deadline_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        loader: threading.Thread | None = None
        if load_on_startup:
            loader = threading.Thread(
                target=registry.load_all, name="jevbert-loader", daemon=True
            )
            loader.start()
        try:
            yield
        finally:
            engine.shutdown(wait=False)
            if loader is not None:
                loader.join(timeout=5.0)
                if loader.is_alive():
                    logger.warning("bundle loading was still running at shutdown")

    app = FastAPI(
        title="JevBERT",
        version=__version__,
        lifespan=lifespan,
        # The interactive docs would advertise a schema this server does not use for
        # parsing; the contract lives in contracts/schemas instead.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.engine = engine

    app.add_exception_handler(JevBERTError, jevbert_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)

    app.include_router(router)
    app.add_middleware(RequestContextMiddleware, contract_profile=settings.contract_profile)
    return app
