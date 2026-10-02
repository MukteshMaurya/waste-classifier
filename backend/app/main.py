"""FastAPI application factory for the AI Waste Classifier.

Start locally with::

    uvicorn app.main:app --reload --port 8000

Start in production (Render) with::

    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.model.predictor import ModelLoadError, WasteClassifier
from app.utils.image_utils import ImageValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

DESCRIPTION = """
Classify a waste image into a disposal category.

**How it works**

1. `POST /predict` with a `multipart/form-data` field named `file`.
2. The upload is validated (type, size, decodability) and held **in memory only**.
3. A MobileNetV3-Small ONNX model classifies it on the CPU.
4. The response carries the predicted category, a confidence score, the top-3
   classes and disposal guidance.

If the model is not confident enough the response still returns the best
guess, but `is_confident` is `false` and `message` explains that the result
should not be relied on. The service never overstates its certainty.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at start-up, before serving traffic.

    A load failure is logged and reflected in ``/health`` rather than raised:
    the process stays up so the platform can show *why* it is unhealthy
    instead of crash-looping.
    """
    settings: Settings = app.state.settings
    classifier: WasteClassifier = app.state.classifier

    LOGGER.info("Starting %s v%s", settings.app_name, settings.app_version)
    LOGGER.info("Model path: %s", settings.model_path)
    LOGGER.info("Allowed origins: %s", settings.allowed_origins or "(none)")

    classifier.load()
    if classifier.is_loaded:
        try:
            classifier.validate_against_manifest()
            LOGGER.info("Model ready with %d classes", len(classifier.class_names))
        except ModelLoadError as exc:
            LOGGER.error("Model/manifest mismatch: %s", exc)
    else:
        LOGGER.error("Model unavailable: %s", classifier.load_error)

    yield
    LOGGER.info("Shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application. Exposed for tests."""
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved
    app.state.classifier = WasteClassifier(resolved)

    origins = resolved.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=resolved.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=["Content-Length"],
        max_age=600,
    )
    if not origins:
        LOGGER.warning("FRONTEND_URL is empty - all cross-origin requests are blocked")

    app.include_router(router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log method, path, status and duration for every request."""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "%s %s -> %d (%.1f ms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response

    @app.exception_handler(ImageValidationError)
    async def handle_validation_error(
        request: Request, exc: ImageValidationError
    ) -> JSONResponse:
        """Safety net: return a clean 4xx body instead of a stack trace.

        ``POST /predict`` already handles its own validation failures; this
        catches the same error type raised anywhere else in the app.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": "invalid_image", "message": exc.message},
        )

    @app.exception_handler(404)
    async def handle_not_found(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": "not_found",
                "message": "That endpoint does not exist. See /docs for the API.",
            },
        )

    return app


app = create_app()
