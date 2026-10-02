"""API routes: ``GET /``, ``GET /health``, ``GET /categories``, ``POST /predict``."""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.model.predictor import ModelLoadError, WasteClassifier
from app.schemas import (
    CategoriesResponse,
    ClassInfo,
    ClassPrediction,
    DisposalInfo,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    RootResponse,
)
from app.utils.image_utils import ImageValidationError, validate_upload

LOGGER = logging.getLogger(__name__)

router = APIRouter()

#: Chunk size used to read the upload without loading it all at once.
_READ_CHUNK = 1024 * 1024

#: Message shown when the top prediction is not trustworthy.
LOW_CONFIDENCE_MESSAGE = (
    "Unable to confidently identify this waste item. "
    "Try again with a clearer photo of the item on a plain background."
)

#: Extra note for the catch-all class.
UNKNOWN_CLASS_MESSAGE = (
    "The model could not confidently map this item to a supported waste category."
)


def get_classifier(request: Request) -> WasteClassifier:
    """FastAPI dependency returning the process-wide classifier."""
    return request.app.state.classifier


@lru_cache
def _load_disposal_info() -> dict[str, dict]:
    """Read the disposal reference once per process."""
    path = Path(__file__).resolve().parent.parent / "data" / "disposal_info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Could not read disposal info from %s: %s", path, exc)
        return {}


def disposal_for(class_name: str) -> DisposalInfo:
    """Look up disposal guidance, falling back to a generic entry.

    A new model class with no reference entry still returns a valid response
    instead of raising.
    """
    table = _load_disposal_info()
    entry = table.get(class_name) or table.get("_default") or {}
    return DisposalInfo(
        category=entry.get("category", "Check locally"),
        instructions=entry.get("instructions", "Please check local recycling rules."),
        icon=entry.get("icon", "❓"),
        bin_colour=entry.get("bin_colour", "Varies"),
        tips=list(entry.get("tips", [])),
    )


async def _read_limited(upload: UploadFile, settings: Settings) -> bytes:
    """Read an upload, aborting as soon as it exceeds the size limit.

    Reading in chunks means an oversized body is rejected without buffering the
    whole thing into memory.
    """
    buffer = bytearray()
    while chunk := await upload.read(_READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / (1024 * 1024)
            raise ImageValidationError(
                f"The image is larger than the {limit_mb:.0f} MB limit.", 413
            )
    return bytes(buffer)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get("/", response_model=RootResponse, tags=["meta"], summary="API information")
def read_root(
    classifier: Annotated[WasteClassifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RootResponse:
    """Basic service description and the live class list."""
    return RootResponse(
        name=settings.app_name,
        version=settings.app_version,
        description=(
            "Classifies a waste image into one of the supported categories and "
            "returns disposal guidance. CPU-only inference with ONNX Runtime."
        ),
        docs="/docs",
        model_loaded=classifier.is_loaded,
        supported_classes=classifier.class_names,
        endpoints={
            "health": "/health",
            "classes": "/categories",
            "predict": "/predict",
            "docs": "/docs",
        },
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness probe")
def read_health(
    classifier: Annotated[WasteClassifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse:
    """Report model status.

    Returns ``503`` when the model failed to load so Render surfaces the
    problem instead of routing traffic to a broken service.
    """
    loaded = classifier.is_loaded
    body = HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        classes=len(classifier.class_names),
        error=None if loaded else (classifier.load_error or "model not loaded"),
        version=settings.app_version,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(),
    )


# ---------------------------------------------------------------------------
# GET /categories
# ---------------------------------------------------------------------------


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    tags=["meta"],
    summary="Supported waste categories",
)
def read_categories(
    classifier: Annotated[WasteClassifier, Depends(get_classifier)],
) -> CategoriesResponse:
    """The classes this model can predict, with disposal guidance.

    The list comes from ``class_names.json``, so it always matches the
    deployed model.
    """
    names = classifier.class_names
    if not names:
        return CategoriesResponse(count=0, classes=[])
    classes = []
    for index, name in enumerate(names):
        info = disposal_for(name)
        classes.append(
            ClassInfo(
                name=name,
                index=index,
                category=info.category,
                icon=info.icon,
                bin_colour=info.bin_colour,
                instructions=info.instructions,
            )
        )
    return CategoriesResponse(count=len(classes), classes=classes)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Classify a waste image",
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="The waste image to classify.")],
    classifier: Annotated[WasteClassifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse | PredictionResponse:
    """Validate an upload, run inference, and return the ranked prediction.

    The image is held in memory only and is never written to disk.
    """
    if not classifier.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error": "model_unavailable",
                "message": "The classification model is not available right now.",
            },
        )

    started = time.perf_counter()
    try:
        payload = await _read_limited(file, settings)
        validated = validate_upload(
            payload=payload,
            filename=file.filename,
            content_type=file.content_type,
            settings=settings,
        )
        result = classifier.predict(validated.image)
    except ImageValidationError as exc:
        # User-correctable problems get 4xx with a safe, specific message.
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": _error_code(exc),
                "message": exc.message,
            },
        )
    except ModelLoadError:
        LOGGER.exception("Model error during prediction")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error": "model_error",
                "message": "The classification model could not process this image.",
            },
        )
    except Exception:  # noqa: BLE001 - never leak a traceback to the client
        LOGGER.exception("Unhandled error during prediction")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "internal_error",
                "message": "Something went wrong while analyzing the image.",
            },
        )
    finally:
        await file.close()

    message: str | None = None
    if not result.is_confident:
        message = LOW_CONFIDENCE_MESSAGE
    elif result.predicted_class == "Other/Unknown":
        message = UNKNOWN_CLASS_MESSAGE

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    LOGGER.info(
        "Predicted %s at %.2f%% (%s) in %.1f ms",
        result.predicted_class, result.confidence * 100,
        result.confidence_level, elapsed_ms,
    )

    return PredictionResponse(
        success=True,
        prediction=result.predicted_class,
        confidence=round(result.confidence * 100, 2),
        confidence_level=result.confidence_level,
        is_confident=result.is_confident,
        message=message,
        top_predictions=[
            ClassPrediction(name=item.class_name, confidence=round(item.confidence * 100, 2))
            for item in result.top_predictions
        ],
        disposal=disposal_for(result.predicted_class),
        image={
            "width": validated.width,
            "height": validated.height,
            "size_kb": round(validated.byte_size / 1024, 1),
        },
        model=ModelInfo(
            name=settings.model_path.name,
            num_classes=len(classifier.class_names),
            architecture=settings.class_names_path.stem,
        ).model_dump(),
        processing_time_ms=elapsed_ms,
    )


def _error_code(exc: ImageValidationError) -> str:
    """Map a validation failure to a stable machine-readable code."""
    message = exc.message.lower()
    if "larger than" in message:
        return "file_too_large"
    if "too small" in message:
        return "image_too_small"
    if "corrupt" in message or "could not be read" in message or "could not be decoded" in message:
        return "corrupt_image"
    if "empty" in message:
        return "empty_file"
    if "unsupported" in message:
        return "unsupported_file"
    return "invalid_image"
