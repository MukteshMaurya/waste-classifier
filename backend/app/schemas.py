"""Pydantic response models.

These define the public API contract. Field names are stable - the frontend
depends on them - so they are only changed deliberately.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    """One entry in the ranked prediction list."""

    model_config = {"json_schema_extra": {"example": {"class": "Plastic", "confidence": 94.32}}}

    name: str = Field(..., alias="class", description="Waste category name.")
    confidence: float = Field(
        ..., description="Confidence percentage, 0-100, rounded to 2 decimals."
    )


class DisposalInfo(BaseModel):
    """How to dispose of the detected item."""

    category: str = Field(..., description="e.g. Recyclable, Compostable, Hazardous.")
    instructions: str = Field(..., description="Plain-language disposal guidance.")
    icon: str = Field(..., description="Emoji shown in the UI.")
    bin_colour: str = Field(..., description="Typical bin or drop-off point.")
    tips: list[str] = Field(default_factory=list, description="Extra practical notes.")


class PredictionResponse(BaseModel):
    """Successful ``POST /predict`` response."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "prediction": "Plastic",
                "confidence": 94.32,
                "confidence_level": "High",
                "is_confident": True,
                "message": None,
                "top_predictions": [
                    {"class": "Plastic", "confidence": 94.32},
                    {"class": "Glass", "confidence": 3.12},
                    {"class": "Metal", "confidence": 1.24},
                ],
                "disposal": {
                    "category": "Recyclable",
                    "instructions": "Rinse the container and place it in the plastic bin.",
                    "icon": "🧴",
                    "bin_colour": "Yellow / Plastic",
                    "tips": ["Check the resin code on the container."],
                },
                "image": {"width": 1024, "height": 768, "size_kb": 245.3},
                "model": {"name": "waste_classifier.onnx", "num_classes": 10},
                "processing_time_ms": 41.2,
            }
        }
    }

    success: bool = True
    prediction: str = Field(..., description="Top predicted waste category.")
    confidence: float = Field(..., ge=0.0, le=100.0)
    confidence_level: str = Field(..., description="High, Moderate or Low.")
    is_confident: bool = Field(
        ..., description="False when confidence is below the configured threshold."
    )
    message: str | None = Field(
        default=None,
        description="Set when the model is not confident enough to commit to a label.",
    )
    top_predictions: list[ClassPrediction]
    disposal: DisposalInfo
    image: dict[str, float] = Field(..., description="Dimensions of the uploaded image.")
    model: dict[str, object] = Field(..., description="Which model produced this.")
    processing_time_ms: float


class ImageInfo(BaseModel):
    """Metadata about the uploaded image."""

    width: int
    height: int
    size_kb: float


class ModelInfo(BaseModel):
    """Model identity returned to the client."""

    name: str
    num_classes: int
    architecture: str | None = None


class HealthResponse(BaseModel):
    """``GET /health`` response."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "classes": 10,
                "error": None,
                "version": "1.0.0",
            }
        }
    }

    status: str = Field(..., description="healthy, degraded or unhealthy.")
    model_loaded: bool
    classes: int
    error: str | None = None
    version: str


class ClassInfo(BaseModel):
    """A single supported waste category."""

    name: str
    index: int
    category: str
    icon: str
    bin_colour: str
    instructions: str


class CategoriesResponse(BaseModel):
    """``GET /categories`` response - drives the UI's category list."""

    count: int
    classes: list[ClassInfo]


class ErrorResponse(BaseModel):
    """Uniform error body. Never contains a stack trace."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": "unsupported_file",
                "message": "Unsupported file type '.gif'. Allowed types: .jpg, .jpeg, .png, .webp.",
            }
        }
    }

    success: bool = False
    error: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable, safe to display.")


class RootResponse(BaseModel):
    """``GET /`` response."""

    name: str
    version: str
    description: str
    docs: str
    model_loaded: bool
    supported_classes: list[str]
    endpoints: dict[str, str]
