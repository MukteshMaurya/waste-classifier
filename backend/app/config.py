"""Runtime configuration for the FastAPI backend.

Every value is overridable through environment variables (or a ``.env`` file)
so that the same image runs unchanged on a laptop and on Render.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``backend/app/config.py`` -> ``backend`` is the deployment root.
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR: Path = BACKEND_ROOT / "models"


class Settings(BaseSettings):
    """Application settings.

    Field names map to upper-case environment variables, e.g.
    ``CONFIDENCE_THRESHOLD`` or ``FRONTEND_URL``.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- model -----------------------------------------------------------
    model_path: Path = Field(
        default=DEFAULT_MODELS_DIR / "waste_classifier.onnx",
        description="Path to the exported ONNX model.",
    )
    class_names_path: Path = Field(
        default=DEFAULT_MODELS_DIR / "class_names.json",
        description="JSON file holding class order and preprocessing spec.",
    )
    onnx_intra_op_threads: int = Field(
        default=0,
        ge=0,
        description="CPU threads for onnxruntime. 0 lets onnxruntime decide.",
    )

    # ---- confidence ------------------------------------------------------
    confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Below this the API reports it cannot identify the item.",
    )
    high_confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="At or above this the confidence level is 'High'.",
    )
    top_k: int = Field(default=3, ge=1, le=20)

    # ---- uploads ---------------------------------------------------------
    max_upload_bytes: int = Field(
        default=5 * 1024 * 1024, description="Maximum accepted upload size (5 MB)."
    )
    max_image_pixels: int = Field(
        default=25_000_000,
        description="Decompression-bomb guard for absurdly large images.",
    )
    min_image_side: int = Field(
        default=16, description="Reject images smaller than this on either side."
    )
    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    )
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    # ---- cors ------------------------------------------------------------
    frontend_url: str = Field(
        default="http://localhost:5173",
        description=(
            "Comma-separated list of allowed frontend origins. Set this to the "
            "Vercel URL in production."
        ),
    )
    cors_allow_credentials: bool = False

    # ---- api -------------------------------------------------------------
    app_name: str = "AI Waste Classifier API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    @field_validator("model_path", "class_names_path", mode="after")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("allowed_mime_types", "allowed_extensions", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept either a comma-separated string or an already-parsed tuple."""
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @property
    def allowed_origins(self) -> list[str]:
        """Normalised CORS origins from ``FRONTEND_URL``."""
        return [
            origin.strip().rstrip("/")
            for origin in self.frontend_url.split(",")
            if origin.strip()
        ]

    def confidence_level(self, confidence: float) -> str:
        """Map a confidence score onto a human label.

        ``>= high_confidence_threshold`` -> High, ``>= confidence_threshold``
        -> Moderate, otherwise Low. The thresholds are configuration, not
        constants, so they can be tuned without touching route code.
        """
        if confidence >= self.high_confidence_threshold:
            return "High"
        if confidence >= self.confidence_threshold:
            return "Moderate"
        return "Low"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (FastAPI dependency)."""
    return Settings()
