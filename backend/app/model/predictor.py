"""ONNX Runtime inference wrapper.

The session is created once at start-up and reused for every request. Nothing
here trains, fine-tunes or mutates the model - the deployed API only *loads*
the artefact produced by ``training/export_model.py``.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import Settings
from app.model.preprocessing import PreprocessSpec, preprocess_image, softmax

LOGGER = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the ONNX model or its class manifest cannot be loaded."""


@dataclass(frozen=True)
class Prediction:
    """A single ranked class prediction."""

    class_name: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence * 100.0, 2),
        }


@dataclass(frozen=True)
class PredictionResult:
    """Full result of one inference call."""

    top_predictions: list[Prediction]
    confidence: float
    predicted_class: str
    confidence_level: str
    is_confident: bool


class WasteClassifier:
    """Loads the ONNX graph and its class manifest, then predicts from images."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: ort.InferenceSession | None = None
        self._lock = threading.Lock()
        self._class_names: list[str] = []
        self._spec = PreprocessSpec()
        self._input_name = "input"
        self._load_error: str | None = None

    # -- loading ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def spec(self) -> PreprocessSpec:
        return self._spec

    def load(self) -> None:
        """Load the model and class manifest. Idempotent.

        Failures are recorded on ``self._load_error`` rather than raised, so the
        service can still start and report a degraded ``/health`` instead of
        crash-looping on Render.
        """
        with self._lock:
            if self._session is not None:
                return
            try:
                self._class_names = self._read_class_names(
                    self._settings.class_names_path
                )
                self._spec = self._read_spec(self._settings.class_names_path)
                session = self._create_session(self._settings.model_path)
                self._session = session
                self._input_name = session.get_inputs()[0].name
                self._load_error = None
                LOGGER.info(
                    "Model loaded: %s (%d classes, %s)",
                    self._settings.model_path.name,
                    len(self._class_names),
                    ", ".join(self._class_names),
                )
            except Exception as exc:  # noqa: BLE001 - reported via /health
                self._load_error = f"{type(exc).__name__}: {exc}"
                LOGGER.error("Failed to load model: %s", self._load_error)

    @staticmethod
    def _read_class_names(path: Path) -> list[str]:
        if not path.is_file():
            raise ModelLoadError(f"Class manifest not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data.get("class_names")
        if not isinstance(names, list) or not names:
            raise ModelLoadError(f"'{path}' has no non-empty 'class_names' list")
        if not all(isinstance(name, str) and name for name in names):
            raise ModelLoadError(f"'{path}' contains a non-string or empty class name")
        return names

    def _read_spec(self, path: Path) -> PreprocessSpec:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PreprocessSpec.from_mapping(data)

    @staticmethod
    def _create_session(model_path: Path) -> ort.InferenceSession:
        if not model_path.is_file():
            raise ModelLoadError(f"ONNX model not found: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if model_path.stat().st_size == 0:
            raise ModelLoadError(f"ONNX model is empty: {model_path}")
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        LOGGER.info(
            "ONNX input %s%s -> output %s%s",
            input_meta.name, input_meta.shape,
            output_meta.name, output_meta.shape,
        )
        return session

    def validate_against_manifest(self) -> None:
        """Confirm the graph's output width matches the class manifest."""
        if self._session is None:
            raise ModelLoadError("Session not loaded")
        output_shape = self._session.get_outputs()[0].shape
        declared = output_shape[-1]
        if isinstance(declared, int) and declared != len(self._class_names):
            raise ModelLoadError(
                f"ONNX produces {declared} logits but the manifest lists "
                f"{len(self._class_names)} class names"
            )

    # -- inference --------------------------------------------------------

    def predict(self, image: Image.Image) -> PredictionResult:
        """Run inference on a PIL image and return the ranked result.

        Raises
        ------
        ModelLoadError
            If the model is not loaded.
        """
        session = self._session
        if session is None:
            raise ModelLoadError(self._load_error or "Model is not loaded")

        batch = preprocess_image(image, self._spec)
        logits = session.run(None, {self._input_name: batch})[0]
        probabilities = softmax(np.asarray(logits, dtype=np.float32))[0]

        if probabilities.shape[0] != len(self._class_names):
            raise ModelLoadError(
                f"Model returned {probabilities.shape[0]} probabilities for "
                f"{len(self._class_names)} classes"
            )

        order = np.argsort(probabilities)[::-1]
        top_k = min(self._settings.top_k, len(self._class_names))
        ranked = [
            Prediction(
                class_name=self._class_names[int(index)],
                confidence=float(probabilities[index]),
            )
            for index in order[:top_k]
        ]

        best = ranked[0]
        return PredictionResult(
            top_predictions=ranked,
            confidence=best.confidence,
            predicted_class=best.class_name,
            confidence_level=self._settings.confidence_level(best.confidence),
            is_confident=best.confidence >= self._settings.confidence_threshold,
        )
