"""Export the best PyTorch checkpoint to ONNX for CPU deployment.

Writes two files that the FastAPI backend loads at start-up:

* ``backend/models/waste_classifier.onnx``   - the network itself
* ``backend/models/class_names.json``        - class order + preprocessing spec

The PyTorch checkpoint is never deleted or modified.

The export also runs an internal parity check: the ONNX session and the
PyTorch model are run over real test images and their probability vectors are
compared. A mismatch above ``--max-prob-diff`` aborts with a non-zero exit
code, so a broken export can never reach the backend.

Usage::

    python training/export_model.py
    python training/export_model.py --checkpoint artifacts/checkpoints/last_model.pth
    python training/export_model.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # allow `python training/export_model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (  # noqa: E402
    BACKEND_MODELS_DIR,
    CANONICAL_CLASSES,
    CHECKPOINTS_DIR,
    CLASS_NAMES_FILENAME,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ONNX_FILENAME,
    PREPARED_DIR,
    SUPPORTED_ARCHITECTURES,
    TrainingConfig,
    ensure_directories,
)
from training.dataset import build_dataloader, build_dataset  # noqa: E402
from training.models import build_model  # noqa: E402

LOGGER = logging.getLogger("export")

#: ONNX opset 13 is the safest common denominator for onnxruntime >= 1.14.
DEFAULT_OPSET = 13


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        payload.get("architecture", TrainingConfig.architecture),
        payload.get("num_classes", len(CANONICAL_CLASSES)),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def export_onnx(
    model: torch.nn.Module, output_path: Path, image_size: int, opset: int
) -> None:
    """Trace the model to ONNX with a fixed ``[1, 3, size, size]`` input."""
    dummy_input = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed batch of 1 keeps onnxruntime simple + fast
    )
    LOGGER.info("Wrote %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)

    # onnxslim strips dead nodes left over from tracing; run it only if present.
    try:
        import onnxslim

        slimmed = output_path.with_suffix(".slim.onnx")
        onnxslim.slim(str(output_path), str(slimmed))
        slimmed.replace(output_path)
        LOGGER.info("Slimmed to %.2f MB", output_path.stat().st_size / 1e6)
    except ImportError:  # pragma: no cover - optional
        LOGGER.info("onnxslim not installed - keeping the original graph")


def write_class_names(
    output_path: Path,
    image_size: int,
    payload: dict,
) -> None:
    """Write the class order plus everything the backend needs to preprocess.

    The backend reads ``image_size`` and the normalisation statistics from this
    file, so the training and serving pipelines can never drift apart.
    """
    manifest = {
        "class_names": CANONICAL_CLASSES,
        "num_classes": len(CANONICAL_CLASSES),
        "image_size": image_size,
        "architecture": payload.get("architecture", TrainingConfig.architecture),
        "architecture_display": SUPPORTED_ARCHITECTURES.get(
            payload.get("architecture", TrainingConfig.architecture),
            payload.get("architecture", TrainingConfig.architecture),
        ),
        "input_name": "input",
        "output_name": "logits",
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "resample_filter": "bilinear",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": payload.get("_checkpoint_name", "best_model.pth"),
        "training_epoch": payload.get("epoch"),
        "validation_macro_f1": payload.get("val_macro_f1"),
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", output_path)


def verify_parity(
    model: torch.nn.Module, onnx_path: Path, num_images: int = 40
) -> dict:
    """Compare PyTorch and ONNX on identical preprocessed test images.

    The dataset transform is the *same* code the backend uses conceptually:
    resize to ``int(size * 1.14)``, centre crop, normalise. Because both
    runtimes consume the exact same float array, any difference comes purely
    from the graph conversion.
    """
    import onnxruntime as ort

    config = TrainingConfig()
    device = torch.device("cpu")
    dataset = build_dataset(PREPARED_DIR, "test", config)
    loader = build_dataloader(dataset, config, shuffle=False)
    images, targets = next(iter(loader))
    images = images[:num_images]

    with torch.no_grad():
        torch_probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # The graph is exported with a fixed batch of 1, which is exactly what the
    # API does per request. Parity is therefore checked one image at a time so
    # that the test exercises the real production path rather than a shape the
    # server never uses.
    onnx_probs = np.empty_like(torch_probs)
    for row in range(len(images)):
        single = images[row : row + 1].numpy()
        onnx_probs[row] = _softmax(session.run(None, {input_name: single})[0])[0]

    predicted_torch = torch_probs.argmax(axis=1)
    predicted_onnx = onnx_probs.argmax(axis=1)
    class_agreement = float((predicted_torch == predicted_onnx).mean())
    max_abs_diff = float(np.abs(torch_probs - onnx_probs).max())
    mean_abs_diff = float(np.abs(torch_probs - onnx_probs).mean())

    result = {
        "num_images": int(len(images)),
        "class_agreement": round(class_agreement, 4),
        "max_abs_prob_diff": round(max_abs_diff, 6),
        "mean_abs_prob_diff": round(mean_abs_diff, 6),
        "torch_mean_top1": round(float(torch_probs.max(axis=1).mean()), 4),
        "onnx_mean_top1": round(float(onnx_probs.max(axis=1).mean()), 4),
        "labels": targets[:num_images].tolist(),
    }
    LOGGER.info("Parity over %d images:", result["num_images"])
    LOGGER.info("  class agreement      : %.4f", result["class_agreement"])
    LOGGER.info("  max |prob difference|: %.6f", result["max_abs_prob_diff"])
    LOGGER.info("  mean |prob diff|     : %.6f", result["mean_abs_prob_diff"])
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def run(
    checkpoint: Path | None = None, opset: int = DEFAULT_OPSET, verify_only: bool = False
) -> dict:
    ensure_directories()
    config = TrainingConfig()
    checkpoint_path = checkpoint or (CHECKPOINTS_DIR / config.best_checkpoint_name)
    onnx_path = BACKEND_MODELS_DIR / ONNX_FILENAME

    if verify_only:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        model, payload = load_checkpoint(checkpoint_path, torch.device("cpu"))
        result = verify_parity(model, onnx_path)
        if result["class_agreement"] < 1.0 or result["max_abs_prob_diff"] > 1e-3:
            LOGGER.error("PARITY CHECK FAILED")
            raise RuntimeError(f"ONNX parity check failed: {result}")
        LOGGER.info("PARITY CHECK PASSED")
        return result

    model, payload = load_checkpoint(checkpoint_path, torch.device("cpu"))
    payload["_checkpoint_name"] = checkpoint_path.name

    export_onnx(model, onnx_path, config.image_size, opset)
    write_class_names(BACKEND_MODELS_DIR / CLASS_NAMES_FILENAME, config.image_size, payload)

    # Smoke-test the artefact through onnxruntime and compare to PyTorch.
    result = verify_parity(model, onnx_path)
    if result["class_agreement"] < 1.0 or result["max_abs_prob_diff"] > 1e-3:
        LOGGER.error("PARITY CHECK FAILED - export is not trustworthy")
        raise RuntimeError(f"ONNX parity check failed: {result}")
    LOGGER.info("PARITY CHECK PASSED")

    LOGGER.info("-" * 60)
    LOGGER.info("Exported for backend use:")
    LOGGER.info("  %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    LOGGER.info("  %s", BACKEND_MODELS_DIR / CLASS_NAMES_FILENAME)
    LOGGER.info("  PyTorch checkpoint kept at %s", checkpoint_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="only re-run the PyTorch vs ONNX parity check",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        run(checkpoint=args.checkpoint, opset=args.opset, verify_only=args.verify_only)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("Export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
