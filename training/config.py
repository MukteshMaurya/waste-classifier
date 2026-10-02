"""Central configuration for the AI Waste Classifier training pipeline.

All paths are resolved relative to the repository root so that every script
(``train.py``, ``evaluate.py``, ``export_model.py``) behaves identically no
matter which working directory it is launched from.

Nothing in this module trains anything; it only declares constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Raw, untouched dataset downloads. Never committed to git (see .gitignore).
RAW_DATA_ROOT: Path = Path(
    os.getenv("WASTE_RAW_DATA_ROOT", Path(os.getenv("TEMP", "/tmp")) / "opencode")
)

#: Primary dataset - "Garbage Classification Dataset" by Suman Kunwar (D.Waste.app)
#: https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset
#: License: MIT. 19,762 images across 10 folders.
GARBAGE_DATASET_DIR: Path = RAW_DATA_ROOT / "garbage_dataset"

#: Secondary dataset used only to fill the "E-Waste" class, which the primary
#: dataset does not contain.  https://huggingface.co/datasets/Kishore2412/
#: Multi-Class-Waste-Image-Classification-Dataset  License: CC-BY-2.5.
WASTESNAP_DIR: Path = RAW_DATA_ROOT / "wastesnap" / "extracted" / "WasteSnap_Dataset"

#: Deterministic, class-per-folder dataset produced by ``prepare_data.py``.
PREPARED_DIR: Path = REPO_ROOT / "data" / "prepared"

#: Training outputs.
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR: Path = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"

#: Deployment artefacts consumed by the FastAPI backend.
BACKEND_MODELS_DIR: Path = REPO_ROOT / "backend" / "models"

# ---------------------------------------------------------------------------
# Waste taxonomy
# ---------------------------------------------------------------------------

#: The canonical product taxonomy. The order defines the model's output index,
#: so it is written explicitly rather than derived from a directory listing.
CANONICAL_CLASSES: list[str] = [
    "Cardboard",
    "Paper",
    "Plastic",
    "Glass",
    "Metal",
    "Organic",
    "Textile",
    "E-Waste",
    "Battery",
    "Other/Unknown",
]

#: Maps a folder name in the primary dataset to a canonical class.
#: ``clothes`` and ``shoes`` are both folded into ``Textile`` on purpose:
#: footwear is textile waste, and merging gives the class enough data to learn.
GARBAGE_SOURCE_MAP: dict[str, str] = {
    "cardboard": "Cardboard",
    "paper": "Paper",
    "plastic": "Plastic",
    "glass": "Glass",
    "metal": "Metal",
    "biological": "Organic",
    "clothes": "Textile",
    "shoes": "Textile",
    "battery": "Battery",
    "trash": "Other/Unknown",
}

#: Maps a folder name in the WasteSnap supplement to a canonical class.
WASTESNAP_SOURCE_MAP: dict[str, str] = {
    "Electronic waste": "E-Waste",
}


def class_dir_name(class_name: str) -> str:
    """Return the on-disk folder name for a canonical class.

    ``Other/Unknown`` is a fine *display* name but contains a path separator,
    which would silently create a nested ``Other/Unknown`` directory. On disk it
    becomes ``Other-Unknown`` while the model class name is unchanged.
    """
    return class_name.replace("/", "-")

#: Per-class cap. The primary dataset is very unbalanced (Textile has 7,304
#: images, Metal has 1,020). Capping the large classes keeps CPU training time
#: reasonable and improves macro-F1. E-Waste is *not* capped: it has only 100
#: source images, so every single one is used.
MAX_IMAGES_PER_CLASS: dict[str, int] = {
    "Cardboard": 1_200,
    "Paper": 1_200,
    "Plastic": 1_200,
    "Glass": 1_200,
    "Metal": 1_200,
    "Organic": 1_200,
    "Textile": 1_200,
    "E-Waste": 10_000,
    "Battery": 1_200,
    "Other/Unknown": 1_200,
}

# ---------------------------------------------------------------------------
# Model / training hyper-parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Everything needed to reproduce a training run."""

    # Backbone. MobileNetV3-Small is 2.5M parameters / ~10 MB ONNX and runs
    # at ~13 ms per image on a 4-thread CPU, which is what Render gives us.
    # ``efficientnet_b0`` and ``resnet18`` are supported drop-in alternatives.
    architecture: str = "mobilenet_v3_small"
    image_size: int = 224
    num_classes: int = len(CANONICAL_CLASSES)

    # Optimisation
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    early_stopping_patience: int = 3
    #: Linear warm-up + cosine decay; helps the small dataset converge faster.
    warmup_epochs: int = 1
    #: Use class-balanced loss so the 100-image E-Waste class is not ignored.
    use_class_weights: bool = True

    # Data
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 4
    #: Reuse ``data/prepared`` if it is newer than the raw data. Set to False
    #: to force a rebuild after editing a mapping in this file.
    reuse_prepared: bool = True

    # Mixed precision is pointless on CPU; the flag is kept for GPU runs.
    use_amp: bool = False

    best_checkpoint_name: str = "best_model.pth"
    last_checkpoint_name: str = "last_model.pth"

    def __post_init__(self) -> None:
        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if not 0.0 < self.test_split < 1.0:
            raise ValueError("test_split must be between 0 and 1")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must leave room for training data")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture {self.architecture!r}. "
                f"Choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
            )

    @property
    def train_split(self) -> float:
        return round(1.0 - self.val_split - self.test_split, 6)

    @property
    def run_name(self) -> str:
        return f"{self.architecture}_seed{self.seed}"


SUPPORTED_ARCHITECTURES: dict[str, str] = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
}

#: ImageNet normalisation statistics (the pretrained backbones expect these).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Name of the exported ONNX file the backend loads at start-up.
ONNX_FILENAME: str = "waste_classifier.onnx"
CLASS_NAMES_FILENAME: str = "class_names.json"


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in (
        PREPARED_DIR,
        CHECKPOINTS_DIR,
        METRICS_DIR,
        BACKEND_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
