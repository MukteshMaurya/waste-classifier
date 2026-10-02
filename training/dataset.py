"""Dataset, augmentation pipeline and class balancing for waste classification.

The class index of the model is always the index of the class name inside
``CANONICAL_CLASSES`` - never an alphabetical directory listing - so the
exported ``class_names.json`` lines up with the ONNX output layer no matter
how the folders happen to be sorted on disk.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:  # normal import when used as a package
    from training.config import (
        CANONICAL_CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PREPARED_DIR,
        TrainingConfig,
        class_dir_name,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from config import (  # type: ignore[no-redef]
        CANONICAL_CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PREPARED_DIR,
        TrainingConfig,
        class_dir_name,
    )

LOGGER = logging.getLogger("dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    """Return the preprocessing pipeline for train or eval.

    Training uses ``RandomResizedCrop`` with scale 0.65-1.0 so the model sees
    zoomed-in views (waste photos are usually one object on a cluttered
    background) plus flips and mild colour jitter. Evaluation is a plain
    resize + centre crop so validation metrics are deterministic.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.2,
                    hue=0.03,
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class WasteImageDataset(Dataset):
    """Image dataset backed by an explicit ``(path, label)`` list.

    Parameters
    ----------
    samples:
        Explicit sample list. Preferred, because it comes straight from the
        manifest produced by ``prepare_data.py``.
    transform:
        Torchvision transform applied to each decoded PIL image.
    corrupt_policy:
        ``"raise"`` during evaluation, ``"skip"`` during training.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: transforms.Compose,
        corrupt_policy: str = "raise",
    ) -> None:
        if not samples:
            raise ValueError("Dataset is empty - run training/prepare_data.py first")
        self.samples = samples
        self.transform = transform
        self.corrupt_policy = corrupt_policy
        self.class_names = list(CANONICAL_CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as handle:
                image = handle.convert("RGB")
        except (OSError, ValueError) as exc:
            if self.corrupt_policy == "raise":
                raise RuntimeError(f"Unreadable image: {sample.path}") from exc
            LOGGER.warning("Skipping unreadable image %s (%s)", sample.path, exc)
            return self[(index + 1) % len(self.samples)]

        return self.transform(image), sample.label

    @property
    def label_counts(self) -> Counter[int]:
        return Counter(sample.label for sample in self.samples)


def _scan_split(prepared_dir: Path, split: str) -> list[Sample]:
    """Fallback loader that walks the prepared tree directly."""
    split_dir = prepared_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    samples: list[Sample] = []
    for label, class_name in enumerate(CANONICAL_CLASSES):
        class_dir = split_dir / class_dir_name(class_name)
        if not class_dir.is_dir():
            continue
        for entry in sorted(class_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(entry, label))
    return samples


def load_samples(prepared_dir: Path, split: str) -> list[Sample]:
    """Load a split from ``manifest.json`` when available, else scan the tree."""
    manifest_path = prepared_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label_by_name = {name: i for i, name in enumerate(CANONICAL_CLASSES)}
        samples = [
            Sample(prepared_dir / record["target_path"], label_by_name[record["class_name"]])
            for record in manifest.get("records", [])
            if record["split"] == split
        ]
        if samples:
            return samples
        LOGGER.warning("Manifest contained no records for split %r - scanning", split)
    return _scan_split(prepared_dir, split)


def build_dataset(
    prepared_dir: Path, split: str, config: TrainingConfig
) -> WasteImageDataset:
    """Instantiate the dataset for ``split`` (``train``/``val``/``test``)."""
    samples = load_samples(prepared_dir, split)
    is_train = split == "train"
    dataset = WasteImageDataset(
        samples,
        transform=build_transform(config.image_size, train=is_train),
        # Never let a single unreadable file abort a long CPU training run.
        corrupt_policy="skip" if is_train else "raise",
    )
    LOGGER.info("%-5s split: %5d images", split, len(dataset))
    return dataset


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset: WasteImageDataset, config: TrainingConfig, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=False,  # CPU-only training
        drop_last=shuffle and len(dataset) > config.batch_size,
        persistent_workers=config.num_workers > 0,
    )


def compute_class_weights(
    dataset: WasteImageDataset, num_classes: int
) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1.0.

    Without this, cross-entropy optimises for total accuracy and effectively
    ignores E-Waste (100 images against ~1,200 for every other class).
    """
    counts = dataset.label_counts
    total = sum(counts.get(i, 0) for i in range(num_classes))
    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count == 0:
            # Never seen in training: keep the loss finite, flag it in logs.
            weights[class_index] = 1.0
            continue
        weights[class_index] = total / (num_classes * count)
    return weights / weights.mean()
