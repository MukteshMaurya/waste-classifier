"""Preprocessing that mirrors training exactly, using only Pillow + NumPy.

The training pipeline used ``torchvision.transforms`` with
``Resize(255) -> CenterCrop(224) -> ToTensor -> Normalize``. This module
reproduces that pixel-for-pixel without importing torchvision, which keeps the
deployed container small (no PyTorch needed at runtime - only onnxruntime).

The exact equivalence was verified empirically: for a range of source sizes,
``transforms.Resize(255)`` produces byte-identical output to
``PIL.Image.resize((w, h), Image.BILINEAR)`` with the shorter side scaled to 255.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

#: torchvision's Resize(int) scales the *shorter* side and preserves aspect.
_RESIZE_SCALE = 1.14


@dataclass(frozen=True)
class PreprocessSpec:
    """Everything needed to turn a PIL image into a model input tensor.

    Loaded from ``class_names.json`` so training and serving cannot drift.
    """

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    resample_filter: str = "bilinear"

    @classmethod
    def from_mapping(cls, data: dict) -> "PreprocessSpec":
        normalization = data.get("normalization", {})
        return cls(
            image_size=int(data.get("image_size", 224)),
            mean=tuple(normalization.get("mean", cls.mean)),  # type: ignore[arg-type]
            std=tuple(normalization.get("std", cls.std)),  # type: ignore[arg-type]
            resample_filter=str(data.get("resample_filter", "bilinear")),
        )

    @property
    def resize_to(self) -> int:
        """Size the shorter side is scaled to before the centre crop."""
        return int(self.image_size * _RESIZE_SCALE)


_PIL_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
}


def target_size(width: int, height: int, shorter_side: int) -> tuple[int, int]:
    """Scale so the shorter side equals ``shorter_side``, keeping aspect ratio.

    Mirrors ``torchvision.transforms.Resize(int)``, which rounds both
    dimensions to the nearest pixel.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = shorter_side / min(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop the largest centred square, matching ``transforms.CenterCrop``."""
    width, height = image.size
    if width < size or height < size:
        raise ValueError(
            f"Image ({width}x{height}) is smaller than the crop size ({size})"
        )
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    """Convert a PIL image into a ``(1, C, H, W)`` float32 batch for onnxruntime.

    Steps, in order, matching training:
      1. convert to RGB
      2. resize shorter side to ``spec.resize_to`` with bilinear interpolation
      3. centre crop to ``spec.image_size``
      4. scale to [0, 1]
      5. normalise with the ImageNet mean/std
      6. transpose HWC -> CHW and add a batch dimension
    """
    rgb = image.convert("RGB")
    resized = rgb.resize(
        target_size(*rgb.size, spec.resize_to),
        _PIL_FILTERS.get(spec.resample_filter, Image.Resampling.BILINEAR),
    )
    cropped = center_crop(resized, spec.image_size)

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
        spec.std, dtype=np.float32
    )
    # HWC -> CHW -> NCHW
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def softmax(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)
