"""Model factory shared by training, evaluation and export.

Keeping the architecture definition in one place guarantees the ONNX file that
the backend loads is byte-for-byte derived from the same network that was
trained and evaluated.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models

try:  # normal import when used as a package
    from training.config import SUPPORTED_ARCHITECTURES
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from config import SUPPORTED_ARCHITECTURES  # type: ignore[no-redef]

#: Maps our architecture keys to the torchvision constructor + default weights.
_BACKBONES: dict[str, tuple[str, Any]] = {
    "mobilenet_v3_small": ("mobilenet_v3_small", models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": ("mobilenet_v3_large", models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "efficientnet_b0": ("efficientnet_b0", models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": ("resnet18", models.ResNet18_Weights.IMAGENET1K_V1),
}


def build_model(
    architecture: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """Create a backbone with a fresh ``num_classes`` head.

    ``pretrained=False`` is only used by the unit tests, which need a randomly
    initialised network that does not hit the network for ImageNet weights.
    """
    if architecture not in _BACKBONES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    constructor_name, weights = _BACKBONES[architecture]
    weights = weights if pretrained else None
    model: nn.Module = getattr(models, constructor_name)(weights=weights)

    if architecture.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # resnet18
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
