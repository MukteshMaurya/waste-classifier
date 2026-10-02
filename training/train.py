"""Fine-tune a lightweight CNN on the prepared waste dataset.

This script is the *only* place training happens. The deployed API never runs
it - it loads the ONNX file that ``export_model.py`` produces.

Usage::

    python training/train.py                      # default 8 epochs
    python training/train.py --epochs 3           # quick smoke run
    python training/train.py --arch efficientnet_b0

Design notes
------------
* Transfer learning: ImageNet backbone + fresh classification head.
* Class-balanced cross-entropy, because E-Waste has ~10x fewer images.
* 1 warm-up epoch then cosine decay.
* Early stopping on validation macro-F1 (not raw accuracy) so that a rare
  class cannot be traded away for a higher overall score.
* Best checkpoint is written to ``artifacts/checkpoints/best_model.pth`` and
  every epoch is appended to ``artifacts/metrics/training_history.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

if __package__ in (None, ""):  # allow `python training/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (  # noqa: E402
    CANONICAL_CLASSES,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    PREPARED_DIR,
    TrainingConfig,
    ensure_directories,
)
from training.dataset import (  # noqa: E402
    build_dataloader,
    build_dataset,
    compute_class_weights,
    seed_everything,
)
from training.models import build_model, count_parameters  # noqa: E402

LOGGER = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Metrics (kept local so evaluate.py owns the rich reporting)
# ---------------------------------------------------------------------------


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    """Macro-averaged F1 computed from a ``(N, N)`` confusion matrix."""
    true_positive = confusion.diag()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()

    denominator = predicted + actual
    f1 = torch.where(
        denominator > 0,
        2 * true_positive.float() / denominator.clamp(min=1),
        torch.zeros_like(true_positive, dtype=torch.float32),
    )
    ignored = (denominator == 0).sum().item()
    return f1.sum().item() / max(len(f1) - ignored, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, float]:
    """Run the model over a loader and return accuracy + macro-F1."""
    model.eval()
    num_classes = len(CANONICAL_CLASSES)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        total_items += targets.size(0)

        predictions = logits.argmax(dim=1)
        for actual, predicted in zip(targets.cpu(), predictions.cpu()):
            confusion[actual, predicted] += 1

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": confusion.diag().sum().item() / max(total_items, 1),
        "macro_f1": macro_f1_from_confusion(confusion),
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """AdamW with a lower learning rate for the pretrained backbone.

    The randomly initialised head needs a bigger step than the already-trained
    features, otherwise the head is still random when the backbone stops moving.
    """
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if "classifier" in name or name.startswith("fc.") else backbone_params).append(param)

    return AdamW(
        [
            {"params": backbone_params, "lr": config.learning_rate * 0.3},
            {"params": head_params, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    """Linear warm-up for ``warmup_epochs`` then cosine decay to ~1% of peak."""
    warmup_steps = max(1, config.warmup_epochs)
    total_steps = max(1, config.epochs)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps
        progress = (epoch - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: TrainingConfig) -> Path:
    seed_everything(config.seed)
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s | threads: %d", device, torch.get_num_threads())
    LOGGER.info("Config: %s", asdict(config))

    train_dataset = build_dataset(PREPARED_DIR, "train", config)
    val_dataset = build_dataset(PREPARED_DIR, "val", config)

    missing = [
        name
        for name in CANONICAL_CLASSES
        if name not in {CANONICAL_CLASSES[i] for i in train_dataset.label_counts}
    ]
    if missing:
        raise RuntimeError(f"Training split has no images for: {missing}")

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    model = build_model(config.architecture, config.num_classes, pretrained=True)
    model.to(device)
    LOGGER.info(
        "Model: %s | trainable parameters: %d",
        config.architecture,
        count_parameters(model),
    )

    if config.use_class_weights:
        weights = compute_class_weights(train_dataset, config.num_classes).to(device)
        LOGGER.info("Class weights: %s", [round(w, 3) for w in weights.tolist()])
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_started = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
            if step % 20 == 0 or step == len(train_loader):
                LOGGER.info(
                    "  epoch %d/%d step %d/%d loss %.4f",
                    epoch, config.epochs, step, len(train_loader),
                    running_loss / max(seen, 1),
                )

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(float(val_metrics["loss"]), 5),
            "val_accuracy": round(float(val_metrics["accuracy"]), 5),
            "val_macro_f1": round(float(val_metrics["macro_f1"]), 5),
            "lr": round(optimizer.param_groups[-1]["lr"], 6),
            "epoch_seconds": round(time.time() - epoch_started, 1),
        }
        history.append(record)
        LOGGER.info(
            "epoch %d/%d | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_macroF1 %.4f | %.1fs",
            epoch, config.epochs, train_loss, record["val_loss"],
            record["val_accuracy"], record["val_macro_f1"], record["epoch_seconds"],
        )

        checkpoint_path = CHECKPOINTS_DIR / config.last_checkpoint_name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "architecture": config.architecture,
                "num_classes": config.num_classes,
                "class_names": CANONICAL_CLASSES,
                "val_macro_f1": record["val_macro_f1"],
                "config": asdict(config),
            },
            checkpoint_path,
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "architecture": config.architecture,
                    "num_classes": config.num_classes,
                    "class_names": CANONICAL_CLASSES,
                    "val_macro_f1": best_macro_f1,
                    "config": asdict(config),
                },
                CHECKPOINTS_DIR / config.best_checkpoint_name,
            )
            LOGGER.info("  new best macro-F1 %.4f -> saved", best_macro_f1)
        else:
            epochs_without_improvement += 1
            LOGGER.info(
                "  no improvement (%d/%d)", epochs_without_improvement,
                config.early_stopping_patience,
            )
            if epochs_without_improvement >= config.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    summary = {
        "architecture": config.architecture,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_macro_f1, 5),
        "total_minutes": round((time.time() - started) / 60, 2),
        "parameters": count_parameters(model),
        "config": asdict(config),
        "history": history,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "training_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Done in %.1f min. best val macro-F1 %.4f (epoch %d). History -> %s",
        summary["total_minutes"], best_macro_f1, best_epoch,
        METRICS_DIR / "training_history.json",
    )
    return CHECKPOINTS_DIR / config.best_checkpoint_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--arch", default=TrainingConfig.architecture)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument(
        "--force-prepare", action="store_true",
        help="rebuild data/prepared before training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = TrainingConfig(
        epochs=args.epochs,
        architecture=args.arch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.force_prepare or not PREPARED_DIR.is_dir():
        from training.prepare_data import prepare

        LOGGER.info("Preparing dataset...")
        prepare(config, force=args.force_prepare)

    try:
        train(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
