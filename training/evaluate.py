"""Evaluate the trained waste classifier on the held-out test split.

Produces the artefacts that the README cites:

* ``metrics.json``          - overall + per-class precision/recall/F1/support
* ``classification_report.txt`` - scikit-learn report, human readable
* ``confusion_matrix.csv``  - raw counts
* ``confusion_matrix.png``  - normalised heatmap
* ``per_class_f1.png``      - per-class F1 bar chart

It also runs an explicit E-Waste diagnostic. E-Waste has 70 training images
against ~800 for every other class, so its loss weight is roughly 10x higher.
That trade-off is reported verbatim rather than smoothed over.

Usage::

    python training/evaluate.py
    python training/evaluate.py --split val
    python training/evaluate.py --checkpoint artifacts/checkpoints/last_model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # allow `python training/evaluate.py`
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
from training.dataset import build_dataloader, build_dataset  # noqa: E402
from training.models import build_model  # noqa: E402

LOGGER = logging.getLogger("evaluate")

#: The class singled out for a dedicated diagnostic.
FOCUS_CLASS = "E-Waste"


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from a checkpoint saved by ``train.py``."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload.get("architecture", TrainingConfig.architecture)
    num_classes = payload.get("num_classes", len(CANONICAL_CLASSES))
    saved_classes = payload.get("class_names")

    if saved_classes and list(saved_classes) != list(CANONICAL_CLASSES):
        raise RuntimeError(
            "Checkpoint class order does not match training/config.CANONICAL_CLASSES.\n"
            f"  checkpoint: {list(saved_classes)}\n"
            f"  config:     {list(CANONICAL_CLASSES)}"
        )

    model = build_model(architecture, num_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info(
        "Loaded %s (epoch %s, val macro-F1 %s)",
        path.name, payload.get("epoch"), payload.get("val_macro_f1"),
    )
    return model, payload


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader.

    Returns ``(probs[N, C], predictions[N], targets[N])``.
    """
    all_probs, all_preds, all_targets = [], [], []
    model.eval()
    for step, (images, targets) in enumerate(loader, start=1):
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())
        if step % 20 == 0 or step == len(loader):
            LOGGER.info("  inference step %d/%d", step, len(loader))
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def focus_class_diagnostics(
    confusion: np.ndarray, class_names: list[str], focus: str = FOCUS_CLASS
) -> dict:
    """Explain exactly how the focus class behaves, in plain numbers.

    * ``precision``  - of everything predicted as ``focus``, how much is real.
    * ``recall``     - of real ``focus`` items, how many were found.
    * ``over_predicted`` - predicted count far exceeds the true count, which is
      the classic symptom of an over-weighted minority class.
    * ``false_positive_sources`` - which classes are being mislabelled as focus.
    * ``false_negative_targets`` - which classes the focus class is mistaken for.
    """
    if focus not in class_names:
        raise ValueError(f"{focus!r} is not in {class_names}")

    index = class_names.index(focus)
    tp = int(confusion[index, index])
    support = int(confusion[index].sum())
    predicted_count = int(confusion[:, index].sum())
    fp = predicted_count - tp
    fn = support - tp
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    fp_sources = [
        {"class": class_names[i], "count": int(confusion[i, index])}
        for i in range(len(class_names))
        if i != index and confusion[i, index] > 0
    ]
    fp_sources.sort(key=lambda row: row["count"], reverse=True)

    fn_targets = [
        {"class": class_names[i], "count": int(confusion[index, i])}
        for i in range(len(class_names))
        if i != index and confusion[index, i] > 0
    ]
    fn_targets.sort(key=lambda row: row["count"], reverse=True)

    # "Over-predicted" = it claims many more items than actually exist.
    ratio = predicted_count / support if support else float("inf")
    return {
        "class": focus,
        "true_positives": tp,
        "support": support,
        "predicted_count": predicted_count,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_to_true_ratio": round(ratio, 3),
        "over_predicted": bool(ratio > 1.5),
        "under_predicted": bool(0 < ratio < 0.67),
        "poor_recall": bool(recall < 0.5),
        "poor_precision": bool(precision < 0.5),
        "top_false_positive_sources": fp_sources[:5],
        "top_false_negative_targets": fn_targets[:5],
    }


def build_plots(
    confusion: np.ndarray,
    class_names: list[str],
    per_class: list[dict],
    output_dir: Path,
) -> None:
    """Render the confusion heatmap and per-class F1 chart as PNGs."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not installed - skipping plots")
        return

    matrix = confusion.astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0
    )

    size = max(9, len(class_names) * 0.85)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(normalised, cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised, test split)")
    threshold = 0.5 * normalised.max() if normalised.max() > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{normalised[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if normalised[i, j] > threshold else "#1a1a1a",
            )
    fig.colorbar(image, ax=ax, label="Share of true class")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    names = [row["class"] for row in per_class]
    f1_scores = [row["f1"] for row in per_class]
    colours = ["#c0392b" if name == FOCUS_CLASS else "#2e7d32" for name in names]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.7), 4.6))
    bars = ax.bar(names, f1_scores, color=colours)
    for bar, value in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 on the test split (red = class with limited data)")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_f1.png", dpi=140)
    plt.close(fig)
    LOGGER.info("Wrote confusion_matrix.png and per_class_f1.png")


def evaluate(split: str = "test", checkpoint: Path | None = None) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()

    checkpoint_path = checkpoint or (CHECKPOINTS_DIR / config.best_checkpoint_name)
    model, payload = load_checkpoint(checkpoint_path, device)

    dataset = build_dataset(PREPARED_DIR, split, config)
    loader = build_dataloader(dataset, config, shuffle=False)

    LOGGER.info("Evaluating %d images from the %r split on %s", len(dataset), split, device)
    probs, predictions, targets = collect_predictions(model, loader, device)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(range(len(CANONICAL_CLASSES)))
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    confusion = confusion_matrix(targets, predictions, labels=labels)

    per_class = [
        {
            "class": CANONICAL_CLASSES[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in labels
    ]

    mean_confidence = float(probs.max(axis=1).mean())
    correct = predictions == targets
    report = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "architecture": payload.get("architecture"),
        "device": str(device),
        "num_test_images": int(len(targets)),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * support).sum() / support.sum()), 4),
        "mean_top1_confidence": round(mean_confidence, 4),
        "mean_confidence_when_correct": round(float(probs.max(axis=1)[correct].mean()), 4),
        "mean_confidence_when_wrong": round(
            float(probs.max(axis=1)[~correct].mean()) if (~correct).any() else 0.0, 4
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": CANONICAL_CLASSES,
        "focus_class_diagnostics": focus_class_diagnostics(confusion, CANONICAL_CLASSES),
    }

    # ---- text report -------------------------------------------------------
    report_text = classification_report(
        targets, predictions, labels=labels, target_names=CANONICAL_CLASSES,
        zero_division=0, digits=4,
    )
    header = (
        f"AI Waste Classifier - evaluation report\n"
        f"split={split}  checkpoint={checkpoint_path.name}  epoch={payload.get('epoch')}\n"
        f"images={len(targets)}  accuracy={accuracy:.4f}  macro_f1={f1.mean():.4f}\n"
        f"{'=' * 78}\n\n{report_text}\n"
        f"Confusion matrix (rows = true, cols = predicted):\n"
    )
    matrix_text = " " * 16 + "".join(f"{i:>6}" for i in labels) + "\n"
    for i, name in enumerate(CANONICAL_CLASSES):
        matrix_text += f"{i:>2} {name:<13}" + "".join(
            f"{confusion[i, j]:>6}" for j in labels
        ) + "\n"
    (METRICS_DIR / "classification_report.txt").write_text(
        header + matrix_text, encoding="utf-8"
    )
    (METRICS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(
        METRICS_DIR / "confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d",
        header=",".join(CANONICAL_CLASSES),
        comments="",
    )
    build_plots(confusion, CANONICAL_CLASSES, per_class, METRICS_DIR)

    # ---- console summary ---------------------------------------------------
    LOGGER.info("=" * 72)
    LOGGER.info("split=%s  images=%d  accuracy=%.4f  macro_f1=%.4f", split, len(targets), accuracy, f1.mean())
    LOGGER.info("-" * 72)
    LOGGER.info("%-16s %9s %9s %9s %8s", "class", "precision", "recall", "f1", "support")
    for row in per_class:
        LOGGER.info(
            "%-16s %9.4f %9.4f %9.4f %8d",
            row["class"], row["precision"], row["recall"], row["f1"], row["support"],
        )
    focus = report["focus_class_diagnostics"]
    LOGGER.info("-" * 72)
    LOGGER.info(
        "FOCUS %s: support=%d predicted=%d precision=%.3f recall=%.3f f1=%.3f ratio=%.2f",
        focus["class"], focus["support"], focus["predicted_count"],
        focus["precision"], focus["recall"], focus["f1"], focus["predicted_to_true_ratio"],
    )
    LOGGER.info("  over_predicted=%s poor_recall=%s poor_precision=%s",
                focus["over_predicted"], focus["poor_recall"], focus["poor_precision"])
    LOGGER.info("  top false-positive sources: %s", focus["top_false_positive_sources"])
    LOGGER.info("  top false-negative targets: %s", focus["top_false_negative_targets"])
    LOGGER.info("=" * 72)
    LOGGER.info("Artefacts written to %s", METRICS_DIR)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to artifacts/checkpoints/best_model.pth",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        evaluate(split=args.split, checkpoint=args.checkpoint)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
