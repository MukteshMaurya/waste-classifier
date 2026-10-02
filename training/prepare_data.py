"""Build a deterministic, class-balanced training dataset from the raw downloads.

The raw downloads contain two sources with different folder naming and very
different sizes. This script normalises them into the canonical taxonomy once,
so that every later stage (training, evaluation, export) sees a plain
``ImageFolder``-compatible tree:

    data/prepared/
        train/Cardboard/*.jpg
        train/Paper/*.jpg
        ...
        val/...
        test/...

Run it with::

    python training/prepare_data.py
    python training/prepare_data.py --force     # rebuild from scratch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):  # allow `python training/prepare_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402  (import after sys.path bootstrap)
    CANONICAL_CLASSES,
    GARBAGE_DATASET_DIR,
    GARBAGE_SOURCE_MAP,
    MAX_IMAGES_PER_CLASS,
    PREPARED_DIR,
    WASTESNAP_DIR,
    WASTESNAP_SOURCE_MAP,
    TrainingConfig,
    class_dir_name,
    ensure_directories,
)

LOGGER = logging.getLogger("prepare_data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

#: Maps a split folder name to the fraction of that class that goes into it.
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitRecord:
    """One prepared training example, kept so the split is auditable."""

    source: str
    source_path: str
    target_path: str
    class_name: str
    split: str


def _iter_images(directory: Path) -> list[Path]:
    """Return image files in ``directory`` (non-recursive, sorted)."""
    if not directory.is_dir():
        return []
    return sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )


def _fingerprint(paths: list[Path]) -> str:
    """Cheap stable hash of a file list, used to detect configuration changes."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
    return digest.hexdigest()[:16]


def collect_sources(seed: int) -> dict[str, list[tuple[str, Path]]]:
    """Gather every usable image, grouped by canonical class name.

    Returns
    -------
    dict[str, list[tuple[str, Path]]]
        ``class_name -> [(source_name, path), ...]``
    """
    by_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    missing_sources: list[str] = []
    for folder, class_name in GARBAGE_SOURCE_MAP.items():
        source_dir = GARBAGE_DATASET_DIR / folder
        images = _iter_images(source_dir)
        if not images:
            missing_sources.append(f"{GARBAGE_DATASET_DIR.name}/{folder}")
            continue
        for image in images:
            by_class[class_name].append(("garbage-dataset", image))
        LOGGER.info("  %-14s -> %-14s %5d images", folder, class_name, len(images))

    for folder, class_name in WASTESNAP_SOURCE_MAP.items():
        source_dir = WASTESNAP_DIR / folder
        images = _iter_images(source_dir)
        if not images:
            missing_sources.append(f"{WASTESNAP_DIR.name}/{folder}")
            continue
        for image in images:
            by_class[class_name].append(("wastesnap", image))
        LOGGER.info("  %-14s -> %-14s %5d images", folder, class_name, len(images))

    if missing_sources:
        raise FileNotFoundError(
            "Expected dataset folders are missing: "
            + ", ".join(missing_sources)
            + ". See training/README or the root README for the download commands."
        )

    missing_classes = set(CANONICAL_CLASSES) - set(by_class)
    if missing_classes:
        raise RuntimeError(
            "No images found for canonical class(es): "
            + ", ".join(sorted(missing_classes))
        )
    return by_class


def stratified_split(
    items: list[tuple[str, Path]], config: TrainingConfig, rng: random.Random
) -> dict[str, list[tuple[str, Path]]]:
    """Split one class into train/val/test, guaranteeing every split is non-empty.

    Small classes (E-Waste has 100 images) are split with *at least* one image
    per split so that validation and evaluation never crash on an empty loader.
    """
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        raise ValueError(f"Class has only {n} image(s); at least 3 are required")

    n_test = max(1, round(n * config.test_split))
    n_val = max(1, round(n * config.val_split))
    if n_test + n_val > n - 1:  # keep at least one training example
        n_val = max(1, n - n_test - 1)

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def prepare(config: TrainingConfig, force: bool = False) -> Path:
    """Create the prepared dataset tree and return its path."""
    ensure_directories()
    manifest_path = PREPARED_DIR / "manifest.json"

    by_class = collect_sources(config.seed)

    if not force and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        # Rebuild when the class list, caps or splits changed.
        if (
            previous.get("canonical_classes") == CANONICAL_CLASSES
            and previous.get("config") == asdict(config)
            and previous.get("version") == MANIFEST_VERSION
        ):
            LOGGER.info(
                "Reusing prepared dataset at %s (pass --force to rebuild)", PREPARED_DIR
            )
            return PREPARED_DIR
        LOGGER.info("Configuration changed - rebuilding prepared dataset")

    if PREPARED_DIR.exists():
        LOGGER.info("Removing previous prepared dataset at %s", PREPARED_DIR)
        shutil.rmtree(PREPARED_DIR)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    records: list[SplitRecord] = []
    summary: dict[str, dict[str, int]] = {}

    for class_name in CANONICAL_CLASSES:
        items = by_class[class_name]
        cap = MAX_IMAGES_PER_CLASS.get(class_name, len(items))
        if len(items) > cap:
            LOGGER.info(
                "  capping %-14s from %5d to %d images", class_name, len(items), cap
            )
            rng.shuffle(items)
            items = items[:cap]

        splits = stratified_split(items, config, rng)
        counts: dict[str, int] = {}
        for split_name, split_items in splits.items():
            target_dir = PREPARED_DIR / split_name / class_dir_name(class_name)
            target_dir.mkdir(parents=True, exist_ok=True)
            counts[split_name] = len(split_items)
            for source_name, source_path in split_items:
                # Keep the original extension so Pillow picks the right decoder.
                target_path = target_dir / f"{source_path.stem}{source_path.suffix.lower()}"
                shutil.copy2(source_path, target_path)
                records.append(
                    SplitRecord(
                        source=source_name,
                        source_path=str(source_path),
                        target_path=str(target_path.relative_to(PREPARED_DIR)),
                        class_name=class_name,
                        split=split_name,
                    )
                )
        summary[class_name] = counts
        LOGGER.info(
            "  %-14s train=%4d val=%3d test=%3d", class_name,
            counts["train"], counts["val"], counts["test"],
        )

    totals = {
        split: sum(summary[c][split] for c in CANONICAL_CLASSES) for split in SPLIT_NAMES
    }
    LOGGER.info("Totals: %s", totals)

    manifest = {
        "version": MANIFEST_VERSION,
        "canonical_classes": CANONICAL_CLASSES,
        "sources": {
            "garbage-dataset": {
                "path": str(GARBAGE_DATASET_DIR),
                "license": "MIT",
                "url": "https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset",
            },
            "wastesnap": {
                "path": str(WASTESNAP_DIR),
                "license": "CC-BY-2.5",
                "url": "https://huggingface.co/datasets/Kishore2412/Multi-Class-Waste-Image-Classification-Dataset",
            },
        },
        "config": asdict(config),
        "splits": {split: totals[split] for split in SPLIT_NAMES},
        "split_ratios": {
            "train": config.train_split,
            "val": config.val_split,
            "test": config.test_split,
        },
        "per_class_counts": summary,
        "records": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)
    return PREPARED_DIR


MANIFEST_VERSION = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the manifest matches"
    )
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = TrainingConfig(seed=args.seed)
    try:
        prepare(config, force=args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("Data preparation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
