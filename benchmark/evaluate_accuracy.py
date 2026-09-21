"""
Foundry M3 — Accuracy Evaluation

Evaluates YOLO26n on the frozen COCO val2017 1,000-image
evaluation subset.

Primary metric:
    mAP50-95

Additional metrics:
    mAP50
    mAP75
    precision
    recall
    per-class mAP
    validation speed breakdown

The evaluation dataset is fixed by:
    benchmark/evaluation/coco1000/manifest.json
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo26n.pt"
)

DATA_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "coco1000"
    / "data.yaml"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "coco1000"
    / "manifest.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "accuracy_baseline.json"
)

IMAGE_SIZE = 640
BATCH_SIZE = 1
DEVICE = "cpu"
WORKERS = 0


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest() -> dict:
    """Load the frozen evaluation manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: "
            f"{MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_inputs() -> None:
    """Validate required evaluation artifacts."""
    required = [
        MODEL_PATH,
        DATA_PATH,
        MANIFEST_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )


def main() -> None:
    """Run the baseline accuracy evaluation."""

    validate_inputs()

    manifest = load_manifest()

    print(
        "Foundry M3 — YOLO26n Accuracy Evaluation"
    )
    print("=" * 55)

    print(
        f"Model:        {MODEL_PATH.name}"
    )

    print(
        f"Dataset:      COCO val2017"
    )

    print(
        f"Subset:       "
        f"{manifest['selection_size']} images"
    )

    print(
        f"Selection:    deterministic "
        f"(seed {manifest['selection_seed']})"
    )

    print(
        f"Image size:   {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size:   {BATCH_SIZE}"
    )

    print(
        f"Device:       {DEVICE}"
    )

    print(
        f"Workers:      {WORKERS}"
    )

    print(
        f"Torch:        {torch.__version__}"
    )

    print(
        f"Platform:     {platform.platform()}"
    )

    print(
        f"Manifest SHA: {sha256_file(MANIFEST_PATH)}"
    )

    print("\nRunning validation...")
    print("This may take some time on CPU.")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_PATH),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=False,
        verbose=True,
    )

    # --------------------------------------------------------------
    # Detection metrics
    # --------------------------------------------------------------

    box_metrics = metrics.box

    map50_95 = float(box_metrics.map)
    map50 = float(box_metrics.map50)
    map75 = float(box_metrics.map75)

    precision = float(
        getattr(
            box_metrics,
            "mp",
            0.0,
        )
    )

    recall = float(
        getattr(
            box_metrics,
            "mr",
            0.0,
        )
    )

    per_class_map = [
        float(value)
        for value in box_metrics.maps
    ]

    # --------------------------------------------------------------
    # Speed metrics
    # --------------------------------------------------------------

    speed = {
        key: float(value)
        for key, value in getattr(
            metrics,
            "speed",
            {},
        ).items()
    }

    # --------------------------------------------------------------
    # Build result
    # --------------------------------------------------------------

    result = {
        "benchmark": (
            "m3-yolo26n-coco1000-accuracy"
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": {
            "name": "YOLO26n",
            "path": str(
                MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "dataset": {
            "name": "COCO val2017",
            "subset_size": manifest[
                "selection_size"
            ],
            "manifest": str(
                MANIFEST_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "manifest_sha256": sha256_file(
                MANIFEST_PATH
            ),
            "selection_seed": manifest[
                "selection_seed"
            ],
            "classes": manifest[
                "classes_represented"
            ],
        },
        "configuration": {
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "device": DEVICE,
            "workers": WORKERS,
        },
        "metrics": {
            "map50_95": round(
                map50_95,
                6,
            ),
            "map50": round(
                map50,
                6,
            ),
            "map75": round(
                map75,
                6,
            ),
            "precision": round(
                precision,
                6,
            ),
            "recall": round(
                recall,
                6,
            ),
            "per_class_map50_95": [
                round(
                    value,
                    6,
                )
                for value in per_class_map
            ],
        },
        "speed_ms_per_image": {
            key: round(
                value,
                4,
            )
            for key, value in speed.items()
        },
        "notes": [
            "Evaluation uses the frozen COCO val2017 1,000-image subset.",
            "The subset was selected deterministically with seed 20260921.",
            "Ground-truth labels were generated directly from instances_val2017.json.",
            "COCO category IDs were mapped to contiguous YOLO class IDs.",
            "This is a fixed subset evaluation, not an official full COCO benchmark score.",
            "The same evaluation subset must be reused for FP32, FP16, and INT8 comparisons.",
        ],
    }

    # --------------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print("\n" + "=" * 55)
    print("ACCURACY RESULTS")
    print("=" * 55)

    print(
        f"mAP50-95:  {map50_95:.4f}"
    )

    print(
        f"mAP50:     {map50:.4f}"
    )

    print(
        f"mAP75:     {map75:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"\nPer-class mAP entries: "
        f"{len(per_class_map)}"
    )

    if speed:
        print("\nValidation speed:")
        for key, value in speed.items():
            print(
                f"  {key:12s}: "
                f"{value:.3f} ms/image"
            )

    print("\nResult saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()