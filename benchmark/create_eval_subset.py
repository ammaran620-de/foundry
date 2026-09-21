"""
Foundry M3 — Create Deterministic Evaluation Subset

Creates a fixed 1,000-image evaluation subset from COCO val2017.

Source of truth:
    datasets/coco/annotations/instances_val2017.json

Selection considers:
- class coverage
- object-count distribution
- object-scale distribution
- negative images

The selection is deterministic and produces:
- evaluation images
- YOLO detection-format bounding-box labels
- data.yaml
- manifest.json

The resulting subset is frozen and reused for all M3
accuracy comparisons.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "coco"
    / "images"
    / "val2017"
)

ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "coco"
    / "annotations"
    / "instances_val2017.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "coco1000"
)

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"

MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
DATA_YAML_PATH = OUTPUT_DIR / "data.yaml"

TARGET_COUNT = 1000
SEED = 20260921


def deterministic_key(value: str) -> int:
    """Create a stable deterministic ordering key."""
    digest = hashlib.sha256(
        f"{SEED}:{value}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def load_coco() -> dict:
    """Load the official COCO validation annotations."""
    with ANNOTATIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_category_mapping(
    coco: dict,
) -> tuple[dict[int, int], dict[int, str]]:
    """
    Build COCO-category-ID -> contiguous YOLO-ID mapping.

    COCO category IDs contain gaps, so they must not be used
    directly as YOLO class indices.
    """
    categories = sorted(
        coco["categories"],
        key=lambda category: category["id"],
    )

    category_to_yolo = {
        int(category["id"]): index
        for index, category in enumerate(categories)
    }

    yolo_names = {
        index: category["name"]
        for index, category in enumerate(categories)
    }

    return category_to_yolo, yolo_names


def classify_scale(area: float) -> str:
    """
    Classify object scale using COCO area thresholds.

    small  : area < 32^2
    medium : 32^2 <= area <= 96^2
    large  : area > 96^2
    """
    if area < 32 * 32:
        return "small"

    if area <= 96 * 96:
        return "medium"

    return "large"


def build_records(coco: dict) -> list[dict]:
    """Build per-image evaluation statistics."""
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in coco["annotations"]:
        # Ignore crowd annotations for detection evaluation.
        if annotation.get("iscrowd", 0):
            continue

        annotations_by_image[
            annotation["image_id"]
        ].append(annotation)

    records: list[dict] = []

    for image in coco["images"]:
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])

        annotations = annotations_by_image.get(
            image_id,
            [],
        )

        classes = sorted(
            {
                int(annotation["category_id"])
                for annotation in annotations
            }
        )

        scales = sorted(
            {
                classify_scale(
                    max(
                        float(annotation["area"]),
                        0.0,
                    )
                )
                for annotation in annotations
            }
        )

        object_count = len(annotations)

        if object_count == 0:
            object_bin = "negative"
        elif object_count == 1:
            object_bin = "1"
        elif object_count <= 3:
            object_bin = "2-3"
        elif object_count <= 10:
            object_bin = "4-10"
        else:
            object_bin = "11+"

        image_path = IMAGE_DIR / image["file_name"]

        records.append(
            {
                "image_id": image_id,
                "filename": image["file_name"],
                "width": width,
                "height": height,
                "image_path": image_path,
                "annotations": annotations,
                "classes": classes,
                "scales": scales,
                "object_count": object_count,
                "object_bin": object_bin,
                "sort_key": deterministic_key(
                    image["file_name"]
                ),
            }
        )

    return records


def select_subset(
    records: list[dict],
) -> list[dict]:
    """
    Select TARGET_COUNT images deterministically.

    Selection priorities:
    1. Include all negative images.
    2. Cover rare classes.
    3. Cover object-count bins.
    4. Cover object-scale categories.
    5. Fill remaining slots deterministically.
    """
    if len(records) < TARGET_COUNT:
        raise RuntimeError(
            f"Only {len(records)} COCO images are available; "
            f"{TARGET_COUNT} are required."
        )

    records = sorted(
        records,
        key=lambda record: record["sort_key"],
    )

    selected: list[dict] = []
    selected_ids: set[int] = set()

    # --------------------------------------------------------------
    # Phase 1 — preserve all available negative images
    # --------------------------------------------------------------

    negatives = [
        record
        for record in records
        if record["object_count"] == 0
    ]

    for record in negatives:
        if len(selected) >= TARGET_COUNT:
            break

        selected.append(record)
        selected_ids.add(record["image_id"])

    # --------------------------------------------------------------
    # Remaining candidates
    # --------------------------------------------------------------

    remaining = [
        record
        for record in records
        if record["image_id"] not in selected_ids
    ]

    # --------------------------------------------------------------
    # Class frequencies
    # --------------------------------------------------------------

    class_frequency: Counter[int] = Counter()

    for record in remaining:
        for class_id in record["classes"]:
            class_frequency[class_id] += 1

    class_coverage: Counter[int] = Counter()

    # --------------------------------------------------------------
    # Phase 2 — rare-class coverage
    # --------------------------------------------------------------

    while len(selected) < TARGET_COUNT:
        candidates = [
            record
            for record in remaining
            if record["image_id"] not in selected_ids
        ]

        if not candidates:
            break

        def class_score(record: dict) -> tuple:
            unseen_classes = sum(
                1
                for class_id in record["classes"]
                if class_coverage[class_id] == 0
            )

            rarity = sum(
                1.0
                / max(
                    class_frequency[class_id],
                    1,
                )
                for class_id in record["classes"]
            )

            return (
                unseen_classes,
                rarity,
                -record["sort_key"],
            )

        best = max(
            candidates,
            key=class_score,
        )

        introduces_new_class = any(
            class_coverage[class_id] == 0
            for class_id in best["classes"]
        )

        if not introduces_new_class:
            break

        selected.append(best)
        selected_ids.add(best["image_id"])

        for class_id in best["classes"]:
            class_coverage[class_id] += 1

    # --------------------------------------------------------------
    # Phase 3 — object-count and scale coverage
    # --------------------------------------------------------------

    object_bin_coverage: Counter[str] = Counter(
        record["object_bin"]
        for record in selected
    )

    scale_coverage: Counter[str] = Counter(
        scale
        for record in selected
        for scale in record["scales"]
    )

    desired_object_bins = {
        "1",
        "2-3",
        "4-10",
        "11+",
    }

    desired_scales = {
        "small",
        "medium",
        "large",
    }

    while len(selected) < TARGET_COUNT:
        candidates = [
            record
            for record in remaining
            if record["image_id"] not in selected_ids
        ]

        if not candidates:
            break

        def coverage_score(record: dict) -> tuple:
            object_gain = (
                1
                if (
                    record["object_bin"]
                    in desired_object_bins
                    and object_bin_coverage[
                        record["object_bin"]
                    ]
                    == 0
                )
                else 0
            )

            scale_gain = sum(
                1
                for scale in record["scales"]
                if (
                    scale in desired_scales
                    and scale_coverage[scale] == 0
                )
            )

            return (
                object_gain,
                scale_gain,
                -record["sort_key"],
            )

        best = max(
            candidates,
            key=coverage_score,
        )

        selected.append(best)
        selected_ids.add(best["image_id"])

        object_bin_coverage[
            best["object_bin"]
        ] += 1

        for scale in best["scales"]:
            scale_coverage[scale] += 1

        coverage_complete = (
            all(
                object_bin_coverage[bin_name] > 0
                for bin_name in desired_object_bins
            )
            and all(
                scale_coverage[scale] > 0
                for scale in desired_scales
            )
        )

        if coverage_complete:
            break

    # --------------------------------------------------------------
    # Phase 4 — deterministic fill
    # --------------------------------------------------------------

    final_candidates = [
        record
        for record in records
        if record["image_id"] not in selected_ids
    ]

    final_candidates.sort(
        key=lambda record: record["sort_key"]
    )

    required = TARGET_COUNT - len(selected)

    selected.extend(
        final_candidates[:required]
    )

    if len(selected) != TARGET_COUNT:
        raise RuntimeError(
            f"Selection produced {len(selected)} images; "
            f"expected {TARGET_COUNT}."
        )

    selected.sort(
        key=lambda record: record["sort_key"]
    )

    return selected


def coco_bbox_to_yolo(
    annotation: dict,
    image_width: int,
    image_height: int,
    category_to_yolo: dict[int, int],
) -> str:
    """Convert a COCO bounding box to YOLO detection format."""
    coco_category_id = int(
        annotation["category_id"]
    )

    if coco_category_id not in category_to_yolo:
        raise KeyError(
            "COCO category ID "
            f"{coco_category_id} is not present "
            "in the category mapping."
        )

    x, y, width, height = annotation["bbox"]

    x = max(float(x), 0.0)
    y = max(float(y), 0.0)
    width = max(float(width), 0.0)
    height = max(float(height), 0.0)

    # Clip the box to image boundaries.
    x = min(
        x,
        float(image_width),
    )

    y = min(
        y,
        float(image_height),
    )

    width = min(
        width,
        max(
            float(image_width) - x,
            0.0,
        ),
    )

    height = min(
        height,
        max(
            float(image_height) - y,
            0.0,
        ),
    )

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            "Invalid COCO bounding box after clipping: "
            f"{annotation['bbox']}"
        )

    x_center = x + width / 2.0
    y_center = y + height / 2.0

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    x_center = min(
        max(x_center, 0.0),
        1.0,
    )

    y_center = min(
        max(y_center, 0.0),
        1.0,
    )

    width = min(
        max(width, 0.0),
        1.0,
    )

    height = min(
        max(height, 0.0),
        1.0,
    )

    category_id = category_to_yolo[
        coco_category_id
    ]

    return (
        f"{category_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


def write_yaml(
    yolo_names: dict[int, str],
) -> None:
    """Write the YOLO evaluation dataset configuration."""

    # Use an explicit project-relative dataset root.
    # This prevents Ultralytics from resolving '.' against
    # the wrong working directory.
    lines = [
        "path: benchmark/evaluation/coco1000",
        "train: images",
        "val: images",
        "names:",
    ]

    for class_id in sorted(yolo_names):
        lines.append(
            f"  {class_id}: "
            f"{yolo_names[class_id]}"
        )

    with DATA_YAML_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines) + "\n"
        )


def copy_subset(
    selected: list[dict],
    category_to_yolo: dict[int, int],
) -> None:
    """Copy images and create YOLO detection labels."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for record in selected:
        source = record["image_path"]

        if not source.exists():
            raise FileNotFoundError(
                f"Image not found: {source}"
            )

        destination = (
            OUTPUT_IMAGE_DIR
            / record["filename"]
        )

        shutil.copy2(
            source,
            destination,
        )

        label_lines = []

        for annotation in record["annotations"]:
            label_lines.append(
                coco_bbox_to_yolo(
                    annotation=annotation,
                    image_width=record["width"],
                    image_height=record["height"],
                    category_to_yolo=category_to_yolo,
                )
            )

        label_path = (
            OUTPUT_LABEL_DIR
            / f"{Path(record['filename']).stem}.txt"
        )

        label_path.write_text(
            "\n".join(label_lines)
            + (
                "\n"
                if label_lines
                else ""
            ),
            encoding="utf-8",
        )


def build_manifest(
    selected: list[dict],
    category_to_yolo: dict[int, int],
    yolo_names: dict[int, str],
) -> dict:
    """Build the reproducibility manifest."""
    class_counts: Counter[int] = Counter()
    object_bins: Counter[str] = Counter()
    scales: Counter[str] = Counter()

    negative_count = 0

    for record in selected:
        if record["object_count"] == 0:
            negative_count += 1

        object_bins[
            record["object_bin"]
        ] += 1

        for coco_class_id in record["classes"]:
            yolo_class_id = category_to_yolo[
                coco_class_id
            ]

            class_counts[
                yolo_class_id
            ] += 1

        for scale in record["scales"]:
            scales[scale] += 1

    return {
        "dataset": "COCO val2017",
        "source_annotations": str(
            ANNOTATIONS_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "source_image_count": 5000,
        "source_annotated_image_count": 4952,
        "source_negative_image_count": 48,
        "selection_size": len(selected),
        "selection_seed": SEED,
        "selection_method": (
            "Deterministic selection from COCO "
            "val2017 using negative-image "
            "preservation, rare-class coverage, "
            "object-count coverage, object-scale "
            "coverage, and deterministic fill."
        ),
        "label_generation": (
            "COCO bounding boxes converted to YOLO "
            "detection format using the category "
            "mapping derived from "
            "instances_val2017.json."
        ),
        "category_mapping": {
            str(coco_id): yolo_id
            for coco_id, yolo_id in sorted(
                category_to_yolo.items()
            )
        },
        "class_names": {
            str(class_id): name
            for class_id, name in sorted(
                yolo_names.items()
            )
        },
        "negative_images_selected": (
            negative_count
        ),
        "classes_represented": len(
            class_counts
        ),
        "class_image_counts": {
            str(class_id): count
            for class_id, count in sorted(
                class_counts.items()
            )
        },
        "object_count_distribution": dict(
            object_bins
        ),
        "scale_coverage": dict(
            scales
        ),
        "images": [
            {
                "image_id": record["image_id"],
                "filename": record["filename"],
                "object_count": (
                    record["object_count"]
                ),
                "object_bin": (
                    record["object_bin"]
                ),
                "classes": [
                    category_to_yolo[
                        class_id
                    ]
                    for class_id in record[
                        "classes"
                    ]
                ],
                "scales": record["scales"],
            }
            for record in selected
        ],
    }


def print_summary(
    selected: list[dict],
    category_to_yolo: dict[int, int],
) -> None:
    """Print evaluation subset summary."""
    class_counts: Counter[int] = Counter()
    object_bins: Counter[str] = Counter()
    scales: Counter[str] = Counter()

    negative_count = 0

    for record in selected:
        if record["object_count"] == 0:
            negative_count += 1

        object_bins[
            record["object_bin"]
        ] += 1

        for coco_class_id in record[
            "classes"
        ]:
            class_counts[
                category_to_yolo[
                    coco_class_id
                ]
            ] += 1

        for scale in record["scales"]:
            scales[scale] += 1

    print(
        "\nFoundry M3 — Evaluation Subset"
    )
    print("=" * 50)

    print(
        "Source images:       5000"
    )

    print(
        f"Selected images:     "
        f"{len(selected)}"
    )

    print(
        f"Negative images:     "
        f"{negative_count}"
    )

    print(
        f"Classes represented: "
        f"{len(class_counts)} / 80"
    )

    print(
        f"Selection seed:      "
        f"{SEED}"
    )

    print(
        "\nObject-count distribution:"
    )

    for key in [
        "negative",
        "1",
        "2-3",
        "4-10",
        "11+",
    ]:
        print(
            f"  {key:10s}: "
            f"{object_bins.get(key, 0):4d}"
        )

    print("\nScale coverage:")

    for key in [
        "small",
        "medium",
        "large",
    ]:
        print(
            f"  {key:10s}: "
            f"{scales.get(key, 0):4d}"
        )

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nManifest:")
    print(MANIFEST_PATH)

    print("\nDataset config:")
    print(DATA_YAML_PATH)


def main() -> None:
    """Create the frozen evaluation subset."""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            "COCO validation images not found: "
            f"{IMAGE_DIR}"
        )

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            "COCO annotations not found: "
            f"{ANNOTATIONS_PATH}"
        )

    print(
        "Loading official COCO val2017 "
        "annotations..."
    )

    coco = load_coco()

    print(
        f"COCO images:      "
        f"{len(coco['images'])}"
    )

    print(
        f"COCO annotations: "
        f"{len(coco['annotations'])}"
    )

    category_to_yolo, yolo_names = (
        build_category_mapping(coco)
    )

    if len(category_to_yolo) != 80:
        raise RuntimeError(
            "Expected 80 COCO categories, "
            f"found {len(category_to_yolo)}."
        )

    records = build_records(coco)

    print(
        f"Image records:    "
        f"{len(records)}"
    )

    selected = select_subset(records)

    copy_subset(
        selected,
        category_to_yolo,
    )

    write_yaml(yolo_names)

    manifest = build_manifest(
        selected,
        category_to_yolo,
        yolo_names,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print_summary(
        selected,
        category_to_yolo,
    )


if __name__ == "__main__":
    main()