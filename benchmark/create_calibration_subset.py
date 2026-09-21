from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ============================================================
# Foundry M3 — INT8 Calibration Subset Builder
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Official COCO val2017
COCO_ANNOTATIONS = (
    PROJECT_ROOT
    / "datasets"
    / "coco"
    / "annotations"
    / "instances_val2017.json"
)

COCO_IMAGES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "coco"
    / "images"
    / "val2017"
)

# Existing accuracy-evaluation subset.
# These images MUST NOT appear in the calibration set.
EVAL_MANIFEST = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "coco1000"
    / "manifest.json"
)

# New calibration dataset
OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "calibration300"
)

SEED = 20260921
CALIBRATION_SIZE = 300


# ============================================================
# Utility functions
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def object_count_bin(count: int) -> str:
    if count <= 0:
        return "negative"
    if count == 1:
        return "1"
    if 2 <= count <= 3:
        return "2-3"
    if 4 <= count <= 10:
        return "4-10"
    return "11+"


def bbox_to_yolo(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """
    COCO bbox:
        [x, y, width, height]

    YOLO detection:
        x_center, y_center, width, height
        all normalized to [0, 1]
    """

    if len(bbox) != 4:
        return None

    x, y, box_width, box_height = bbox

    if box_width <= 0 or box_height <= 0:
        return None

    if width <= 0 or height <= 0:
        return None

    x_center = x + (box_width / 2.0)
    y_center = y + (box_height / 2.0)

    return (
        x_center / width,
        y_center / height,
        box_width / width,
        box_height / height,
    )


def calculate_scale(
    area: float,
    image_width: int,
    image_height: int,
) -> str:
    """
    COCO-style approximate scale categorization.

    small   < 1% of image area
    medium  1%–10%
    large   > 10%
    """

    image_area = image_width * image_height

    if image_area <= 0:
        return "small"

    relative_area = area / image_area

    if relative_area < 0.01:
        return "small"

    if relative_area < 0.10:
        return "medium"

    return "large"


def build_category_map(coco: dict) -> tuple[dict[int, int], list[str]]:
    """
    Convert COCO category IDs into contiguous YOLO class IDs.

    Example:
        COCO category ID 1 -> YOLO class 0
        COCO category ID 2 -> YOLO class 1
        ...
    """

    categories = sorted(
        coco["categories"],
        key=lambda item: int(item["id"]),
    )

    category_map: dict[int, int] = {}
    names: list[str] = []

    for index, category in enumerate(categories):
        category_id = int(category["id"])
        category_name = str(category["name"])

        category_map[category_id] = index
        names.append(category_name)

    return category_map, names


def prepare_output_directory() -> None:
    """
    Recreate calibration300 from scratch.
    """

    if OUTPUT_DIR.exists():
        print(f"Removing existing calibration directory:")
        print(f"  {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "images").mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "labels").mkdir(
        parents=True,
        exist_ok=True,
    )


def write_yaml(class_names: list[str]) -> Path:
    """
    Write a YOLO-compatible dataset YAML.

    IMPORTANT:
    path is relative to the project root because the script
    is executed from the M3 repository root.
    """

    yaml_path = OUTPUT_DIR / "data.yaml"

    lines = [
        "path: benchmark/evaluation/calibration300",
        "train: images",
        "val: images",
        "names:",
    ]

    for index, name in enumerate(class_names):
        # YAML-safe quoted class name
        safe_name = name.replace('"', '\\"')
        lines.append(f'  {index}: "{safe_name}"')

    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return yaml_path


def write_manifest(
    selected: list[dict],
    eval_ids: set[int],
    object_distribution: dict[str, int],
    scale_distribution: dict[str, int],
    class_names: list[str],
) -> Path:

    manifest = {
        "name": "Foundry M3 — INT8 Calibration Set",
        "source": "COCO val2017",
        "purpose": "OpenVINO INT8 post-training calibration",
        "selection_seed": SEED,
        "size": len(selected),
        "disjoint_from_evaluation": True,
        "evaluation_subset_size": len(eval_ids),
        "object_count_distribution": object_distribution,
        "scale_distribution": scale_distribution,
        "class_count": len(class_names),
        "images": [
            {
                "image_id": int(item["image_id"]),
                "file_name": item["file_name"],
                "width": int(item["width"]),
                "height": int(item["height"]),
                "object_count": int(item["object_count"]),
            }
            for item in selected
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest_path


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("Foundry M3 — INT8 Calibration Subset")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate required inputs
    # --------------------------------------------------------

    if not COCO_ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found:\n{COCO_ANNOTATIONS}"
        )

    if not COCO_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"COCO image directory not found:\n{COCO_IMAGES_DIR}"
        )

    if not EVAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found:\n{EVAL_MANIFEST}"
        )

    # --------------------------------------------------------
    # Load COCO
    # --------------------------------------------------------

    print()
    print("Loading official COCO val2017 annotations...")

    coco = load_json(COCO_ANNOTATIONS)

    print(f"COCO images:      {len(coco['images'])}")
    print(f"COCO annotations: {len(coco['annotations'])}")
    print(f"COCO categories:   {len(coco['categories'])}")

    # --------------------------------------------------------
    # Load evaluation manifest
    # --------------------------------------------------------

    eval_manifest = load_json(EVAL_MANIFEST)

    evaluation_images = eval_manifest.get("images", [])

    eval_ids = {
        int(item["image_id"])
        for item in evaluation_images
    }

    print()
    print(
        f"Evaluation images excluded: {len(eval_ids)}"
    )

    # --------------------------------------------------------
    # Build category mapping
    # --------------------------------------------------------

    category_map, class_names = build_category_map(coco)

    if len(class_names) != 80:
        raise RuntimeError(
            f"Expected 80 COCO classes, found {len(class_names)}."
        )

    # --------------------------------------------------------
    # Build image index
    # --------------------------------------------------------

    image_by_id = {
        int(image["id"]): image
        for image in coco["images"]
    }

    # --------------------------------------------------------
    # Build annotation index
    # --------------------------------------------------------

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in coco["annotations"]:

        image_id = int(annotation["image_id"])

        # Ignore crowd annotations for YOLO calibration labels.
        if int(annotation.get("iscrowd", 0)) == 1:
            continue

        category_id = int(annotation["category_id"])

        if category_id not in category_map:
            continue

        annotations_by_image[image_id].append(annotation)

    # --------------------------------------------------------
    # Candidate pool
    #
    # Exclude:
    #   1. evaluation images
    #   2. images with no usable objects
    # --------------------------------------------------------

    candidates: list[dict] = []

    for image_id, image in image_by_id.items():

        if image_id in eval_ids:
            continue

        annotations = annotations_by_image.get(
            image_id,
            [],
        )

        if not annotations:
            continue

        candidates.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations,
                "object_count": len(annotations),
            }
        )

    print(
        f"Remaining object-containing images: "
        f"{len(candidates)}"
    )

    if len(candidates) < CALIBRATION_SIZE:
        raise RuntimeError(
            "Not enough remaining images to create "
            f"a {CALIBRATION_SIZE}-image calibration set."
        )

    # --------------------------------------------------------
    # Stratify by object count
    # --------------------------------------------------------

    bins: dict[str, list[dict]] = defaultdict(list)

    for item in candidates:
        key = object_count_bin(
            item["object_count"]
        )
        bins[key].append(item)

    rng = random.Random(SEED)

    for items in bins.values():
        rng.shuffle(items)

    # Target distribution
    #
    # Total = 300
    #
    # 1       -> 90
    # 2-3     -> 90
    # 4-10    -> 75
    # 11+     -> 45
    # --------------------------------
    #             300
    #
    quotas = {
        "1": 90,
        "2-3": 90,
        "4-10": 75,
        "11+": 45,
    }

    selected: list[dict] = []

    for bin_name, quota in quotas.items():

        available = bins.get(
            bin_name,
            [],
        )

        take = min(
            quota,
            len(available),
        )

        selected.extend(
            available[:take]
        )

    # --------------------------------------------------------
    # Fill any shortfall
    # --------------------------------------------------------

    selected_ids = {
        int(item["image_id"])
        for item in selected
    }

    remaining = [
        item
        for item in candidates
        if int(item["image_id"]) not in selected_ids
    ]

    rng.shuffle(remaining)

    shortfall = CALIBRATION_SIZE - len(selected)

    if shortfall > 0:
        selected.extend(
            remaining[:shortfall]
        )

    # Deterministic final ordering
    selected.sort(
        key=lambda item: int(item["image_id"])
    )

    # Safety check
    if len(selected) != CALIBRATION_SIZE:
        raise RuntimeError(
            f"Expected {CALIBRATION_SIZE} selected images, "
            f"got {len(selected)}."
        )

    # --------------------------------------------------------
    # Ensure absolute disjointness
    # --------------------------------------------------------

    selected_ids = {
        int(item["image_id"])
        for item in selected
    }

    overlap = selected_ids.intersection(
        eval_ids
    )

    if overlap:
        raise RuntimeError(
            "CRITICAL: calibration/evaluation overlap detected: "
            f"{len(overlap)} images."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Generate calibration dataset
    # --------------------------------------------------------

    object_distribution = {
        "negative": 0,
        "1": 0,
        "2-3": 0,
        "4-10": 0,
        "11+": 0,
    }

    scale_distribution = {
        "small": 0,
        "medium": 0,
        "large": 0,
    }

    manifest_images: list[dict] = []

    copied_count = 0
    label_count = 0

    print()
    print("Generating calibration dataset...")

    for item in selected:

        image_id = int(item["image_id"])
        file_name = item["file_name"]
        width = int(item["width"])
        height = int(item["height"])
        annotations = item["annotations"]

        source_image = (
            COCO_IMAGES_DIR / file_name
        )

        destination_image = (
            OUTPUT_DIR / "images" / file_name
        )

        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing COCO image:\n{source_image}"
            )

        shutil.copy2(
            source_image,
            destination_image,
        )

        copied_count += 1

        # ----------------------------------------------------
        # Generate YOLO label
        # ----------------------------------------------------

        label_file_name = (
            Path(file_name)
            .with_suffix(".txt")
            .name
        )

        label_path = (
            OUTPUT_DIR / "labels" / label_file_name
        )

        yolo_lines: list[str] = []

        for annotation in annotations:

            category_id = int(
                annotation["category_id"]
            )

            class_id = category_map.get(
                category_id
            )

            if class_id is None:
                continue

            yolo_box = bbox_to_yolo(
                annotation["bbox"],
                width,
                height,
            )

            if yolo_box is None:
                continue

            x_center, y_center, box_width, box_height = (
                yolo_box
            )

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            scale = calculate_scale(
                float(annotation.get("area", 0.0)),
                width,
                height,
            )

            scale_distribution[scale] += 1

        label_path.write_text(
            "\n".join(yolo_lines)
            + (
                "\n"
                if yolo_lines
                else ""
            ),
            encoding="utf-8",
        )

        label_count += 1

        # ----------------------------------------------------
        # Object-count distribution
        # ----------------------------------------------------

        count_bin = object_count_bin(
            len(annotations)
        )

        object_distribution[count_bin] += 1

        manifest_images.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "object_count": len(annotations),
            }
        )

    # --------------------------------------------------------
    # Write YAML
    # --------------------------------------------------------

    yaml_path = write_yaml(
        class_names
    )

    # --------------------------------------------------------
    # Write manifest
    # --------------------------------------------------------

    manifest_path = write_manifest(
        selected=selected,
        eval_ids=eval_ids,
        object_distribution=object_distribution,
        scale_distribution=scale_distribution,
        class_names=class_names,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    generated_images = list(
        (OUTPUT_DIR / "images").glob("*")
    )

    generated_labels = list(
        (OUTPUT_DIR / "labels").glob("*.txt")
    )

    if len(generated_images) != CALIBRATION_SIZE:
        raise RuntimeError(
            "Image count verification failed: "
            f"{len(generated_images)}"
        )

    if len(generated_labels) != CALIBRATION_SIZE:
        raise RuntimeError(
            "Label count verification failed: "
            f"{len(generated_labels)}"
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("Calibration dataset created successfully.")
    print("=" * 60)

    print(
        f"Source images:        {len(coco['images'])}"
    )

    print(
        f"Evaluation excluded:  {len(eval_ids)}"
    )

    print(
        f"Calibration images:   {len(selected)}"
    )

    print(
        f"Classes represented:  {len(class_names)} / 80"
    )

    print(
        f"Images copied:        {copied_count}"
    )

    print(
        f"Labels generated:     {label_count}"
    )

    print(
        f"Selection seed:       {SEED}"
    )

    print()
    print("Object-count distribution:")

    for key in [
        "negative",
        "1",
        "2-3",
        "4-10",
        "11+",
    ]:
        print(
            f"  {key:10s}: "
            f"{object_distribution[key]:4d}"
        )

    print()
    print("Object scale distribution:")

    for key in [
        "small",
        "medium",
        "large",
    ]:
        print(
            f"  {key:10s}: "
            f"{scale_distribution[key]:4d}"
        )

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("Images:")
    print(OUTPUT_DIR / "images")

    print()
    print("Labels:")
    print(OUTPUT_DIR / "labels")

    print()
    print("Manifest:")
    print(manifest_path)

    print()
    print("Dataset config:")
    print(yaml_path)

    print()
    print("Disjointness check:")
    print(
        f"  Evaluation overlap: {len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Calibration set is not disjoint from evaluation set."
        )

    print("  PASS")


if __name__ == "__main__":
    main()