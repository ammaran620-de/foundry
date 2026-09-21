from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# Foundry M3 — Deployment Accuracy Cross-Check
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "benchmark"
    / "evaluation"
    / "coco1000"
    / "data.yaml"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "deployment_accuracy.json"
)

IMAGE_SIZE = 640
BATCH_SIZE = 1
DEVICE = "cpu"
WORKERS = 0


# ------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------

MODELS = {
    "onnxruntime_cpu": (
        PROJECT_ROOT
        / "models"
        / "yolo26n.onnx"
    ),

    "openvino_fp16_cpu": (
        PROJECT_ROOT
        / "models"
        / "yolo26n_openvino_model"
    ),
}


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_model(
    name: str,
    model_path: Path,
) -> dict:

    print()
    print("=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Dataset: {DATA_YAML}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print()
    print("Loading model...")

    model = YOLO(
        str(model_path)
    )

    print(
        "Running validation..."
    )

    results = model.val(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=False,
        verbose=True,
    )

    box = results.box

    result = {
        "name":
            name,

        "model":
            str(model_path),

        "mAP50_95":
            float(box.map),

        "mAP50":
            float(box.map50),

        "mAP75":
            float(box.map75),

        "precision":
            float(box.mp),

        "recall":
            float(box.mr),

        "per_class_mAP":
            [
                float(value)
                for value in box.maps
            ],

        "speed_ms_per_image": {
            key:
                float(value)
            for key, value
            in results.speed.items()
        },
    }

    print()
    print("-" * 70)
    print(
        f"mAP50-95 : {result['mAP50_95']:.4f}"
    )

    print(
        f"mAP50    : {result['mAP50']:.4f}"
    )

    print(
        f"mAP75    : {result['mAP75']:.4f}"
    )

    print(
        f"Precision: {result['precision']:.4f}"
    )

    print(
        f"Recall   : {result['recall']:.4f}"
    )

    print(
        "Speed:"
    )

    for key, value in result[
        "speed_ms_per_image"
    ].items():

        print(
            f"  {key:<12}: "
            f"{value:.3f} ms"
        )

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Foundry M3 — Deployment Accuracy Cross-Check"
    )
    print("=" * 80)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n"
            f"{DATA_YAML}"
        )

    print(
        f"Dataset: {DATA_YAML}"
    )

    results = {}

    # --------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------

    results[
        "onnxruntime_cpu"
    ] = evaluate_model(
        "ONNX Runtime CPU",
        MODELS[
            "onnxruntime_cpu"
        ],
    )

    # --------------------------------------------------------
    # OpenVINO FP16
    # --------------------------------------------------------

    results[
        "openvino_fp16_cpu"
    ] = evaluate_model(
        "OpenVINO FP16 CPU",
        MODELS[
            "openvino_fp16_cpu"
        ],
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print()
    print()
    print("=" * 95)
    print(
        "DEPLOYMENT ACCURACY RESULTS"
    )
    print("=" * 95)

    print(
        f"{'Backend':<30}"
        f"{'mAP50-95':>14}"
        f"{'mAP50':>14}"
        f"{'mAP75':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
    )

    print("-" * 95)

    for key, result in results.items():

        print(
            f"{result['name']:<30}"
            f"{result['mAP50_95']:>14.4f}"
            f"{result['mAP50']:>14.4f}"
            f"{result['mAP75']:>14.4f}"
            f"{result['precision']:>14.4f}"
            f"{result['recall']:>14.4f}"
        )

    print("=" * 95)

    # --------------------------------------------------------
    # Compare against FP32
    # --------------------------------------------------------

    fp32_map = 0.4053

    print()
    print(
        "COMPARISON TO PYTORCH FP32 BASELINE"
    )
    print("-" * 70)

    print(
        f"PyTorch FP32 mAP50-95: "
        f"{fp32_map:.4f}"
    )

    for key, result in results.items():

        delta = (
            result["mAP50_95"]
            - fp32_map
        )

        percentage = (
            delta
            / fp32_map
            * 100.0
        )

        print(
            f"{result['name']:<25}"
            f"{result['mAP50_95']:.4f} "
            f"({delta:+.4f}, "
            f"{percentage:+.2f}%)"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment":
            "Foundry M3 deployment accuracy cross-check",

        "dataset":
            str(DATA_YAML),

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "device":
            DEVICE,

        "pytorch_fp32_reference": {
            "mAP50_95":
                fp32_map,
            "mAP50":
                0.5629,
            "mAP75":
                0.4356,
            "precision":
                0.6610,
            "recall":
                0.5181,
        },

        "results":
            results,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        "Deployment accuracy cross-check complete."
    )


if __name__ == "__main__":
    main()
    