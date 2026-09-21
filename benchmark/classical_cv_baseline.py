from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Foundry M3 — Classical Computer Vision Baseline
# ============================================================
#
# Task:
#   Detect a moving object in a controlled static scene.
#
# Methods:
#   1. Background subtraction
#   2. Background subtraction + morphology
#   3. Dense optical flow
#   4. Template matching
#
# This experiment is intentionally constrained.
# It demonstrates where classical CV can be sufficient
# without claiming equivalence to general object detection.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "classical_cv_baseline.json"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

NUM_FRAMES = 300
WARMUP_FRAMES = 30

MORPH_KERNEL_SIZE = 5

MOTION_THRESHOLD = 25

OPTICAL_FLOW_THRESHOLD = 1.5

TEMPLATE_THRESHOLD = 0.65


# ============================================================
# Statistics
# ============================================================

def percentile(
    values: list[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    index = (
        (len(ordered) - 1)
        * p
        / 100.0
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = index - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def summarize(
    values: list[float],
) -> dict:

    if not values:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    return {
        "mean_ms":
            statistics.mean(values),

        "p50_ms":
            statistics.median(values),

        "p95_ms":
            percentile(values, 95),

        "min_ms":
            min(values),

        "max_ms":
            max(values),

        "std_ms":
            (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
    }


# ============================================================
# Synthetic scene generation
# ============================================================

def create_background() -> np.ndarray:
    """
    Create a deterministic static background.
    """

    background = np.zeros(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    # Background gradient
    for y in range(
        IMAGE_HEIGHT
    ):

        intensity = int(
            40
            + (
                y
                / IMAGE_HEIGHT
                * 35
            )
        )

        background[y, :] = (
            intensity,
            intensity,
            intensity,
        )

    # Static scene elements
    cv2.rectangle(
        background,
        (60, 80),
        (210, 210),
        (90, 90, 90),
        -1,
    )

    cv2.rectangle(
        background,
        (430, 100),
        (570, 230),
        (70, 70, 70),
        -1,
    )

    cv2.circle(
        background,
        (130, 500),
        80,
        (80, 80, 80),
        -1,
    )

    # Static grid lines
    for x in range(
        0,
        IMAGE_WIDTH,
        80,
    ):

        cv2.line(
            background,
            (x, 0),
            (x, IMAGE_HEIGHT),
            (55, 55, 55),
            1,
        )

    for y in range(
        0,
        IMAGE_HEIGHT,
        80,
    ):

        cv2.line(
            background,
            (0, y),
            (IMAGE_WIDTH, y),
            (55, 55, 55),
            1,
        )

    return background


def generate_frame(
    background: np.ndarray,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Generate one frame and its ground-truth moving-object mask.

    The moving object is a fixed-size rectangle.
    """

    frame = background.copy()

    # Horizontal movement
    max_x = IMAGE_WIDTH - 100

    x = int(
        (
            frame_index
            * 6
        )
        % max_x
    )

    y = 330

    box_width = 80
    box_height = 60

    x2 = x + box_width
    y2 = y + box_height

    # Ground truth mask
    ground_truth = np.zeros(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ),
        dtype=np.uint8,
    )

    cv2.rectangle(
        ground_truth,
        (x, y),
        (x2, y2),
        255,
        -1,
    )

    # Moving object
    cv2.rectangle(
        frame,
        (x, y),
        (x2, y2),
        (240, 240, 240),
        -1,
    )

    # Object internal pattern
    cv2.circle(
        frame,
        (
            x + 20,
            y + 20,
        ),
        10,
        (40, 40, 40),
        -1,
    )

    cv2.line(
        frame,
        (
            x + 45,
            y + 10,
        ),
        (
            x + 70,
            y + 45,
        ),
        (40, 40, 40),
        4,
    )

    return (
        frame,
        ground_truth,
        (
            x,
            y,
            x2,
            y2,
        ),
    )


# ============================================================
# Mask metrics
# ============================================================

def mask_iou(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:

    prediction_bool = (
        prediction > 0
    )

    truth_bool = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction_bool,
        truth_bool,
    ).sum()

    union = np.logical_or(
        prediction_bool,
        truth_bool,
    ).sum()

    if union == 0:
        return 1.0

    return (
        float(intersection)
        / float(union)
    )


def mask_precision(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:

    prediction_bool = (
        prediction > 0
    )

    truth_bool = (
        ground_truth > 0
    )

    tp = np.logical_and(
        prediction_bool,
        truth_bool,
    ).sum()

    fp = np.logical_and(
        prediction_bool,
        ~truth_bool,
    ).sum()

    if (
        tp + fp
    ) == 0:
        return 0.0

    return (
        float(tp)
        / float(tp + fp)
    )


def mask_recall(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:

    prediction_bool = (
        prediction > 0
    )

    truth_bool = (
        ground_truth > 0
    )

    tp = np.logical_and(
        prediction_bool,
        truth_bool,
    ).sum()

    fn = np.logical_and(
        ~prediction_bool,
        truth_bool,
    ).sum()

    if (
        tp + fn
    ) == 0:
        return 0.0

    return (
        float(tp)
        / float(tp + fn)
    )


# ============================================================
# Background subtraction
# ============================================================

def run_background_subtraction(
    frames: list[np.ndarray],
    truths: list[np.ndarray],
) -> dict:

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50,
        varThreshold=16,
        detectShadows=False,
    )

    kernel = None

    latencies = []
    ious = []
    precisions = []
    recalls = []

    # Warm up background model
    for frame in frames[:WARMUP_FRAMES]:

        subtractor.apply(
            frame,
            learningRate=0.05,
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    for index in range(
        WARMUP_FRAMES,
        len(frames),
    ):

        frame = frames[index]
        truth = truths[index]

        start = time.perf_counter()

        mask = subtractor.apply(
            frame,
            learningRate=0.002,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        ious.append(
            mask_iou(
                mask,
                truth,
            )
        )

        precisions.append(
            mask_precision(
                mask,
                truth,
            )
        )

        recalls.append(
            mask_recall(
                mask,
                truth,
            )
        )

    return {
        "latency":
            summarize(latencies),

        "mean_iou":
            statistics.mean(ious),

        "mean_precision":
            statistics.mean(precisions),

        "mean_recall":
            statistics.mean(recalls),

        "fps":
            (
                1000.0
                / statistics.mean(latencies)
                if latencies
                else 0.0
            ),
    }


# ============================================================
# Background subtraction + morphology
# ============================================================

def run_morphology(
    frames: list[np.ndarray],
    truths: list[np.ndarray],
) -> dict:

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50,
        varThreshold=16,
        detectShadows=False,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    latencies = []
    ious = []
    precisions = []
    recalls = []

    for frame in frames[:WARMUP_FRAMES]:

        subtractor.apply(
            frame,
            learningRate=0.05,
        )

    for index in range(
        WARMUP_FRAMES,
        len(frames),
    ):

        frame = frames[index]
        truth = truths[index]

        start = time.perf_counter()

        mask = subtractor.apply(
            frame,
            learningRate=0.002,
        )

        # Remove small noise
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        # Fill small gaps
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        # Threshold to binary mask
        _, mask = cv2.threshold(
            mask,
            MOTION_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        ious.append(
            mask_iou(
                mask,
                truth,
            )
        )

        precisions.append(
            mask_precision(
                mask,
                truth,
            )
        )

        recalls.append(
            mask_recall(
                mask,
                truth,
            )
        )

    return {
        "latency":
            summarize(latencies),

        "mean_iou":
            statistics.mean(ious),

        "mean_precision":
            statistics.mean(precisions),

        "mean_recall":
            statistics.mean(recalls),

        "fps":
            (
                1000.0
                / statistics.mean(latencies)
                if latencies
                else 0.0
            ),
    }


# ============================================================
# Dense optical flow
# ============================================================

def run_optical_flow(
    frames: list[np.ndarray],
    truths: list[np.ndarray],
) -> dict:

    latencies = []
    ious = []
    precisions = []
    recalls = []

    previous_gray = cv2.cvtColor(
        frames[0],
        cv2.COLOR_BGR2GRAY,
    )

    for index in range(
        1,
        len(frames),
    ):

        frame = frames[index]
        truth = truths[index]

        start = time.perf_counter()

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        magnitude, _ = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
        )

        mask = np.where(
            magnitude >= OPTICAL_FLOW_THRESHOLD,
            255,
            0,
        ).astype(
            np.uint8
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        if index >= WARMUP_FRAMES:

            latencies.append(
                elapsed_ms
            )

            ious.append(
                mask_iou(
                    mask,
                    truth,
                )
            )

            precisions.append(
                mask_precision(
                    mask,
                    truth,
                )
            )

            recalls.append(
                mask_recall(
                    mask,
                    truth,
                )
            )

        previous_gray = current_gray

    return {
        "latency":
            summarize(latencies),

        "mean_iou":
            statistics.mean(ious),

        "mean_precision":
            statistics.mean(precisions),

        "mean_recall":
            statistics.mean(recalls),

        "fps":
            (
                1000.0
                / statistics.mean(latencies)
                if latencies
                else 0.0
            ),
    }


# ============================================================
# Template matching
# ============================================================

def run_template_matching(
    frames: list[np.ndarray],
    truths: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> dict:

    latencies = []
    ious = []
    precisions = []
    recalls = []

    # Template from the first frame.
    x1, y1, x2, y2 = boxes[0]

    template = cv2.cvtColor(
        frames[0][
            y1:y2,
            x1:x2,
        ],
        cv2.COLOR_BGR2GRAY,
    )

    template_h, template_w = (
        template.shape
    )

    for index in range(
        WARMUP_FRAMES,
        len(frames),
    ):

        frame = frames[index]
        truth = truths[index]

        start = time.perf_counter()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        result = cv2.matchTemplate(
            gray,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, max_value, _, max_location = (
            cv2.minMaxLoc(result)
        )

        prediction = np.zeros(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            ),
            dtype=np.uint8,
        )

        if max_value >= TEMPLATE_THRESHOLD:

            px, py = max_location

            cv2.rectangle(
                prediction,
                (
                    px,
                    py,
                ),
                (
                    px + template_w,
                    py + template_h,
                ),
                255,
                -1,
            )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        ious.append(
            mask_iou(
                prediction,
                truth,
            )
        )

        precisions.append(
            mask_precision(
                prediction,
                truth,
            )
        )

        recalls.append(
            mask_recall(
                prediction,
                truth,
            )
        )

    return {
        "latency":
            summarize(latencies),

        "mean_iou":
            statistics.mean(ious),

        "mean_precision":
            statistics.mean(precisions),

        "mean_recall":
            statistics.mean(recalls),

        "fps":
            (
                1000.0
                / statistics.mean(latencies)
                if latencies
                else 0.0
            ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Foundry M3 — Classical CV Baseline"
    )
    print("=" * 80)

    print()
    print(
        "Task: constrained moving-object detection"
    )

    print(
        f"Resolution: "
        f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
    )

    print(
        f"Frames: {NUM_FRAMES}"
    )

    print(
        f"Warmup frames: {WARMUP_FRAMES}"
    )

    # --------------------------------------------------------
    # Generate deterministic scene
    # --------------------------------------------------------

    print()
    print(
        "Generating controlled motion dataset..."
    )

    background = create_background()

    frames = []
    truths = []
    boxes = []

    for frame_index in range(
        NUM_FRAMES
    ):

        frame, truth, box = (
            generate_frame(
                background,
                frame_index,
            )
        )

        frames.append(
            frame
        )

        truths.append(
            truth
        )

        boxes.append(
            box
        )

    print(
        f"Generated {len(frames)} frames."
    )

    # --------------------------------------------------------
    # Run methods
    # --------------------------------------------------------

    print()
    print(
        "Running background subtraction..."
    )

    background_result = (
        run_background_subtraction(
            frames,
            truths,
        )
    )

    print(
        "Running background subtraction + morphology..."
    )

    morphology_result = (
        run_morphology(
            frames,
            truths,
        )
    )

    print(
        "Running dense optical flow..."
    )

    optical_flow_result = (
        run_optical_flow(
            frames,
            truths,
        )
    )

    print(
        "Running template matching..."
    )

    template_result = (
        run_template_matching(
            frames,
            truths,
            boxes,
        )
    )

    results = {
        "background_subtraction":
            background_result,

        "background_subtraction_morphology":
            morphology_result,

        "optical_flow":
            optical_flow_result,

        "template_matching":
            template_result,
    }

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print()
    print()
    print("=" * 110)
    print(
        "CLASSICAL CV RESULTS"
    )
    print("=" * 110)

    print(
        f"{'Method':<38}"
        f"{'Mean ms':>14}"
        f"{'P50 ms':>14}"
        f"{'P95 ms':>14}"
        f"{'FPS':>12}"
        f"{'IoU':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
    )

    print("-" * 110)

    labels = {
        "background_subtraction":
            "Background subtraction",

        "background_subtraction_morphology":
            "Background subtraction + morphology",

        "optical_flow":
            "Dense optical flow",

        "template_matching":
            "Template matching",
    }

    for key, result in results.items():

        latency = result[
            "latency"
        ]

        print(
            f"{labels[key]:<38}"
            f"{latency['mean_ms']:>14.3f}"
            f"{latency['p50_ms']:>14.3f}"
            f"{latency['p95_ms']:>14.3f}"
            f"{result['fps']:>12.2f}"
            f"{result['mean_iou']:>12.4f}"
            f"{result['mean_precision']:>12.4f}"
            f"{result['mean_recall']:>12.4f}"
        )

    print("=" * 110)

    # --------------------------------------------------------
    # Engineering interpretation
    # --------------------------------------------------------

    print()
    print(
        "ENGINEERING INTERPRETATION"
    )
    print("-" * 80)

    print(
        "Background subtraction + morphology is appropriate "
        "when the camera is fixed and the objective is simply "
        "to detect motion against a mostly static background."
    )

    print(
        "Optical flow is useful when motion direction or "
        "motion magnitude matters."
    )

    print(
        "Template matching can work when the target has a "
        "stable appearance and scale."
    )

    print(
        "These methods do not replace general object detection "
        "when the scene contains multiple object categories, "
        "camera motion, changing appearance, or semantic "
        "classification requirements."
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment":
            "Foundry M3 — Classical CV baseline",

        "task":
            "Constrained moving-object detection",

        "resolution":
            [
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            ],

        "num_frames":
            NUM_FRAMES,

        "warmup_frames":
            WARMUP_FRAMES,

        "methods":
            results,

        "notes": [
            (
                "Synthetic deterministic scene with known "
                "ground-truth moving-object masks."
            ),
            (
                "This is a constrained baseline and is not "
                "a general substitute for neural object detection."
            ),
            (
                "Metrics are measured on the same generated "
                "sequence for all classical methods."
            ),
        ],
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
        "Classical CV baseline complete."
    )


if __name__ == "__main__":
    main()