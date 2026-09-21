from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ============================================================
# Foundry M3 — Full Pipeline Profiling
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_XML = (
    PROJECT_ROOT
    / "models"
    / "yolo26n_int8_openvino_model"
    / "yolo26n.xml"
)

INPUT_IMAGE = (
    PROJECT_ROOT
    / "benchmark"
    / "input"
    / "benchmark_640.png"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "pipeline_profile_int8.json"
)

WARMUP_RUNS = 10
TIMED_RUNS = 50
CPU_THREADS = 4


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
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


# ============================================================
# Timed helper
# ============================================================

def timed_call(
    function,
) -> tuple[object, float]:

    start = time.perf_counter()

    result = function()

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return result, elapsed_ms


# ============================================================
# Pipeline stages
# ============================================================

def decode_image() -> np.ndarray:

    image = cv2.imread(
        str(INPUT_IMAGE),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to decode image:\n"
            f"{INPUT_IMAGE}"
        )

    return image


def resize_image(
    image: np.ndarray,
) -> np.ndarray:

    return cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )


def convert_color(
    image: np.ndarray,
) -> np.ndarray:

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def normalize_image(
    image: np.ndarray,
) -> np.ndarray:

    tensor = (
        image.astype(np.float32)
        / 255.0
    )

    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    return np.ascontiguousarray(
        tensor,
        dtype=np.float32,
    )


# ============================================================
# Benchmark one device
# ============================================================

def benchmark_device(
    core: ov.Core,
    device: str,
) -> dict:

    print()
    print()
    print("=" * 80)
    print(
        f"PIPELINE PROFILE — {device}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {}

    if device == "CPU":
        config[
            ov.properties.inference_num_threads()
        ] = CPU_THREADS

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    print()
    print("Compiling model...")

    compile_start = time.perf_counter()

    compiled_model = core.compile_model(
        str(MODEL_XML),
        device,
        config,
    )

    compile_time = (
        time.perf_counter()
        - compile_start
    )

    request = (
        compiled_model.create_infer_request()
    )

    input_port = compiled_model.inputs[0]
    input_name = input_port.get_any_name()

    try:
        device_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )
    except Exception:
        device_name = device

    print(
        f"Device: {device_name}"
    )

    print(
        f"Compile: "
        f"{compile_time:.3f} s"
    )

    actual_threads = None

    if device == "CPU":

        try:
            actual_threads = int(
                compiled_model.get_property(
                    ov.properties.inference_num_threads()
                )
            )
        except Exception:
            actual_threads = CPU_THREADS

        print(
            f"CPU threads: "
            f"{actual_threads}"
        )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        f"Warmup: {WARMUP_RUNS} runs..."
    )

    for _ in range(WARMUP_RUNS):

        image = decode_image()
        image = resize_image(image)
        image = convert_color(image)
        tensor = normalize_image(image)

        request.infer(
            {
                input_name: tensor
            }
        )

    # --------------------------------------------------------
    # Stage storage
    # --------------------------------------------------------

    stages = {
        "decode_ms": [],
        "resize_ms": [],
        "color_conversion_ms": [],
        "normalization_ms": [],
        "inference_ms": [],
        "postprocess_ms": [],
        "end_to_end_ms": [],
    }

    # --------------------------------------------------------
    # Timed pipeline
    # --------------------------------------------------------

    print()
    print(
        f"Timed pipeline runs: "
        f"{TIMED_RUNS}"
    )

    for run_index in range(
        TIMED_RUNS
    ):

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        image, decode_ms = timed_call(
            decode_image
        )

        stages[
            "decode_ms"
        ].append(
            decode_ms
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image, resize_ms = timed_call(
            lambda image=image:
                resize_image(image)
        )

        stages[
            "resize_ms"
        ].append(
            resize_ms
        )

        # ----------------------------------------------------
        # Color conversion
        # ----------------------------------------------------

        image, color_ms = timed_call(
            lambda image=image:
                convert_color(image)
        )

        stages[
            "color_conversion_ms"
        ].append(
            color_ms
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        tensor, normalize_ms = timed_call(
            lambda image=image:
                normalize_image(image)
        )

        stages[
            "normalization_ms"
        ].append(
            normalize_ms
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        output, inference_ms = timed_call(
            lambda tensor=tensor:
                request.infer(
                    {
                        input_name: tensor
                    }
                )
        )

        stages[
            "inference_ms"
        ].append(
            inference_ms
        )

        # ----------------------------------------------------
        # Postprocess
        #
        # The exported OpenVINO model already contains NMS.
        # We only touch the output so this stage represents
        # lightweight application-side postprocessing.
        # ----------------------------------------------------

        def postprocess() -> None:
            _ = output

        _, postprocess_ms = timed_call(
            postprocess
        )

        stages[
            "postprocess_ms"
        ].append(
            postprocess_ms
        )

        # ----------------------------------------------------
        # End-to-end
        # ----------------------------------------------------

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        stages[
            "end_to_end_ms"
        ].append(
            total_ms
        )

        if (
            run_index + 1
        ) % 10 == 0:

            print(
                f"  completed "
                f"{run_index + 1}/"
                f"{TIMED_RUNS}"
            )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    summaries = {
        stage: summarize(values)
        for stage, values
        in stages.items()
    }

    end_to_end_mean = (
        summaries[
            "end_to_end_ms"
        ]["mean_ms"]
    )

    # --------------------------------------------------------
    # Stage percentages
    # --------------------------------------------------------

    stage_percentages = {}

    measured_stages = [
        "decode_ms",
        "resize_ms",
        "color_conversion_ms",
        "normalization_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    for stage in measured_stages:

        stage_mean = (
            summaries[stage]["mean_ms"]
        )

        if end_to_end_mean > 0:

            stage_percentages[stage] = (
                stage_mean
                / end_to_end_mean
                * 100.0
            )

        else:

            stage_percentages[stage] = 0.0

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "device": device,
        "device_name": str(device_name),
        "compile_time_s": compile_time,
        "cpu_threads_requested": (
            CPU_THREADS
            if device == "CPU"
            else None
        ),
        "cpu_threads_actual": actual_threads,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "stages": summaries,
        "stage_percentage_of_end_to_end":
            stage_percentages,
    }

    # --------------------------------------------------------
    # Console table
    # --------------------------------------------------------

    print()
    print("-" * 95)

    print(
        f"{'Stage':<28}"
        f"{'Mean ms':>14}"
        f"{'P50 ms':>14}"
        f"{'P95 ms':>14}"
        f"{'Share':>12}"
    )

    print("-" * 95)

    names = {
        "decode_ms":
            "Decode",
        "resize_ms":
            "Resize",
        "color_conversion_ms":
            "Color conversion",
        "normalization_ms":
            "Normalization",
        "inference_ms":
            "Inference",
        "postprocess_ms":
            "Postprocess",
        "end_to_end_ms":
            "END-TO-END",
    }

    for stage in [
        "decode_ms",
        "resize_ms",
        "color_conversion_ms",
        "normalization_ms",
        "inference_ms",
        "postprocess_ms",
        "end_to_end_ms",
    ]:

        summary = summaries[stage]

        if stage == "end_to_end_ms":
            share = 100.0
        else:
            share = stage_percentages[stage]

        print(
            f"{names[stage]:<28}"
            f"{summary['mean_ms']:>14.3f}"
            f"{summary['p50_ms']:>14.3f}"
            f"{summary['p95_ms']:>14.3f}"
            f"{share:>11.1f}%"
        )

    print("-" * 95)

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Foundry M3 — Full Pipeline Profiling"
    )
    print("=" * 80)

    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_XML}"
        )

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_IMAGE}"
        )

    core = ov.Core()

    print(
        f"Available devices: "
        f"{core.available_devices}"
    )

    print(
        f"Model: {MODEL_XML}"
    )

    print(
        f"Input: {INPUT_IMAGE}"
    )

    print()
    print(
        f"Warmup runs: {WARMUP_RUNS}"
    )

    print(
        f"Timed runs: {TIMED_RUNS}"
    )

    results = {}

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    results["CPU"] = benchmark_device(
        core,
        "CPU",
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if "GPU" in core.available_devices:

        results["GPU"] = benchmark_device(
            core,
            "GPU",
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print(
        "PIPELINE PROFILE COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'Stage':<30}"
        f"{'CPU ms':>18}"
        f"{'GPU ms':>18}"
    )

    print("-" * 100)

    comparison_stages = [
        (
            "decode_ms",
            "Decode",
        ),
        (
            "resize_ms",
            "Resize",
        ),
        (
            "color_conversion_ms",
            "Color conversion",
        ),
        (
            "normalization_ms",
            "Normalization",
        ),
        (
            "inference_ms",
            "Inference",
        ),
        (
            "postprocess_ms",
            "Postprocess",
        ),
        (
            "end_to_end_ms",
            "END-TO-END",
        ),
    ]

    cpu_result = results.get("CPU")
    gpu_result = results.get("GPU")

    for key, label in comparison_stages:

        cpu_value = "N/A"
        gpu_value = "N/A"

        if cpu_result is not None:

            cpu_value = (
                f"{cpu_result['stages'][key]['mean_ms']:.3f}"
            )

        if gpu_result is not None:

            gpu_value = (
                f"{gpu_result['stages'][key]['mean_ms']:.3f}"
            )

        print(
            f"{label:<30}"
            f"{cpu_value:>18}"
            f"{gpu_value:>18}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Bottleneck analysis
    # --------------------------------------------------------

    print()
    print(
        "BOTTLENECK ANALYSIS"
    )
    print(
        "-" * 70
    )

    for device_name, result in results.items():

        stage_candidates = {
            key: value["mean_ms"]
            for key, value
            in result["stages"].items()
            if key != "end_to_end_ms"
        }

        bottleneck_name, bottleneck_value = max(
            stage_candidates.items(),
            key=lambda item: item[1],
        )

        print(
            f"{device_name}: "
            f"{bottleneck_name} "
            f"({bottleneck_value:.3f} ms)"
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
            "Foundry M3 — INT8 full pipeline profiling",

        "model":
            str(MODEL_XML),

        "input":
            str(INPUT_IMAGE),

        "warmup_runs":
            WARMUP_RUNS,

        "timed_runs":
            TIMED_RUNS,

        "notes": [
            (
                "The OpenVINO model contains embedded NMS."
            ),
            (
                "Postprocess timing measures only the "
                "application-side output handoff."
            ),
            (
                "Detection counts are intentionally omitted "
                "because they are not required for the "
                "pipeline timing experiment."
            ),
        ],

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
        "Pipeline profiling complete."
    )


if __name__ == "__main__":
    main()