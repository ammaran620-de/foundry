from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ============================================================
# Foundry M3 — CPU Thread Scaling Benchmark
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
    / "thread_scaling_int8.json"
)

# Your CPU:
# 10 physical cores / 12 logical threads
THREAD_COUNTS = [1, 2, 4, 6, 8, 10, 12]

WARMUP_RUNS = 10
TIMED_RUNS = 50


# ============================================================
# Statistics
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Linear-interpolated percentile.
    """

    if not values:
        return 0.0

    ordered = sorted(values)

    index = (
        (len(ordered) - 1)
        * percentile_value
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


# ============================================================
# Input preparation
# ============================================================

def prepare_input() -> np.ndarray:
    """
    Prepare benchmark_640.png into:

        [1, 3, 640, 640]

    float32 RGB tensor in [0, 1].
    """

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Benchmark input not found:\n{INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read benchmark image:\n"
            f"{INPUT_IMAGE}"
        )

    image = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image = np.expand_dims(
        image,
        axis=0,
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32,
    )


# ============================================================
# Single thread-count benchmark
# ============================================================

def benchmark_thread_count(
    core: ov.Core,
    model: ov.Model,
    input_tensor: np.ndarray,
    requested_threads: int,
) -> dict:
    """
    Benchmark one CPU thread configuration.

    Important:
    - One infer request
    - One synchronous inference at a time
    - Only inference thread count is varied
    - Same model and same input for every test
    """

    print()
    print("=" * 60)
    print(
        f"Testing {requested_threads} CPU thread(s)"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # OpenVINO configuration
    # --------------------------------------------------------

    config = {
        ov.properties.inference_num_threads():
            requested_threads,
    }

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    compile_start = time.perf_counter()

    compiled_model = core.compile_model(
        model,
        "CPU",
        config,
    )

    compile_time = (
        time.perf_counter()
        - compile_start
    )

    # --------------------------------------------------------
    # Inference request
    # --------------------------------------------------------

    infer_request = (
        compiled_model.create_infer_request()
    )

    input_port = compiled_model.inputs[0]

    input_name = (
        input_port.get_any_name()
    )

    # --------------------------------------------------------
    # Read back actual runtime thread count
    # --------------------------------------------------------

    try:
        actual_threads = int(
            compiled_model.get_property(
                ov.properties.inference_num_threads()
            )
        )
    except Exception:
        actual_threads = requested_threads

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"Warmup runs: {WARMUP_RUNS}"
    )

    for _ in range(WARMUP_RUNS):
        infer_request.infer(
            {
                input_name: input_tensor
            }
        )

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    print(
        f"Timed runs: {TIMED_RUNS}"
    )

    latencies_ms: list[float] = []

    for run_index in range(
        TIMED_RUNS
    ):

        start = time.perf_counter()

        infer_request.infer(
            {
                input_name: input_tensor
            }
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies_ms.append(
            elapsed_ms
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
    # Statistics
    # --------------------------------------------------------

    mean_ms = statistics.mean(
        latencies_ms
    )

    p50_ms = statistics.median(
        latencies_ms
    )

    p95_ms = percentile(
        latencies_ms,
        95,
    )

    min_ms = min(
        latencies_ms
    )

    max_ms = max(
        latencies_ms
    )

    std_ms = (
        statistics.stdev(
            latencies_ms
        )
        if len(latencies_ms) > 1
        else 0.0
    )

    fps = (
        1000.0 / mean_ms
        if mean_ms > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Result object
    # --------------------------------------------------------

    result = {
        "requested_threads": requested_threads,
        "actual_threads": actual_threads,
        "compile_time_s": compile_time,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "fps": fps,
    }

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print(
        f"Requested threads : "
        f"{requested_threads}"
    )

    print(
        f"Actual threads    : "
        f"{actual_threads}"
    )

    print(
        f"Compile time      : "
        f"{compile_time:.3f} s"
    )

    print(
        f"Mean              : "
        f"{mean_ms:.3f} ms"
    )

    print(
        f"P50               : "
        f"{p50_ms:.3f} ms"
    )

    print(
        f"P95               : "
        f"{p95_ms:.3f} ms"
    )

    print(
        f"Min               : "
        f"{min_ms:.3f} ms"
    )

    print(
        f"Max               : "
        f"{max_ms:.3f} ms"
    )

    print(
        f"Std               : "
        f"{std_ms:.3f} ms"
    )

    print(
        f"FPS               : "
        f"{fps:.2f}"
    )

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Foundry M3 — CPU Thread Scaling"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"INT8 model not found:\n"
            f"{MODEL_XML}"
        )

    # --------------------------------------------------------
    # OpenVINO
    # --------------------------------------------------------

    core = ov.Core()

    cpu_name = core.get_property(
        "CPU",
        "FULL_DEVICE_NAME",
    )

    print(
        f"CPU device: {cpu_name}"
    )

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

    print(
        f"Thread configurations: "
        f"{THREAD_COUNTS}"
    )

    print(
        f"Warmup runs: {WARMUP_RUNS}"
    )

    print(
        f"Timed runs: {TIMED_RUNS}"
    )

    # --------------------------------------------------------
    # Read model once
    # --------------------------------------------------------

    print()
    print("Reading OpenVINO model...")

    model = core.read_model(
        str(MODEL_XML)
    )

    # --------------------------------------------------------
    # Prepare input once
    # --------------------------------------------------------

    input_tensor = prepare_input()

    print(
        f"Input tensor: "
        f"{input_tensor.shape} "
        f"{input_tensor.dtype}"
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results: list[dict] = []

    for thread_count in THREAD_COUNTS:

        result = benchmark_thread_count(
            core=core,
            model=model,
            input_tensor=input_tensor,
            requested_threads=thread_count,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Find best configurations
    # --------------------------------------------------------

    best_mean = min(
        results,
        key=lambda item: item["mean_ms"],
    )

    best_p50 = min(
        results,
        key=lambda item: item["p50_ms"],
    )

    best_p95 = min(
        results,
        key=lambda item: item["p95_ms"],
    )

    best_fps = max(
        results,
        key=lambda item: item["fps"],
    )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print()
    print()
    print("=" * 95)
    print(
        "FINAL THREAD SCALING RESULTS"
    )
    print("=" * 95)

    print(
        f"{'Threads':>8}"
        f"{'Mean ms':>14}"
        f"{'P50 ms':>14}"
        f"{'P95 ms':>14}"
        f"{'Min ms':>14}"
        f"{'Max ms':>14}"
        f"{'FPS':>12}"
    )

    print("-" * 95)

    for result in results:

        print(
            f"{result['actual_threads']:>8}"
            f"{result['mean_ms']:>14.3f}"
            f"{result['p50_ms']:>14.3f}"
            f"{result['p95_ms']:>14.3f}"
            f"{result['min_ms']:>14.3f}"
            f"{result['max_ms']:>14.3f}"
            f"{result['fps']:>12.2f}"
        )

    print("=" * 95)

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    print()
    print(
        "THREAD-SCALING FINDINGS"
    )
    print("-" * 70)

    print(
        f"Best mean latency : "
        f"{best_mean['actual_threads']} threads "
        f"→ {best_mean['mean_ms']:.3f} ms"
    )

    print(
        f"Best P50 latency  : "
        f"{best_p50['actual_threads']} threads "
        f"→ {best_p50['p50_ms']:.3f} ms"
    )

    print(
        f"Best P95 latency  : "
        f"{best_p95['actual_threads']} threads "
        f"→ {best_p95['p95_ms']:.3f} ms"
    )

    print(
        f"Best throughput    : "
        f"{best_fps['actual_threads']} threads "
        f"→ {best_fps['fps']:.2f} FPS"
    )

    # --------------------------------------------------------
    # Compare one-thread baseline with best
    # --------------------------------------------------------

    one_thread = next(
        (
            item
            for item in results
            if item["actual_threads"] == 1
        ),
        None,
    )

    if one_thread is not None:

        speedup = (
            one_thread["mean_ms"]
            / best_mean["mean_ms"]
        )

        print()
        print(
            f"Speedup vs 1 thread: "
            f"{speedup:.2f}x"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "Foundry M3 CPU thread scaling"
        ),
        "model": str(MODEL_XML),
        "device": str(cpu_name),
        "thread_counts": THREAD_COUNTS,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "input_shape": list(
            input_tensor.shape
        ),
        "results": results,
        "best_mean_latency": best_mean,
        "best_p50_latency": best_p50,
        "best_p95_latency": best_p95,
        "best_throughput": best_fps,
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
        f"Results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print()
    print(
        "Thread-scaling benchmark complete."
    )


if __name__ == "__main__":
    main()