from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import psutil


# ============================================================
# Foundry M3 — OpenVINO INT8 CPU vs iGPU Benchmark
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

WARMUP_RUNS = 20
TIMED_RUNS = 100


# ============================================================
# Helpers
# ============================================================

def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * (p / 100.0)

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def prepare_input() -> np.ndarray:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Benchmark image not found:\n{INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read benchmark image:\n{INPUT_IMAGE}"
        )

    # Resize to model input resolution.
    image = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # uint8 -> float32 [0, 1]
    image = image.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0,
    )

    return np.ascontiguousarray(image)


def benchmark_device(
    core: ov.Core,
    device: str,
    input_tensor: np.ndarray,
) -> dict:

    print()
    print("=" * 60)
    print(f"Device: {device}")
    print("=" * 60)

    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    print("Compiling model...")

    compile_start = time.perf_counter()

    compiled_model = core.compile_model(
        str(MODEL_XML),
        device,
    )

    compile_time = (
        time.perf_counter()
        - compile_start
    )

    infer_request = (
        compiled_model.create_infer_request()
    )

    input_layer = compiled_model.inputs[0]

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    input_name = input_layer.get_any_name()

    print(f"Input name: {input_name}")
    print(f"Input shape: {input_layer.shape}")
    print(
        f"Input type: "
        f"{input_layer.get_element_type()}"
    )

    # --------------------------------------------------------
    # Memory before inference
    # --------------------------------------------------------

    process = psutil.Process()

    rss_before = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    peak_rss = rss_before

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"Warmup: {WARMUP_RUNS} runs..."
    )

    for _ in range(WARMUP_RUNS):
        infer_request.infer(
            {input_name: input_tensor}
        )

    rss_after_warmup = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    peak_rss = max(
        peak_rss,
        rss_after_warmup,
    )

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    print(
        f"Timed inference: "
        f"{TIMED_RUNS} runs..."
    )

    latencies_ms: list[float] = []

    for _ in range(TIMED_RUNS):

        start = time.perf_counter()

        infer_request.infer(
            {input_name: input_tensor}
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies_ms.append(
            elapsed_ms
        )

        current_rss = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        peak_rss = max(
            peak_rss,
            current_rss,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_ms = statistics.mean(
        latencies_ms
    )

    median_ms = statistics.median(
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

    rss_after = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    try:
        full_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )
    except Exception:
        full_name = device

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {
        "device": device,
        "device_name": str(full_name),
        "compile_time_s": compile_time,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "mean_ms": mean_ms,
        "p50_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "fps": fps,
        "rss_before_mb": rss_before,
        "rss_after_warmup_mb": rss_after_warmup,
        "rss_after_mb": rss_after,
        "peak_rss_observed_mb": peak_rss,
    }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print(f"Device name : {result['device_name']}")
    print(
        f"Compile     : "
        f"{result['compile_time_s']:.3f} s"
    )
    print(
        f"Mean        : "
        f"{result['mean_ms']:.3f} ms"
    )
    print(
        f"P50         : "
        f"{result['p50_ms']:.3f} ms"
    )
    print(
        f"P95         : "
        f"{result['p95_ms']:.3f} ms"
    )
    print(
        f"Min         : "
        f"{result['min_ms']:.3f} ms"
    )
    print(
        f"Max         : "
        f"{result['max_ms']:.3f} ms"
    )
    print(
        f"Std         : "
        f"{result['std_ms']:.3f} ms"
    )
    print(
        f"FPS         : "
        f"{result['fps']:.2f}"
    )
    print(
        f"Peak RSS    : "
        f"{result['peak_rss_observed_mb']:.1f} MB"
    )

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("Foundry M3 — OpenVINO INT8 Benchmark")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"INT8 model not found:\n{MODEL_XML}"
        )

    print(f"Model: {MODEL_XML}")
    print(f"Input: {INPUT_IMAGE}")

    # --------------------------------------------------------
    # OpenVINO
    # --------------------------------------------------------

    core = ov.Core()

    print()
    print(
        f"Available devices: "
        f"{core.available_devices}"
    )

    # --------------------------------------------------------
    # Prepare input once
    # --------------------------------------------------------

    input_tensor = prepare_input()

    print(
        f"Prepared input: "
        f"{input_tensor.shape} "
        f"{input_tensor.dtype}"
    )

    # --------------------------------------------------------
    # Benchmark CPU
    # --------------------------------------------------------

    cpu_result = benchmark_device(
        core,
        "CPU",
        input_tensor,
    )

    # --------------------------------------------------------
    # Benchmark GPU
    # --------------------------------------------------------

    if "GPU" in core.available_devices:

        gpu_result = benchmark_device(
            core,
            "GPU",
            input_tensor,
        )

    else:
        print()
        print(
            "GPU device is not available. "
            "Skipping GPU benchmark."
        )
        gpu_result = None

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("FINAL INT8 COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<20}"
        f"{'CPU':>15}"
        f"{'GPU':>15}"
    )

    print("-" * 70)

    print(
        f"{'Mean ms':<20}"
        f"{cpu_result['mean_ms']:>15.3f}"
        f"{gpu_result['mean_ms']:>15.3f}"
        if gpu_result
        else
        f"{'Mean ms':<20}"
        f"{cpu_result['mean_ms']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'P50 ms':<20}"
        f"{cpu_result['p50_ms']:>15.3f}"
        f"{gpu_result['p50_ms']:>15.3f}"
        if gpu_result
        else
        f"{'P50 ms':<20}"
        f"{cpu_result['p50_ms']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'P95 ms':<20}"
        f"{cpu_result['p95_ms']:>15.3f}"
        f"{gpu_result['p95_ms']:>15.3f}"
        if gpu_result
        else
        f"{'P95 ms':<20}"
        f"{cpu_result['p95_ms']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'FPS':<20}"
        f"{cpu_result['fps']:>15.2f}"
        f"{gpu_result['fps']:>15.2f}"
        if gpu_result
        else
        f"{'FPS':<20}"
        f"{cpu_result['fps']:>15.2f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'Peak RSS MB':<20}"
        f"{cpu_result['peak_rss_observed_mb']:>15.1f}"
        f"{gpu_result['peak_rss_observed_mb']:>15.1f}"
        if gpu_result
        else
        f"{'Peak RSS MB':<20}"
        f"{cpu_result['peak_rss_observed_mb']:>15.1f}"
        f"{'N/A':>15}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()