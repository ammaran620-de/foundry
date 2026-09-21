from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import psutil


# ============================================================
# Foundry M3 — Robust 10-Minute Sustained Benchmark
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
    / "sustained_int8_robust.json"
)

DURATION_SECONDS = 600
WARMUP_RUNS = 20
REPORT_INTERVAL_SECONDS = 60

CPU_THREADS = 4

# Never allow one inference to block forever.
INFERENCE_TIMEOUT_MS = 5000


# ============================================================
# Statistics
# ============================================================

def percentile(
    values: list[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def summarize(
    values: list[float],
    elapsed_seconds: float,
) -> dict:

    if not values:
        return {
            "runs": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "fps": 0.0,
        }

    mean_ms = statistics.mean(values)

    return {
        "runs": len(values),
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "fps": (
            len(values) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
    }


# ============================================================
# Input preparation
# ============================================================

def prepare_input() -> np.ndarray:

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read:\n{INPUT_IMAGE}"
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
# Memory
# ============================================================

def rss_mb(
    process: psutil.Process,
) -> float:

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


# ============================================================
# Robust inference
# ============================================================

def run_inference(
    request: ov.InferRequest,
    input_name: str,
    input_tensor: np.ndarray,
) -> tuple[bool, float]:

    start = time.perf_counter()

    request.start_async(
        {
            input_name: input_tensor
        }
    )

    completed = request.wait_for(
        INFERENCE_TIMEOUT_MS
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    if not completed:

        try:
            request.cancel()
        except Exception:
            pass

        return False, elapsed_ms

    return True, elapsed_ms


# ============================================================
# One sustained test
# ============================================================

def run_device(
    core: ov.Core,
    device: str,
    input_tensor: np.ndarray,
) -> dict:

    print()
    print()
    print("#" * 85)
    print(
        f"SUSTAINED TEST — {device}"
    )
    print("#" * 85)

    # --------------------------------------------------------
    # Compile configuration
    # --------------------------------------------------------

    config = {}

    if device == "CPU":
        config[
            ov.properties.inference_num_threads()
        ] = CPU_THREADS

    print()
    print("Compiling model...")

    compile_start = time.perf_counter()

    compiled = core.compile_model(
        str(MODEL_XML),
        device,
        config,
    )

    compile_time = (
        time.perf_counter()
        - compile_start
    )

    request = (
        compiled.create_infer_request()
    )

    input_port = compiled.inputs[0]
    input_name = input_port.get_any_name()

    try:
        device_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )
    except Exception:
        device_name = device

    print(
        f"Device name: {device_name}"
    )

    print(
        f"Compile time: "
        f"{compile_time:.3f} s"
    )

    actual_threads = None

    if device == "CPU":

        try:
            actual_threads = int(
                compiled.get_property(
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
    # Memory
    # --------------------------------------------------------

    process = psutil.Process()

    rss_before = rss_mb(process)
    peak_rss = rss_before

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        f"Warmup: {WARMUP_RUNS} runs..."
    )

    warmup_latencies = []

    for i in range(WARMUP_RUNS):

        completed, latency = run_inference(
            request,
            input_name,
            input_tensor,
        )

        if not completed:
            raise RuntimeError(
                "Inference timed out during warmup."
            )

        warmup_latencies.append(
            latency
        )

    rss_after_warmup = rss_mb(
        process
    )

    peak_rss = max(
        peak_rss,
        rss_after_warmup,
    )

    # --------------------------------------------------------
    # Sustained loop
    # --------------------------------------------------------

    print()
    print(
        "Starting 10-minute sustained test..."
    )

    print(
        f"Per-inference timeout: "
        f"{INFERENCE_TIMEOUT_MS} ms"
    )

    print(
        "Progress every 60 seconds."
    )

    print()

    start_time = time.perf_counter()

    next_report = (
        start_time
        + REPORT_INTERVAL_SECONDS
    )

    all_latencies: list[float] = []

    minute_latencies: list[float] = []

    minute_results: list[dict] = []

    timeout_count = 0
    run_count = 0

    while True:

        current_time = time.perf_counter()

        elapsed = (
            current_time
            - start_time
        )

        if elapsed >= DURATION_SECONDS:
            break

        completed, latency = run_inference(
            request,
            input_name,
            input_tensor,
        )

        if completed:

            all_latencies.append(
                latency
            )

            minute_latencies.append(
                latency
            )

            run_count += 1

        else:

            timeout_count += 1

            print(
                f"WARNING: inference timeout "
                f"at {elapsed:.1f}s "
                f"(count={timeout_count})"
            )

        # ----------------------------------------------------
        # Memory sampling
        # ----------------------------------------------------

        if (
            run_count + timeout_count
        ) % 20 == 0:

            current_rss = rss_mb(
                process
            )

            peak_rss = max(
                peak_rss,
                current_rss,
            )

        # ----------------------------------------------------
        # Minute reporting
        # ----------------------------------------------------

        current_time = time.perf_counter()

        if current_time >= next_report:

            report_elapsed = (
                current_time
                - start_time
            )

            minute_summary = summarize(
                minute_latencies,
                REPORT_INTERVAL_SECONDS,
            )

            current_rss = rss_mb(
                process
            )

            peak_rss = max(
                peak_rss,
                current_rss,
            )

            record = {
                "minute":
                    len(minute_results) + 1,
                "elapsed_seconds":
                    report_elapsed,
                "successful_runs":
                    len(minute_latencies),
                "timeouts":
                    timeout_count,
                "mean_ms":
                    minute_summary["mean_ms"],
                "p50_ms":
                    minute_summary["p50_ms"],
                "p95_ms":
                    minute_summary["p95_ms"],
                "fps":
                    minute_summary["fps"],
                "rss_mb":
                    current_rss,
            }

            minute_results.append(
                record
            )

            print(
                f"[{report_elapsed:7.1f}s] "
                f"runs={run_count:6d} "
                f"timeouts={timeout_count:3d} "
                f"mean={minute_summary['mean_ms']:7.2f} ms "
                f"p50={minute_summary['p50_ms']:7.2f} ms "
                f"p95={minute_summary['p95_ms']:7.2f} ms "
                f"fps={minute_summary['fps']:6.2f} "
                f"RSS={current_rss:7.1f} MB"
            )

            minute_latencies = []

            next_report += (
                REPORT_INTERVAL_SECONDS
            )

    # --------------------------------------------------------
    # Final measurements
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - start_time
    )

    rss_after = rss_mb(
        process
    )

    peak_rss = max(
        peak_rss,
        rss_after,
    )

    overall = summarize(
        all_latencies,
        total_elapsed,
    )

    # --------------------------------------------------------
    # First / last minute
    # --------------------------------------------------------

    first_minute = (
        minute_results[0]
        if minute_results
        else None
    )

    last_minute = (
        minute_results[-1]
        if minute_results
        else None
    )

    drift_percent = None

    if (
        first_minute is not None
        and last_minute is not None
        and first_minute["mean_ms"] > 0
    ):

        drift_percent = (
            (
                last_minute["mean_ms"]
                - first_minute["mean_ms"]
            )
            / first_minute["mean_ms"]
        ) * 100.0

    # --------------------------------------------------------
    # Result object
    # --------------------------------------------------------

    result = {
        "device":
            device,

        "device_name":
            str(device_name),

        "model":
            str(MODEL_XML),

        "duration_target_seconds":
            DURATION_SECONDS,

        "duration_actual_seconds":
            total_elapsed,

        "warmup_runs":
            WARMUP_RUNS,

        "successful_runs":
            run_count,

        "timeout_count":
            timeout_count,

        "timeout_limit_ms":
            INFERENCE_TIMEOUT_MS,

        "cpu_threads_requested":
            (
                CPU_THREADS
                if device == "CPU"
                else None
            ),

        "cpu_threads_actual":
            actual_threads,

        "compile_time_s":
            compile_time,

        "overall":
            overall,

        "memory": {
            "rss_before_mb":
                rss_before,

            "rss_after_warmup_mb":
                rss_after_warmup,

            "rss_after_mb":
                rss_after,

            "peak_rss_observed_mb":
                peak_rss,
        },

        "first_minute":
            first_minute,

        "last_minute":
            last_minute,

        "latency_drift_percent":
            drift_percent,

        "per_minute":
            minute_results,
    }

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("-" * 85)

    print(
        f"Successful runs : "
        f"{run_count}"
    )

    print(
        f"Timeouts        : "
        f"{timeout_count}"
    )

    print(
        f"Elapsed         : "
        f"{total_elapsed:.2f} s"
    )

    print(
        f"Mean latency    : "
        f"{overall['mean_ms']:.3f} ms"
    )

    print(
        f"P50 latency     : "
        f"{overall['p50_ms']:.3f} ms"
    )

    print(
        f"P95 latency     : "
        f"{overall['p95_ms']:.3f} ms"
    )

    print(
        f"Min latency     : "
        f"{overall['min_ms']:.3f} ms"
    )

    print(
        f"Max latency     : "
        f"{overall['max_ms']:.3f} ms"
    )

    print(
        f"Std latency     : "
        f"{overall['std_ms']:.3f} ms"
    )

    print(
        f"Sustained FPS   : "
        f"{overall['fps']:.2f}"
    )

    print(
        f"Peak host RSS   : "
        f"{peak_rss:.1f} MB"
    )

    if drift_percent is not None:

        print(
            f"Latency drift   : "
            f"{drift_percent:+.2f}%"
        )

    print("-" * 85)

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Foundry M3 — Robust 10-Minute Sustained Benchmark"
    )
    print("=" * 85)

    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"INT8 model not found:\n{MODEL_XML}"
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

    input_tensor = prepare_input()

    print(
        f"Input tensor: "
        f"{input_tensor.shape} "
        f"{input_tensor.dtype}"
    )

    results = {}

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    results["CPU"] = run_device(
        core,
        "CPU",
        input_tensor,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if "GPU" in core.available_devices:

        results["GPU"] = run_device(
            core,
            "GPU",
            input_tensor,
        )

    else:

        print()
        print(
            "GPU unavailable. "
            "Skipping GPU."
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print()
    print("=" * 105)
    print(
        "10-MINUTE SUSTAINED COMPARISON"
    )
    print("=" * 105)

    print(
        f"{'Metric':<32}"
        f"{'CPU':>20}"
        f"{'GPU':>20}"
    )

    print("-" * 105)

    cpu = results.get("CPU")
    gpu = results.get("GPU")

    def fmt(
        result: dict | None,
        section: str,
        key: str,
        decimals: int = 3,
    ) -> str:

        if result is None:
            return "N/A"

        value = result[section][key]

        return f"{value:.{decimals}f}"

    print(
        f"{'Successful runs':<32}"
        f"{(str(cpu['successful_runs']) if cpu else 'N/A'):>20}"
        f"{(str(gpu['successful_runs']) if gpu else 'N/A'):>20}"
    )

    print(
        f"{'Timeouts':<32}"
        f"{(str(cpu['timeout_count']) if cpu else 'N/A'):>20}"
        f"{(str(gpu['timeout_count']) if gpu else 'N/A'):>20}"
    )

    print(
        f"{'Mean latency (ms)':<32}"
        f"{fmt(cpu, 'overall', 'mean_ms'):>20}"
        f"{fmt(gpu, 'overall', 'mean_ms'):>20}"
    )

    print(
        f"{'P50 latency (ms)':<32}"
        f"{fmt(cpu, 'overall', 'p50_ms'):>20}"
        f"{fmt(gpu, 'overall', 'p50_ms'):>20}"
    )

    print(
        f"{'P95 latency (ms)':<32}"
        f"{fmt(cpu, 'overall', 'p95_ms'):>20}"
        f"{fmt(gpu, 'overall', 'p95_ms'):>20}"
    )

    print(
        f"{'Sustained FPS':<32}"
        f"{fmt(cpu, 'overall', 'fps', 2):>20}"
        f"{fmt(gpu, 'overall', 'fps', 2):>20}"
    )

    def get_peak_rss(result: dict | None) -> str:
        if result is None:
            return "N/A"
        value = result["memory"]["peak_rss_observed_mb"]
        return f"{value:.1f}"

    print(
        f"{'Peak host RSS (MB)':<32}"
        f"{get_peak_rss(cpu):>20}"
        f"{get_peak_rss(gpu):>20}"
    )

    cpu_drift = (
        f"{cpu['latency_drift_percent']:+.2f}%"
        if cpu
        and cpu["latency_drift_percent"]
        is not None
        else "N/A"
    )

    gpu_drift = (
        f"{gpu['latency_drift_percent']:+.2f}%"
        if gpu
        and gpu["latency_drift_percent"]
        is not None
        else "N/A"
    )

    print(
        f"{'Latency drift first→last':<32}"
        f"{cpu_drift:>20}"
        f"{gpu_drift:>20}"
    )

    print("=" * 105)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment":
            "Foundry M3 — Robust 10-minute sustained INT8 benchmark",

        "model":
            str(MODEL_XML),

        "input":
            str(INPUT_IMAGE),

        "duration_seconds":
            DURATION_SECONDS,

        "warmup_runs":
            WARMUP_RUNS,

        "cpu_threads":
            CPU_THREADS,

        "inference_timeout_ms":
            INFERENCE_TIMEOUT_MS,

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
        "Sustained benchmark complete."
    )


if __name__ == "__main__":
    main()