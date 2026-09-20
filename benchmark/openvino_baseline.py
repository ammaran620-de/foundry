"""
Foundry M3 — OpenVINO CPU Benchmark

Measures YOLO26n OpenVINO inference latency on the CPU using
a fixed 640x640 benchmark image.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_openvino_model"
MODEL_PATH = MODEL_DIR / "yolo26n.xml"

INPUT_PATH = PROJECT_ROOT / "benchmark" / "input" / "benchmark_640.png"

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "openvino_baseline.json"


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 5
TIMED_RUNS = 30

DEVICE = "CPU"
INFERENCE_NUM_THREADS = 10


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from src.system_info import get_system_info


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert RGB PIL image into NCHW float32 input."""

    image_array = np.asarray(image, dtype=np.float32)

    image_array /= 255.0

    image_array = np.transpose(image_array, (2, 0, 1))

    image_array = np.expand_dims(image_array, axis=0)

    return np.ascontiguousarray(image_array)


def benchmark() -> dict:
    """Run the OpenVINO CPU benchmark."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"OpenVINO model not found: {MODEL_PATH}"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark input not found: {INPUT_PATH}"
        )

    # -----------------------------------------------------------------------
    # OpenVINO runtime
    # -----------------------------------------------------------------------

    core = ov.Core()

    available_devices = core.available_devices

    if DEVICE not in available_devices:
        raise RuntimeError(
            f"{DEVICE} device is not available. "
            f"Available devices: {available_devices}"
        )

    print("Loading OpenVINO model...")

    model = core.read_model(str(MODEL_PATH))

    # Explicit CPU thread configuration.
    compiled_model = core.compile_model(
        model,
        DEVICE,
        {
            "INFERENCE_NUM_THREADS": INFERENCE_NUM_THREADS,
        },
    )

    infer_request = compiled_model.create_infer_request()

    input_port = compiled_model.input(0)
    output_port = compiled_model.output(0)

    print("Loading benchmark input...")

    image = Image.open(INPUT_PATH).convert("RGB")

    input_tensor = preprocess(image)

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    print()
    print("Benchmark configuration:")
    print(f"  Model:         {MODEL_PATH.name}")
    print(f"  Input:         {INPUT_PATH.name}")
    print(f"  Image size:    {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Device:        {DEVICE}")
    print(f"  Inference threads: {INFERENCE_NUM_THREADS}")
    print(f"  Warm-up:       {WARMUP_RUNS}")
    print(f"  Timed runs:    {TIMED_RUNS}")
    print()

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------

    print("Running warm-up...")

    for _ in range(WARMUP_RUNS):
        infer_request.infer(
            {
                input_port: input_tensor,
            }
        )

    print("Warm-up complete.")
    print()

    # -----------------------------------------------------------------------
    # Timed inference
    # -----------------------------------------------------------------------

    print("Running timed inference...")

    latencies_ms: list[float] = []

    for run_index in range(TIMED_RUNS):
        start = time.perf_counter()

        infer_request.infer(
            {
                input_port: input_tensor,
            }
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0

        latencies_ms.append(latency_ms)

        print(
            f"  Run {run_index + 1:02d}/{TIMED_RUNS}: "
            f"{latency_ms:.2f} ms"
        )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    mean_latency = statistics.mean(latencies_ms)
    median_latency = statistics.median(latencies_ms)
    min_latency = min(latencies_ms)
    max_latency = max(latencies_ms)

    std_latency = (
        statistics.stdev(latencies_ms)
        if len(latencies_ms) > 1
        else 0.0
    )

    fps = 1000.0 / mean_latency

    # -----------------------------------------------------------------------
    # System information
    # -----------------------------------------------------------------------

    system_info = get_system_info()

    # -----------------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------------

    result = {
        "benchmark": "m3-openvino-baseline",
        "version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "runtime": {
            "framework": "openvino",
            "openvino_version": ov.__version__,
            "device": DEVICE,
        },

        "model": {
            "name": MODEL_PATH.name,
            "path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        },

        "input": {
            "file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "format": "RGB",
            "dtype": "float32",
            "layout": "NCHW",
        },

        "configuration": {
            "device": DEVICE,
            "inference_num_threads": INFERENCE_NUM_THREADS,
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
        },

        "metrics": {
            "mean_latency_ms": round(mean_latency, 4),
            "median_latency_ms": round(median_latency, 4),
            "min_latency_ms": round(min_latency, 4),
            "max_latency_ms": round(max_latency, 4),
            "std_latency_ms": round(std_latency, 4),
            "fps": round(fps, 4),
        },

        "latencies_ms": [
            round(value, 4)
            for value in latencies_ms
        ],

        "system": system_info,

        "notes": [
            "OpenVINO CPU baseline.",
            "CPU device explicitly selected.",
            "OpenVINO inference thread count explicitly configured.",
            "Input preprocessing performed once before timing.",
            "Timed section contains inference only.",
            "Warm-up runs excluded from reported metrics.",
            "Model compilation time excluded from timed inference.",
            "Fixed deterministic 640x640 RGB benchmark input.",
        ],
    }

    # -----------------------------------------------------------------------
    # Save result
    # -----------------------------------------------------------------------

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    print()
    print("Benchmark complete.")
    print()

    print(f"Mean latency:   {mean_latency:.2f} ms")
    print(f"Median latency: {median_latency:.2f} ms")
    print(f"Min latency:    {min_latency:.2f} ms")
    print(f"Max latency:    {max_latency:.2f} ms")
    print(f"Std deviation:  {std_latency:.2f} ms")
    print(f"Throughput:     {fps:.2f} FPS")
    print()

    print("Result saved to:")
    print(OUTPUT_PATH)

    return result


if __name__ == "__main__":
    benchmark()