"""
Foundry M3 — ONNX Runtime CPU Benchmark

Measures YOLO26n ONNX inference latency using ONNX Runtime
with the CPU execution provider and a fixed 640x640 input.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.onnx"
INPUT_PATH = PROJECT_ROOT / "benchmark" / "input" / "benchmark_640.png"

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "onnxruntime_baseline.json"


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 5
TIMED_RUNS = 30

EXECUTION_PROVIDER = "CPUExecutionProvider"
INTRA_OP_THREADS = 10
INTER_OP_THREADS = 1


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from src.system_info import get_system_info


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert RGB PIL image into ONNX NCHW float32 input."""

    image_array = np.asarray(image, dtype=np.float32)

    image_array /= 255.0

    image_array = np.transpose(image_array, (2, 0, 1))

    image_array = np.expand_dims(image_array, axis=0)

    return np.ascontiguousarray(image_array)


def benchmark() -> dict:
    """Run the ONNX Runtime CPU benchmark."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark input not found: {INPUT_PATH}"
        )

    available_providers = ort.get_available_providers()

    if EXECUTION_PROVIDER not in available_providers:
        raise RuntimeError(
            f"{EXECUTION_PROVIDER} is not available. "
            f"Available providers: {available_providers}"
        )

    # -----------------------------------------------------------------------
    # ONNX Runtime session configuration
    # -----------------------------------------------------------------------

    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = INTRA_OP_THREADS
    session_options.inter_op_num_threads = INTER_OP_THREADS

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    print("Loading ONNX Runtime session...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=[EXECUTION_PROVIDER],
    )

    actual_provider = session.get_providers()

    print("Loading benchmark input...")

    image = Image.open(INPUT_PATH).convert("RGB")
    input_tensor = preprocess(image)

    input_name = session.get_inputs()[0].name

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    print()
    print("Benchmark configuration:")
    print(f"  Model:         {MODEL_PATH.name}")
    print(f"  Input:         {INPUT_PATH.name}")
    print(f"  Image size:    {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Provider:      {EXECUTION_PROVIDER}")
    print(f"  Intra-op:      {INTRA_OP_THREADS}")
    print(f"  Inter-op:      {INTER_OP_THREADS}")
    print(f"  Warm-up:       {WARMUP_RUNS}")
    print(f"  Timed runs:    {TIMED_RUNS}")
    print()

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------

    print("Running warm-up...")

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {
                input_name: input_tensor,
            },
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

        session.run(
            None,
            {
                input_name: input_tensor,
            },
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
        "benchmark": "m3-onnxruntime-baseline",
        "version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "runtime": {
            "framework": "onnxruntime",
            "onnxruntime_version": ort.__version__,
            "execution_provider": EXECUTION_PROVIDER,
            "session_providers": actual_provider,
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
            "provider": EXECUTION_PROVIDER,
            "intra_op_threads": INTRA_OP_THREADS,
            "inter_op_threads": INTER_OP_THREADS,
            "graph_optimization": "ORT_ENABLE_ALL",
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
            "ONNX Runtime CPU baseline.",
            "CPUExecutionProvider explicitly selected.",
            "Input preprocessing performed once before timing.",
            "Timed section contains session.run() only.",
            "Warm-up runs excluded from reported metrics.",
            "Session creation time excluded from timed inference.",
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