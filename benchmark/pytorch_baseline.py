"""
Foundry M3 — PyTorch CPU Baseline Benchmark

Measures YOLO26n inference latency on the target CPU using a fixed
640x640 benchmark image.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"
INPUT_PATH = PROJECT_ROOT / "benchmark" / "input" / "benchmark_640.png"

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "pytorch_baseline.json"


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 5
TIMED_RUNS = 30

DEVICE = "cpu"
TORCH_THREADS = 10


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from src.system_info import get_system_info


def benchmark() -> dict:
    """Run the PyTorch CPU benchmark."""

    # Explicitly set and capture the actual PyTorch thread count.
    torch.set_num_threads(TORCH_THREADS)
    actual_torch_threads = torch.get_num_threads()

    if actual_torch_threads != TORCH_THREADS:
        raise RuntimeError(
            f"PyTorch thread configuration mismatch: "
            f"requested={TORCH_THREADS}, "
            f"actual={actual_torch_threads}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark input not found: {INPUT_PATH}"
        )

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    print("Loading benchmark input...")
    image = Image.open(INPUT_PATH).convert("RGB")

    print()
    print("Benchmark configuration:")
    print(f"  Model:        {MODEL_PATH.name}")
    print(f"  Input:        {INPUT_PATH.name}")
    print(f"  Image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Device:       {DEVICE}")
    print(f"  Torch threads:{actual_torch_threads}")
    print(f"  Warm-up:      {WARMUP_RUNS}")
    print(f"  Timed runs:   {TIMED_RUNS}")
    print()

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------

    print("Running warm-up...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            verbose=False,
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

        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            verbose=False,
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
    # System metadata
    # -----------------------------------------------------------------------

    system_info = get_system_info()

    # Force the benchmark's actual thread count into the recorded metadata.
    system_info["torch_threads"] = actual_torch_threads

    # -----------------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------------

    result = {
        "benchmark": "m3-pytorch-baseline",
        "version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "runtime": {
            "framework": "ultralytics-pytorch",
            "ultralytics_version": __import__("ultralytics").__version__,
            "pytorch_version": torch.__version__,
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
        },

        "configuration": {
            "device": DEVICE,
            "torch_threads": actual_torch_threads,
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
            "CPU-only PyTorch baseline.",
            "PyTorch thread count explicitly locked to 10.",
            "Actual PyTorch thread count recorded after configuration.",
            "Warm-up runs excluded from reported metrics.",
            "Model loading time excluded from timed inference.",
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
    print(f"Torch threads:  {actual_torch_threads}")
    print()

    print("Result saved to:")
    print(OUTPUT_PATH)

    return result


if __name__ == "__main__":
    benchmark()