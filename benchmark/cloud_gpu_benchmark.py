"""
Foundry M3 — Cloud GPU Reference Benchmark

Purpose:
    Measure the same YOLO26n inference workload on a cloud GPU so the
    M3 deployment study can calculate an evidence-based production cost.

Important:
    This script measures inference performance only.
    Cloud hourly pricing is intentionally NOT hard-coded here.

Environment expected:
    Python 3.11+
    ultralytics
    torch

Usage example:
    python benchmark/cloud_gpu_benchmark.py

The benchmark records:
    - runtime
    - GPU/device
    - model
    - input shape
    - warm-up runs
    - timed runs
    - mean latency
    - p50 latency
    - p95 latency
    - FPS

The output is written to:
    benchmark/results/cloud_gpu_reference.json
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"
INPUT_PATH = PROJECT_ROOT / "benchmark" / "input" / "benchmark_640.png"
OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "results" / "cloud_gpu_reference.json"

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 20
TIMED_RUNS = 100


def percentile(values: list[float], q: float) -> float:
    """Compute a linear-interpolated percentile."""
    if not values:
        raise ValueError("Cannot compute percentile of empty data.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered))
    fraction = index - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def require_files() -> None:
    """Validate benchmark inputs."""
    for path in (MODEL_PATH, INPUT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required benchmark input not found: {path}"
            )


def load_model() -> YOLO:
    """Load the canonical M3 YOLO26n checkpoint."""
    return YOLO(str(MODEL_PATH))


def benchmark(model: YOLO) -> list[float]:
    """Run warm-up and timed inference."""
    image = str(INPUT_PATH)

    device = 0

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            verbose=False,
        )

    latencies_ms: list[float] = []

    for run_index in range(TIMED_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        latencies_ms.append(latency_ms)

        print(
            f"Run {run_index + 1:03d}/{TIMED_RUNS}: "
            f"{latency_ms:.3f} ms"
        )

    return latencies_ms


def main() -> None:
    """Run the cloud GPU reference benchmark."""
    require_files()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run this benchmark on the intended cloud GPU instance."
        )

    print("=" * 70)
    print("Foundry M3 — Cloud GPU Reference Benchmark")
    print("=" * 70)

    print(f"Model:       {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Input:       {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Image size:  {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Warm-up:     {WARMUP_RUNS}")
    print(f"Timed runs:  {TIMED_RUNS}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print()

    model = load_model()
    latencies = benchmark(model)

    mean_ms = statistics.mean(latencies)
    p50_ms = percentile(latencies, 0.50)
    p95_ms = percentile(latencies, 0.95)

    result = {
        "experiment": "Foundry M3 — Cloud GPU Reference Benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "model": {
            "name": "YOLO26n",
            "path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "input": {
            "file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
        },
        "configuration": {
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "device": "cuda:0",
        },
        "metrics": {
            "runs": TIMED_RUNS,
            "mean_ms": mean_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "fps": 1000.0 / mean_ms,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": statistics.stdev(latencies)
            if len(latencies) > 1
            else 0.0,
        },
        "latencies_ms": [round(x, 4) for x in latencies],
        "notes": [
            "This is a performance reference measurement.",
            "Cloud hourly pricing is intentionally not embedded.",
            "Cost calculations must use the exact instance pricing "
            "and region actually selected for the experiment.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Mean:       {mean_ms:.3f} ms")
    print(f"P50:        {p50_ms:.3f} ms")
    print(f"P95:        {p95_ms:.3f} ms")
    print(f"FPS:        {1000.0 / mean_ms:.3f}")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Saved:      {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()