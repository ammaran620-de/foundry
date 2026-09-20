"""
Foundry M3 — Standardized Backend Comparison

Compares PyTorch, ONNX Runtime, and OpenVINO using the same
640x640 input, batch size, warm-up count, and timed runs.

This benchmark measures backend inference only.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov
import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYTORCH_MODEL = PROJECT_ROOT / "models" / "yolo26n.pt"
ONNX_MODEL = PROJECT_ROOT / "models" / "yolo26n.onnx"
OPENVINO_MODEL = (
    PROJECT_ROOT
    / "models"
    / "yolo26n_openvino_model"
    / "yolo26n.xml"
)

INPUT_PATH = PROJECT_ROOT / "benchmark" / "input" / "benchmark_640.png"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "backend_comparison.json"
)

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 5
TIMED_RUNS = 30
THREADS = 10


def load_input() -> np.ndarray:
    """Load and preprocess the fixed benchmark image."""
    image = Image.open(INPUT_PATH).convert("RGB")

    array = np.asarray(image, dtype=np.float32)
    array /= 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array)


def summarize(latencies: list[float]) -> dict:
    """Calculate benchmark statistics."""
    mean = statistics.mean(latencies)

    return {
        "mean_latency_ms": round(mean, 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "min_latency_ms": round(min(latencies), 4),
        "max_latency_ms": round(max(latencies), 4),
        "std_latency_ms": round(
            statistics.stdev(latencies)
            if len(latencies) > 1
            else 0.0,
            4,
        ),
        "fps": round(1000.0 / mean, 4),
    }


def benchmark_pytorch(input_tensor: np.ndarray) -> dict:
    """Benchmark PyTorch model forward pass only."""

    print("\n[PyTorch]")
    torch.set_num_threads(THREADS)

    model = YOLO(str(PYTORCH_MODEL))
    pytorch_model = model.model
    pytorch_model.eval()

    tensor = torch.from_numpy(input_tensor)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            pytorch_model(tensor)

        latencies = []

        for index in range(TIMED_RUNS):
            start = time.perf_counter()

            pytorch_model(tensor)

            end = time.perf_counter()
            latency = (end - start) * 1000.0
            latencies.append(latency)

            print(f"  Run {index + 1:02d}: {latency:.2f} ms")

    return {
        "runtime": "PyTorch",
        "configuration": {
            "device": "CPU",
            "threads": THREADS,
        },
        "metrics": summarize(latencies),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def benchmark_onnxruntime(input_tensor: np.ndarray) -> dict:
    """Benchmark ONNX Runtime inference only."""

    print("\n[ONNX Runtime]")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = THREADS
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: input_tensor})

    latencies = []

    for index in range(TIMED_RUNS):
        start = time.perf_counter()

        session.run(None, {input_name: input_tensor})

        end = time.perf_counter()
        latency = (end - start) * 1000.0
        latencies.append(latency)

        print(f"  Run {index + 1:02d}: {latency:.2f} ms")

    return {
        "runtime": "ONNX Runtime",
        "configuration": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": THREADS,
            "inter_op_threads": 1,
            "graph_optimization": "ORT_ENABLE_ALL",
        },
        "metrics": summarize(latencies),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def benchmark_openvino(input_tensor: np.ndarray) -> dict:
    """Benchmark OpenVINO inference only."""

    print("\n[OpenVINO]")

    core = ov.Core()

    model = core.read_model(str(OPENVINO_MODEL))

    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": THREADS,
        },
    )

    infer_request = compiled_model.create_infer_request()
    input_port = compiled_model.input(0)

    for _ in range(WARMUP_RUNS):
        infer_request.infer({input_port: input_tensor})

    latencies = []

    for index in range(TIMED_RUNS):
        start = time.perf_counter()

        infer_request.infer({input_port: input_tensor})

        end = time.perf_counter()
        latency = (end - start) * 1000.0
        latencies.append(latency)

        print(f"  Run {index + 1:02d}: {latency:.2f} ms")

    return {
        "runtime": "OpenVINO",
        "configuration": {
            "device": "CPU",
            "inference_num_threads": THREADS,
        },
        "metrics": summarize(latencies),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def main() -> None:
    """Run the standardized backend comparison."""

    required_files = [
        PYTORCH_MODEL,
        ONNX_MODEL,
        OPENVINO_MODEL,
        INPUT_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Foundry M3 — Standardized Backend Comparison")
    print("=" * 50)
    print(f"Image:       {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Threads:     {THREADS}")
    print(f"Warm-up:     {WARMUP_RUNS}")
    print(f"Timed runs:  {TIMED_RUNS}")

    input_tensor = load_input()

    pytorch_result = benchmark_pytorch(input_tensor)
    onnx_result = benchmark_onnxruntime(input_tensor)
    openvino_result = benchmark_openvino(input_tensor)

    results = {
        "benchmark": "m3-standardized-backend-comparison",
        "version": "0.1.0",
        "input": {
            "file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "dtype": "float32",
            "layout": "NCHW",
        },
        "configuration": {
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "threads": THREADS,
        },
        "backends": [
            pytorch_result,
            onnx_result,
            openvino_result,
        ],
        "notes": [
            "All backends receive the same preprocessed float32 tensor.",
            "Only backend inference is timed.",
            "Input loading and preprocessing are excluded.",
            "Model loading and compilation are excluded.",
            "Warm-up runs are excluded from reported metrics.",
            "No cherry-picking of individual runs.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 50)
    print("STANDARDIZED RESULTS")
    print("=" * 50)

    for backend in results["backends"]:
        metrics = backend["metrics"]

        print(
            f"{backend['runtime']:16s} "
            f"{metrics['mean_latency_ms']:8.2f} ms | "
            f"{metrics['fps']:6.2f} FPS"
        )

    print("\nResult saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()