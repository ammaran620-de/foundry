"""
Foundry M3 — Standardized Backend Comparison

Compares PyTorch, ONNX Runtime, and OpenVINO using the same
640x640 input, batch size, warm-up count, and timed runs.

The benchmark measures a comparable detection inference path:
- PyTorch uses external NMS after the raw model forward pass.
- ONNX Runtime uses the embedded NMS in the exported ONNX graph.
- OpenVINO uses the embedded NMS in the exported OpenVINO graph.

This keeps the output-processing boundary consistent across backends.
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
from ultralytics.utils.nms import non_max_suppression


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

NMS_CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.70
NMS_MAX_DETECTIONS = 300


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
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
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
    """Benchmark PyTorch inference with external NMS."""

    print("\n[PyTorch]")
    torch.set_num_threads(THREADS)

    model = YOLO(str(PYTORCH_MODEL))
    pytorch_model = model.model
    pytorch_model.eval()

    tensor = torch.from_numpy(input_tensor)

    def infer():
        raw = pytorch_model(tensor)[0]

        return non_max_suppression(
            raw,
            conf_thres=NMS_CONF_THRESHOLD,
            iou_thres=NMS_IOU_THRESHOLD,
            max_det=NMS_MAX_DETECTIONS,
        )

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            infer()

        latencies = []

        for index in range(TIMED_RUNS):
            start = time.perf_counter()

            infer()

            end = time.perf_counter()
            latency = (end - start) * 1000.0
            latencies.append(latency)

            print(f"  Run {index + 1:02d}: {latency:.2f} ms")

    return {
        "runtime": "PyTorch",
        "configuration": {
            "device": "CPU",
            "threads": THREADS,
            "nms": "external",
            "confidence_threshold": NMS_CONF_THRESHOLD,
            "iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": NMS_MAX_DETECTIONS,
        },
        "metrics": summarize(latencies),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def benchmark_onnxruntime(input_tensor: np.ndarray) -> dict:
    """Benchmark ONNX Runtime inference with embedded NMS."""

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
            "nms": "embedded",
        },
        "metrics": summarize(latencies),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def benchmark_openvino(input_tensor: np.ndarray) -> dict:
    """Benchmark OpenVINO inference with embedded NMS."""

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
            "nms": "embedded",
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
        "version": "0.2.0",
        "model": {
            "name": "YOLO26n",
            "pytorch_checkpoint": str(
                PYTORCH_MODEL.relative_to(PROJECT_ROOT)
            ),
            "onnx_model": str(
                ONNX_MODEL.relative_to(PROJECT_ROOT)
            ),
            "openvino_model": str(
                OPENVINO_MODEL.relative_to(PROJECT_ROOT)
            ),
        },
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
            "nms": {
                "confidence_threshold": NMS_CONF_THRESHOLD,
                "iou_threshold": NMS_IOU_THRESHOLD,
                "max_detections": NMS_MAX_DETECTIONS,
                "pytorch": "external",
                "onnxruntime": "embedded",
                "openvino": "embedded",
            },
        },
        "backends": [
            pytorch_result,
            onnx_result,
            openvino_result,
        ],
        "notes": [
            "All backends receive the same preprocessed float32 tensor.",
            "The comparable detection path includes NMS for every backend.",
            "PyTorch performs external NMS after the raw model forward pass.",
            "ONNX Runtime uses NonMaxSuppression embedded in the ONNX graph.",
            "OpenVINO uses NonMaxSuppression embedded in the OpenVINO graph.",
            "Only backend inference and the required NMS path are timed.",
            "Input loading and preprocessing are excluded.",
            "Model loading and compilation are excluded.",
            "Warm-up runs are excluded from reported metrics.",
            "No cherry-picking of individual runs.",
            "All timed observations remain in the recorded latency arrays.",
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
            f"p95 {metrics['p95_latency_ms']:8.2f} ms | "
            f"{metrics['fps']:6.2f} FPS"
        )

    print("\nResult saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()