"""
Foundry M3 — Generate Benchmark Report

Reads the standardized backend comparison JSON and generates
a reproducible Markdown engineering report.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "backend_comparison.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "backend_comparison.md"
)


def load_results() -> dict:
    """Load standardized benchmark results."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_report(data: dict) -> str:
    """Generate Markdown report from benchmark JSON."""

    backends = data["backends"]
    config = data["configuration"]
    input_info = data["input"]
    model_info = data.get("model", {})

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    lines.append("# Foundry M3 — Standardized Backend Benchmark")
    lines.append("")

    lines.append(
        "This report summarizes the standardized CPU inference "
        "benchmark for YOLO26n across PyTorch, ONNX Runtime, "
        "and OpenVINO."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    lines.append("## Model")
    lines.append("")

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")

    lines.append(
        f"| Model | `{model_info.get('name', 'YOLO26n')}` |"
    )

    if model_info.get("pytorch_checkpoint"):
        lines.append(
            f"| PyTorch checkpoint | "
            f"`{model_info['pytorch_checkpoint']}` |"
        )

    if model_info.get("onnx_model"):
        lines.append(
            f"| ONNX model | "
            f"`{model_info['onnx_model']}` |"
        )

    if model_info.get("openvino_model"):
        lines.append(
            f"| OpenVINO model | "
            f"`{model_info['openvino_model']}` |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Benchmark Configuration
    # ------------------------------------------------------------------

    lines.append("## Benchmark Configuration")
    lines.append("")

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")

    lines.append(
        f"| Input | `{input_info['width']}x{input_info['height']}` |"
    )

    lines.append(
        f"| Batch size | `{input_info['batch_size']}` |"
    )

    lines.append(
        f"| Data type | `{input_info['dtype']}` |"
    )

    lines.append(
        f"| Layout | `{input_info['layout']}` |"
    )

    lines.append(
        f"| Warm-up runs | `{config['warmup_runs']}` |"
    )

    lines.append(
        f"| Timed runs | `{config['timed_runs']}` |"
    )

    lines.append(
        f"| Threads | `{config['threads']}` |"
    )

    lines.append("")

    # ------------------------------------------------------------------
    # NMS Configuration
    # ------------------------------------------------------------------

    nms_config = config.get("nms")

    if nms_config:
        lines.append("## NMS Configuration")
        lines.append("")

        lines.append("| Parameter | Value |")
        lines.append("|---|---|")

        lines.append(
            f"| Confidence threshold | "
            f"`{nms_config.get('confidence_threshold', 'N/A')}` |"
        )

        lines.append(
            f"| IoU threshold | "
            f"`{nms_config.get('iou_threshold', 'N/A')}` |"
        )

        lines.append(
            f"| Maximum detections | "
            f"`{nms_config.get('max_detections', 'N/A')}` |"
        )

        lines.append(
            f"| PyTorch NMS | "
            f"`{nms_config.get('pytorch', 'N/A')}` |"
        )

        lines.append(
            f"| ONNX Runtime NMS | "
            f"`{nms_config.get('onnxruntime', 'N/A')}` |"
        )

        lines.append(
            f"| OpenVINO NMS | "
            f"`{nms_config.get('openvino', 'N/A')}` |"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Measured Results
    # ------------------------------------------------------------------

    lines.append("## Measured Results")
    lines.append("")

    lines.append(
        "| Runtime | Mean (ms) | Median / p50 (ms) | "
        "p95 (ms) | Min (ms) | Max (ms) | Std (ms) | FPS |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for backend in backends:
        metrics = backend["metrics"]

        p95 = metrics.get(
            "p95_latency_ms",
            float("nan"),
        )

        lines.append(
            f"| {backend['runtime']} "
            f"| {metrics['mean_latency_ms']:.2f} "
            f"| {metrics['median_latency_ms']:.2f} "
            f"| {p95:.2f} "
            f"| {metrics['min_latency_ms']:.2f} "
            f"| {metrics['max_latency_ms']:.2f} "
            f"| {metrics['std_latency_ms']:.2f} "
            f"| {metrics['fps']:.2f} |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Relative Latency
    # ------------------------------------------------------------------

    lines.append("## Relative Latency")
    lines.append("")

    baseline = backends[0]
    baseline_mean = baseline["metrics"]["mean_latency_ms"]

    lines.append(
        "PyTorch is used as the reference runtime for this "
        "relative comparison."
    )

    lines.append("")

    lines.append(
        "| Runtime | Mean latency vs PyTorch | "
        "Median latency vs PyTorch |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for backend in backends:
        mean = backend["metrics"]["mean_latency_ms"]
        median = backend["metrics"]["median_latency_ms"]

        mean_ratio = mean / baseline_mean

        baseline_median = baseline["metrics"]["median_latency_ms"]
        median_ratio = median / baseline_median

        lines.append(
            f"| {backend['runtime']} "
            f"| `{mean_ratio:.2f}x` "
            f"| `{median_ratio:.2f}x` |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    lines.append("## Throughput")
    lines.append("")

    lines.append("| Runtime | Throughput |")
    lines.append("|---|---:|")

    for backend in backends:
        fps = backend["metrics"]["fps"]

        lines.append(
            f"| {backend['runtime']} | `{fps:.2f} FPS` |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Tail Latency
    # ------------------------------------------------------------------

    lines.append("## Tail Latency")
    lines.append("")

    lines.append(
        "p95 represents the latency below which approximately "
        "95% of the recorded inference runs fall."
    )

    lines.append("")

    lines.append("| Runtime | p95 latency |")
    lines.append("|---|---:|")

    for backend in backends:
        p95 = backend["metrics"].get(
            "p95_latency_ms",
            float("nan"),
        )

        lines.append(
            f"| {backend['runtime']} | `{p95:.2f} ms` |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Measurement Method
    # ------------------------------------------------------------------

    lines.append("## Measurement Method")
    lines.append("")

    lines.append(
        "- All backends receive the same fixed 640x640 RGB input."
    )

    lines.append(
        "- Input is converted to float32 NCHW format before timing."
    )

    lines.append(
        "- Input preprocessing is excluded from the timed section."
    )

    lines.append(
        "- Model loading and compilation are excluded."
    )

    lines.append(
        "- Five warm-up runs are performed before measurement."
    )

    lines.append(
        "- Thirty timed inference runs are collected per backend."
    )

    lines.append(
        "- PyTorch times the raw model forward pass followed by "
        "external NMS."
    )

    lines.append(
        "- ONNX Runtime uses embedded NonMaxSuppression in the "
        "exported ONNX graph."
    )

    lines.append(
        "- OpenVINO uses embedded NonMaxSuppression in the "
        "exported OpenVINO graph."
    )

    lines.append(
        "- Mean, median, p95, minimum, maximum, standard deviation, "
        "and FPS are reported."
    )

    lines.append(
        "- No individual runs are removed from the reported results."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Runtime Configuration
    # ------------------------------------------------------------------

    lines.append("## Runtime Configuration")
    lines.append("")

    for backend in backends:
        lines.append(f"### {backend['runtime']}")
        lines.append("")

        for key, value in backend["configuration"].items():
            lines.append(
                f"- **{key}:** `{value}`"
            )

        lines.append("")

    # ------------------------------------------------------------------
    # Recorded Latencies
    # ------------------------------------------------------------------

    lines.append("## Recorded Latencies")
    lines.append("")

    lines.append(
        "All timed observations are retained in the benchmark JSON "
        "to preserve visibility into variance and outliers."
    )

    lines.append("")

    for backend in backends:
        lines.append(f"### {backend['runtime']}")
        lines.append("")

        latencies = backend.get("latencies_ms", [])

        if latencies:
            formatted = ", ".join(
                f"{value:.2f}" for value in latencies
            )

            lines.append(
                f"`{formatted}`"
            )
        else:
            lines.append(
                "No raw latency observations were recorded."
            )

        lines.append("")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    lines.append("## Interpretation")
    lines.append("")

    lines.append(
        "The measurements show how the same YOLO26n detection model "
        "behaves under three CPU inference backends on the target "
        "machine."
    )

    lines.append("")

    lines.append(
        "The comparison uses a common detection path: PyTorch "
        "performs external NMS, while the exported ONNX and OpenVINO "
        "artifacts contain embedded NMS."
    )

    lines.append("")

    lines.append(
        "The measurements are specific to the recorded hardware, "
        "software versions, thread configuration, input resolution, "
        "and benchmark protocol. They should not be treated as "
        "universal performance characteristics."
    )

    lines.append("")

    lines.append(
        "The results establish the CPU reference point for subsequent "
        "M3 optimization stages."
    )

    lines.append("")

    lines.append(
        "Future stages, including INT8 CPU, Intel Iris Xe GPU, and "
        "browser deployment, should be evaluated against this "
        "reference using the same evaluation contract."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------

    lines.append("## Current Limitations")
    lines.append("")

    lines.append(
        "- This report contains performance measurements only; "
        "accuracy/mAP has not yet been evaluated on a ground-truth "
        "dataset."
    )

    lines.append(
        "- Peak RAM usage is not yet captured by this benchmark."
    )

    lines.append(
        "- Sustained 10-minute throughput has not yet been measured."
    )

    lines.append(
        "- Preprocessing and postprocessing have not yet been "
        "decomposed into individual pipeline stages."
    )

    lines.append(
        "- INT8 calibration and accuracy evaluation are not yet final."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    lines.append("## Reproducibility")
    lines.append("")

    lines.append(
        "The benchmark is generated from version-controlled code "
        "and the recorded JSON result artifact."
    )

    lines.append("")

    lines.append(
        "Benchmark configuration, runtime configuration, raw latency "
        "observations, and derived statistics are preserved in the "
        "generated result."
    )

    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate the Markdown benchmark report."""

    data = load_results()
    report = generate_report(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print("Benchmark report generated:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()