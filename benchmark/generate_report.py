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

    lines: list[str] = []

    lines.append("# Foundry M3 — Standardized Backend Benchmark")
    lines.append("")
    lines.append(
        "This report summarizes the standardized CPU inference "
        "benchmark for YOLO26n across PyTorch, ONNX Runtime, "
        "and OpenVINO."
    )
    lines.append("")

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

    lines.append("## Measured Results")
    lines.append("")
    lines.append(
        "| Runtime | Mean (ms) | Median (ms) | "
        "Min (ms) | Max (ms) | Std (ms) | FPS |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for backend in backends:
        metrics = backend["metrics"]

        lines.append(
            f"| {backend['runtime']} "
            f"| {metrics['mean_latency_ms']:.2f} "
            f"| {metrics['median_latency_ms']:.2f} "
            f"| {metrics['min_latency_ms']:.2f} "
            f"| {metrics['max_latency_ms']:.2f} "
            f"| {metrics['std_latency_ms']:.2f} "
            f"| {metrics['fps']:.2f} |"
        )

    lines.append("")

    lines.append("## Relative Latency")
    lines.append("")

    baseline = backends[0]
    baseline_mean = baseline["metrics"]["mean_latency_ms"]

    lines.append(
        f"PyTorch is used as the reference runtime for this "
        f"relative comparison."
    )
    lines.append("")

    lines.append("| Runtime | Mean latency vs PyTorch |")
    lines.append("|---|---:|")

    for backend in backends:
        mean = backend["metrics"]["mean_latency_ms"]
        ratio = mean / baseline_mean

        lines.append(
            f"| {backend['runtime']} | `{ratio:.2f}x` |"
        )

    lines.append("")

    lines.append("## Throughput")
    lines.append("")

    lines.append(
        "| Runtime | Throughput |"
    )
    lines.append("|---|---:|")

    for backend in backends:
        fps = backend["metrics"]["fps"]

        lines.append(
            f"| {backend['runtime']} | `{fps:.2f} FPS` |"
        )

    lines.append("")

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
        "- Mean, median, minimum, maximum, and standard deviation "
        "are reported."
    )
    lines.append(
        "- No individual runs are removed from the reported results."
    )
    lines.append("")

    lines.append("## Runtime Configuration")
    lines.append("")

    for backend in backends:
        lines.append(f"### {backend['runtime']}")
        lines.append("")

        for key, value in backend["configuration"].items():
            lines.append(f"- **{key}:** `{value}`")

        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The measurements show how the same model behaves under "
        "three CPU inference backends on the target machine."
    )
    lines.append("")
    lines.append(
        "The results are measurements from this specific hardware "
        "and software configuration. They should not be treated "
        "as universal performance characteristics."
    )
    lines.append("")
    lines.append(
        "Further optimization stages, including quantization, "
        "should be evaluated against these measurements using "
        "the same benchmark methodology."
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