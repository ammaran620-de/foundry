"""
Foundry M3 — Production Cost Analysis

Purpose:
    Convert a measured GPU throughput result and an actual cloud
    hourly price into reproducible production-cost estimates.

Calculations:
    1. Cost per 1,000 single-image inferences
    2. Cost to process one hour of video

Important:
    The cloud benchmark measures inference throughput.
    This calculator does NOT invent or hard-code a cloud price.

Example:

    python benchmark/cost_analysis.py `
        --benchmark benchmark/results/cloud_gpu_reference.json `
        --hourly-price 0.526 `
        --video-fps 30

The resulting report is written to:

    benchmark/results/cloud_cost_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "cloud_cost_analysis.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate production cost from measured cloud GPU throughput."
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to cloud_gpu_reference.json",
    )

    parser.add_argument(
        "--hourly-price",
        type=float,
        required=True,
        help="Actual cloud instance hourly price in USD.",
    )

    parser.add_argument(
        "--video-fps",
        type=float,
        required=True,
        help=(
            "Source video frame rate. "
            "Example: 30 for a 30 FPS video."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve paths relative to the repository root when needed."""
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_benchmark(path: Path) -> dict:
    """Load and validate the cloud benchmark result."""
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark result not found: {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    try:
        fps = float(data["metrics"]["fps"])
        mean_ms = float(data["metrics"]["mean_ms"])
        p50_ms = float(data["metrics"]["p50_ms"])
        p95_ms = float(data["metrics"]["p95_ms"])
        runs = int(data["metrics"]["runs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Benchmark JSON does not contain the expected metrics."
        ) from exc

    if fps <= 0:
        raise ValueError("Benchmark FPS must be greater than zero.")

    if mean_ms <= 0:
        raise ValueError("Mean latency must be greater than zero.")

    if runs <= 0:
        raise ValueError("Benchmark run count must be greater than zero.")

    data["_validated"] = {
        "fps": fps,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "runs": runs,
    }

    return data


def calculate_costs(
    fps: float,
    hourly_price: float,
    video_fps: float,
) -> dict:
    """
    Calculate inference-only production costs.

    Image workload:
        cost_per_image = hourly_price / (fps * 3600)

    Video workload:
        processing_time_hours =
            video_frames / (fps * 3600)

        For one hour of source video:
            video_frames = video_fps * 3600

        Therefore:
            cost_per_video_hour =
                hourly_price * video_fps / fps

    These estimates assume:
        - one inference per image/frame
        - one cloud GPU instance
        - no batching beyond the benchmark configuration
        - no storage/network/egress/model-hosting costs
        - measured FPS represents sustained processing throughput
    """

    if hourly_price <= 0:
        raise ValueError(
            "Hourly cloud price must be greater than zero."
        )

    if video_fps <= 0:
        raise ValueError(
            "Video FPS must be greater than zero."
        )

    seconds_per_image = 1.0 / fps
    hours_per_image = seconds_per_image / 3600.0

    cost_per_image = hourly_price * hours_per_image
    cost_per_1000_images = cost_per_image * 1000.0

    frames_per_video_hour = video_fps * 3600.0

    processing_hours_for_video_hour = (
        frames_per_video_hour / fps / 3600.0
    )

    cost_per_video_hour = (
        hourly_price * processing_hours_for_video_hour
    )

    realtime_capacity_ratio = fps / video_fps

    return {
        "images": {
            "throughput_fps": fps,
            "cost_per_image_usd": cost_per_image,
            "cost_per_1000_images_usd": cost_per_1000_images,
        },
        "video": {
            "source_fps": video_fps,
            "source_duration_hours": 1.0,
            "frames": int(frames_per_video_hour),
            "processing_hours": processing_hours_for_video_hour,
            "cost_per_source_video_hour_usd": cost_per_video_hour,
            "realtime_capacity_ratio": realtime_capacity_ratio,
            "realtime_capable": fps >= video_fps,
        },
    }


def main() -> None:
    args = parse_args()

    benchmark_path = resolve_project_path(args.benchmark)
    output_path = resolve_project_path(args.output)

    benchmark = load_benchmark(benchmark_path)

    metrics = benchmark["_validated"]

    cost = calculate_costs(
        fps=metrics["fps"],
        hourly_price=args.hourly_price,
        video_fps=args.video_fps,
    )

    result = {
        "experiment": (
            "Foundry M3 — Cloud GPU Production Cost Analysis"
        ),
        "source_benchmark": str(
            benchmark_path.relative_to(PROJECT_ROOT)
        ),
        "cloud": {
            "hourly_price_usd": args.hourly_price,
        },
        "benchmark": {
            "runtime": benchmark.get("runtime"),
            "gpu": benchmark.get("host", {}).get("gpu"),
            "model": benchmark.get("model"),
            "input": benchmark.get("input"),
            "runs": metrics["runs"],
            "mean_ms": metrics["mean_ms"],
            "p50_ms": metrics["p50_ms"],
            "p95_ms": metrics["p95_ms"],
            "fps": metrics["fps"],
        },
        "cost_analysis": cost,
        "scope": {
            "included": [
                "GPU compute time implied by measured throughput",
                "One inference per image/frame",
            ],
            "excluded": [
                "Cloud storage",
                "Network transfer",
                "Data egress",
                "Load balancing",
                "Autoscaling overhead",
                "Idle instances",
                "Model hosting overhead",
                "Preprocessing and decode outside the benchmark",
            ],
        },
        "notes": [
            "Hourly price must correspond to the exact instance, "
            "region, and pricing model used for the experiment.",
            "The image estimate is inference-throughput based.",
            "The video estimate assumes one inference per source frame.",
            "For production budgeting, infrastructure overhead should "
            "be added separately.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("Foundry M3 — Cloud GPU Production Cost Analysis")
    print("=" * 70)
    print(f"GPU:                 {benchmark.get('host', {}).get('gpu')}")
    print(f"Measured FPS:        {metrics['fps']:.3f}")
    print(f"Cloud price:         ${args.hourly_price:.6f}/hour")
    print()
    print(
        "Cost / 1,000 images: "
        f"${cost['images']['cost_per_1000_images_usd']:.6f}"
    )
    print(
        "Cost / video hour:   "
        f"${cost['video']['cost_per_source_video_hour_usd']:.6f}"
    )
    print(
        "Real-time capacity:  "
        f"{cost['video']['realtime_capacity_ratio']:.2f}x"
    )
    print(
        "Real-time capable:   "
        f"{cost['video']['realtime_capable']}"
    )
    print()
    print(f"Saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()