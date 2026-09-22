from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ACCURACY_TOLERANCE_MAP_POINTS = 1.0
DEFAULT_LATENCY_TOLERANCE_PERCENT = 10.0


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def extract_map_50_95(data: dict) -> float:
    if isinstance(data.get("mAP50_95"), (int, float)):
        return float(data["mAP50_95"])

    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(
        metrics.get("mAP50_95"),
        (int, float),
    ):
        return float(metrics["mAP50_95"])

    reference = data.get("pytorch_fp32_reference")
    if isinstance(reference, dict) and isinstance(
        reference.get("mAP50_95"),
        (int, float),
    ):
        return float(reference["mAP50_95"])

    raise KeyError(
        "Could not find mAP50_95 in the supplied JSON."
    )


def extract_cpu_p95_ms(data: dict) -> float:
    results = data.get("results")

    if isinstance(results, dict):
        cpu = results.get("CPU")

        if isinstance(cpu, dict):
            overall = cpu.get("overall")

            if isinstance(overall, dict) and isinstance(
                overall.get("p95_ms"),
                (int, float),
            ):
                return float(overall["p95_ms"])

    metrics = data.get("metrics")

    if isinstance(metrics, dict) and isinstance(
        metrics.get("p95_latency_ms"),
        (int, float),
    ):
        return float(metrics["p95_latency_ms"])

    if isinstance(data.get("p95_latency_ms"), (int, float)):
        return float(data["p95_latency_ms"])

    raise KeyError(
        "Could not find CPU p95 latency in the supplied JSON."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Foundry M3 accuracy + latency regression gate."
    )

    parser.add_argument(
        "--reference-accuracy",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-accuracy",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--reference-latency",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-latency",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--accuracy-tolerance-map-points",
        type=float,
        default=DEFAULT_ACCURACY_TOLERANCE_MAP_POINTS,
    )

    parser.add_argument(
        "--latency-tolerance-percent",
        type=float,
        default=DEFAULT_LATENCY_TOLERANCE_PERCENT,
    )

    args = parser.parse_args()

    try:
        reference_accuracy = load_json(
            args.reference_accuracy
        )

        candidate_accuracy = load_json(
            args.candidate_accuracy
        )

        reference_latency = load_json(
            args.reference_latency
        )

        candidate_latency = load_json(
            args.candidate_latency
        )

        reference_map = extract_map_50_95(
            reference_accuracy
        )

        candidate_map = extract_map_50_95(
            candidate_accuracy
        )

        reference_p95 = extract_cpu_p95_ms(
            reference_latency
        )

        candidate_p95 = extract_cpu_p95_ms(
            candidate_latency
        )

    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"REGRESSION GATE ERROR: {exc}")
        return 2

    map_delta_points = (
        candidate_map - reference_map
    ) * 100.0

    map_loss_points = -map_delta_points

    latency_delta_percent = (
        (candidate_p95 - reference_p95)
        / reference_p95
    ) * 100.0

    accuracy_failed = (
        map_loss_points
        > args.accuracy_tolerance_map_points
    )

    latency_failed = (
        latency_delta_percent
        > args.latency_tolerance_percent
    )

    print()
    print("=" * 72)
    print("Foundry M3 — Regression Gate")
    print("=" * 72)

    print("ACCURACY")
    print(f"Reference mAP50-95 : {reference_map:.6f}")
    print(f"Candidate mAP50-95 : {candidate_map:.6f}")
    print(f"mAP delta          : {map_delta_points:+.3f} points")
    print(
        f"Allowed mAP loss   : "
        f"{args.accuracy_tolerance_map_points:.3f} points"
    )

    print()

    print("LATENCY")
    print(f"Reference CPU p95  : {reference_p95:.3f} ms")
    print(f"Candidate CPU p95  : {candidate_p95:.3f} ms")
    print(f"p95 delta          : {latency_delta_percent:+.2f}%")
    print(
        f"Allowed p95 growth : "
        f"{args.latency_tolerance_percent:.2f}%"
    )

    print()
    print("-" * 72)

    if accuracy_failed:
        print(
            "FAIL: accuracy regression exceeds "
            "the allowed mAP loss."
        )
    else:
        print(
            "PASS: accuracy regression is within tolerance."
        )

    if latency_failed:
        print(
            "FAIL: latency regression exceeds "
            "the allowed CPU p95 growth."
        )
    else:
        print(
            "PASS: latency regression is within tolerance."
        )

    print("-" * 72)

    if accuracy_failed or latency_failed:
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

