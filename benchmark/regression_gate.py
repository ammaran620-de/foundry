from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_ACCURACY_TOLERANCE_MAP_POINTS = 1.0
DEFAULT_LATENCY_TOLERANCE_PERCENT = 10.0

# Slack for floating-point error, so that a change of exactly the allowed size (for example a
# 1.0 point mAP loss with a 1.0 point tolerance) passes instead of failing on 1.0000000000000009.
EPSILON = 1e-9

# Block of deployment_accuracy.json that holds the PyTorch FP32 reference run.
FP32_REFERENCE_BLOCK = "pytorch_fp32_reference"


class GateInputError(ValueError):
    """The supplied files cannot be compared meaningfully. Reported with exit status 2."""


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def finite_number(value, label: str) -> float:
    """A real, finite number. Booleans, strings, NaN and infinity are rejected: NaN compares false
    with everything, so a NaN candidate would otherwise pass every threshold."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(f"{label} is not a number: {value!r}")
    if not math.isfinite(value):
        raise GateInputError(f"{label} is not finite: {value!r}")
    return float(value)


def _map_value(block: dict, label: str) -> float | None:
    for key in ("mAP50_95", "map50_95"):
        if key in block:
            return finite_number(block[key], f"{label} {key}")
    return None


def deployment_names(data: dict) -> list[str]:
    """Deployments a deployment-accuracy file holds: the FP32 reference block and every entry
    of its `results` mapping that carries an mAP."""
    names = []
    if isinstance(data.get(FP32_REFERENCE_BLOCK), dict):
        names.append(FP32_REFERENCE_BLOCK)
    results = data.get("results")
    if isinstance(results, dict):
        names += [name for name, block in results.items() if isinstance(block, dict) and _map_value(block, name) is not None]
    return names


def extract_map_50_95(data: dict, deployment: str | None = None, label: str = "accuracy file") -> float:
    """mAP50-95 as a fraction in [0, 1].

    A deployment-accuracy file holds several runs (the PyTorch FP32 reference plus each deployed
    runtime), so the run to compare must be named with `deployment`. Guessing would silently compare
    the FP32 reference with itself and never see a deployment lose accuracy.
    """
    names = deployment_names(data)

    if deployment is not None:
        if deployment == FP32_REFERENCE_BLOCK and isinstance(data.get(FP32_REFERENCE_BLOCK), dict):
            block = data[FP32_REFERENCE_BLOCK]
        elif isinstance(data.get("results"), dict) and isinstance(data["results"].get(deployment), dict):
            block = data["results"][deployment]
        else:
            raise KeyError(f"{label} has no deployment {deployment!r}; it holds: {', '.join(names) or 'none'}")
        value = _map_value(block, f"{label} [{deployment}]")
        if value is None:
            raise KeyError(f"{label} [{deployment}] has no mAP50_95")
    else:
        if names:
            raise GateInputError(
                f"{label} holds several runs ({', '.join(names)}); name the one to compare with "
                "--reference-deployment / --candidate-deployment"
            )
        value = _map_value(data, label)
        if value is None and isinstance(data.get("metrics"), dict):
            value = _map_value(data["metrics"], f"{label} metrics")
        if value is None:
            raise KeyError("Could not find mAP50_95 in the supplied JSON.")

    if not 0.0 <= value <= 1.0:
        raise GateInputError(
            f"{label} mAP50_95 is {value}, outside 0 to 1. The gate expects a fraction, not a percentage."
        )
    return value


def extract_cpu_p95_ms(data: dict, label: str = "latency file") -> float:
    value = None
    results = data.get("results")

    if isinstance(results, dict):
        cpu = results.get("CPU")

        if isinstance(cpu, dict):
            overall = cpu.get("overall")

            if isinstance(overall, dict) and "p95_ms" in overall:
                value = finite_number(overall["p95_ms"], f"{label} CPU p95_ms")

    if value is None:
        metrics = data.get("metrics")

        if isinstance(metrics, dict) and "p95_latency_ms" in metrics:
            value = finite_number(metrics["p95_latency_ms"], f"{label} p95_latency_ms")

    if value is None and "p95_latency_ms" in data:
        value = finite_number(data["p95_latency_ms"], f"{label} p95_latency_ms")

    if value is None:
        raise KeyError("Could not find CPU p95 latency in the supplied JSON.")

    if value <= 0:
        raise GateInputError(f"{label} CPU p95 latency is {value} ms; it must be positive.")
    return value


def cpu_device_name(data: dict) -> str | None:
    cpu = (data.get("results") or {}).get("CPU") if isinstance(data.get("results"), dict) else None
    name = cpu.get("device_name") if isinstance(cpu, dict) else None
    return name if isinstance(name, str) else None


def check_comparable(
    reference_accuracy: dict,
    candidate_accuracy: dict,
    reference_latency: dict,
    candidate_latency: dict,
    allow_device_mismatch: bool,
) -> None:
    """Refuse comparisons that cannot mean anything: different input sizes, or latencies measured
    on different CPUs (unless the caller says that is intended)."""
    ref_size = reference_accuracy.get("image_size")
    cand_size = candidate_accuracy.get("image_size")
    if ref_size is not None and cand_size is not None and ref_size != cand_size:
        raise GateInputError(f"accuracy files were evaluated at different image sizes ({ref_size} and {cand_size})")

    ref_device = cpu_device_name(reference_latency)
    cand_device = cpu_device_name(candidate_latency)
    if ref_device and cand_device and ref_device != cand_device and not allow_device_mismatch:
        raise GateInputError(
            f"latency was measured on different CPUs ({ref_device!r} and {cand_device!r}); "
            "pass --allow-device-mismatch if that comparison is intended"
        )


def exceeds(amount: float, tolerance: float) -> bool:
    return amount - tolerance > EPSILON


def tolerance_value(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}")
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("tolerance must be a finite number, zero or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Foundry M3 accuracy + latency regression gate. Exit status 0: within tolerance, "
            "1: regression, 2: the inputs cannot be compared."
        )
    )

    parser.add_argument("--reference-accuracy", required=True, type=Path)
    parser.add_argument("--candidate-accuracy", required=True, type=Path)
    parser.add_argument("--reference-latency", required=True, type=Path)
    parser.add_argument("--candidate-latency", required=True, type=Path)

    parser.add_argument(
        "--reference-deployment",
        help="run to use from a deployment-accuracy file, e.g. pytorch_fp32_reference or onnxruntime_cpu",
    )
    parser.add_argument(
        "--candidate-deployment",
        help="run to use from a deployment-accuracy file, e.g. openvino_fp16_cpu",
    )

    parser.add_argument(
        "--accuracy-tolerance-map-points",
        type=tolerance_value,
        default=DEFAULT_ACCURACY_TOLERANCE_MAP_POINTS,
    )
    parser.add_argument(
        "--latency-tolerance-percent",
        type=tolerance_value,
        default=DEFAULT_LATENCY_TOLERANCE_PERCENT,
    )
    parser.add_argument(
        "--allow-device-mismatch",
        action="store_true",
        help="compare latencies even if they were measured on different CPUs",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        reference_accuracy = load_json(args.reference_accuracy)
        candidate_accuracy = load_json(args.candidate_accuracy)
        reference_latency = load_json(args.reference_latency)
        candidate_latency = load_json(args.candidate_latency)

        reference_map = extract_map_50_95(
            reference_accuracy, args.reference_deployment, "reference accuracy"
        )
        candidate_map = extract_map_50_95(
            candidate_accuracy, args.candidate_deployment, "candidate accuracy"
        )
        reference_p95 = extract_cpu_p95_ms(reference_latency, "reference latency")
        candidate_p95 = extract_cpu_p95_ms(candidate_latency, "candidate latency")

        check_comparable(
            reference_accuracy,
            candidate_accuracy,
            reference_latency,
            candidate_latency,
            args.allow_device_mismatch,
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

    accuracy_failed = exceeds(
        map_loss_points,
        args.accuracy_tolerance_map_points,
    )

    latency_failed = exceeds(
        latency_delta_percent,
        args.latency_tolerance_percent,
    )

    print()
    print("=" * 72)
    print("Foundry M3 — Regression Gate")
    print("=" * 72)

    print("ACCURACY")
    print(f"Reference run      : {args.reference_deployment or 'single run'}")
    print(f"Candidate run      : {args.candidate_deployment or 'single run'}")
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
