"""
Foundry M3 — Baseline Benchmark

This module records the benchmark environment before model inference.
Actual inference measurements will be added after the baseline model
is selected and downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.system_info import get_system_info


RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    system_info = get_system_info()

    result = {
        "benchmark": "m3-baseline",
        "version": "0.1.0",
        "status": "environment-only",
        "system": system_info,
        "model": None,
        "input": None,
        "metrics": None,
        "notes": [
            "Environment captured before model inference.",
            "Official performance results require a controlled benchmark run.",
        ],
    }

    output_file = RESULTS_DIR / "baseline_environment.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Benchmark environment saved to:")
    print(output_file)


if __name__ == "__main__":
    main()