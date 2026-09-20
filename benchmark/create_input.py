"""
Foundry M3 — Deterministic Benchmark Input

Creates a fixed 640x640 RGB image for reproducible inference
benchmarking across PyTorch, ONNX Runtime, and OpenVINO.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "benchmark" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = INPUT_DIR / "benchmark_640.png"

IMAGE_SIZE = 640


def create_benchmark_image() -> None:
    """Create a deterministic synthetic RGB benchmark image."""

    # Fixed seed guarantees identical input generation.
    rng = np.random.default_rng(42)

    # Generate deterministic RGB image.
    image = rng.integers(
        low=0,
        high=256,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    Image.fromarray(image, mode="RGB").save(OUTPUT_FILE)

    print(f"Benchmark input saved to:")
    print(OUTPUT_FILE)
    print(f"Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Format: RGB PNG")
    print("Seed: 42")


if __name__ == "__main__":
    create_benchmark_image()