from __future__ import annotations

import platform
import sys

import psutil
import torch


def get_system_info() -> dict:
    """Return reproducible system/runtime information."""
    memory = psutil.virtual_memory()

    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "ram_available_gb": round(memory.available / (1024**3), 2),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


if __name__ == "__main__":
    for key, value in get_system_info().items():
        print(f"{key}: {value}")