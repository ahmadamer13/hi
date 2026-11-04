"""Latency / throughput / GPU memory measurement.

Numbers are only meaningful on the TARGET GPU (RTX 3060/4060 class) with FP16 TensorRT engines
(see benchmark/trt/). On CPU the harness still runs, and the report marks those rows "CPU, not target".
"""
from __future__ import annotations

import platform
import time

import numpy as np


def device_info() -> dict:
    info = {"host": platform.node(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    try:
        import tensorrt  # noqa: F401
        info["tensorrt"] = tensorrt.__version__
    except Exception:  # noqa: BLE001
        pass
    info.setdefault("gpu", None)
    return info


def gpu_mem_used_mb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


def measure(detector, frames: list[np.ndarray], warmup: int = 10, iters: int = 100,
            base_mem_mb: float | None = None) -> dict:
    """End-to-end detector latency including pre/post-processing (what the pipeline pays).

    base_mem_mb: GPU memory in use BEFORE the model was loaded, so the figure includes weights.
    """
    base = base_mem_mb if base_mem_mb is not None else gpu_mem_used_mb()
    for i in range(warmup):
        detector(frames[i % len(frames)])
    lat = []
    for i in range(iters):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        detector(f)
        lat.append(time.perf_counter() - t)
    peak = gpu_mem_used_mb()
    lat_ms = np.array(lat) * 1000
    return {
        "latency_ms_p50": float(np.percentile(lat_ms, 50)),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)),
        "fps": float(1000 / lat_ms.mean()),
        "gpu_mem_mb": (peak - base) if (peak is not None and base is not None) else None,
        "iters": iters,
    }
