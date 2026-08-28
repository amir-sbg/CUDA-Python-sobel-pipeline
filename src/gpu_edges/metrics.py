from __future__ import annotations

import numpy as np


def _validated_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError("reference and candidate must have the same shape")
    if ref.size == 0:
        raise ValueError("reference and candidate must not be empty")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        raise ValueError("reference and candidate must contain only finite values")
    return ref, got


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref, got = _validated_pair(reference, candidate)
    difference = got - ref
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def edge_statistics(edges: np.ndarray, threshold: float = 0.20) -> dict[str, float]:
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("edges must be two-dimensional")
    if values.size == 0:
        raise ValueError("edges must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("edges must contain only finite values")

    return {
        "mean_magnitude": float(np.mean(values)),
        "std_magnitude": float(np.std(values)),
        "max_magnitude": float(np.max(values)),
        "edge_density": float(np.mean(values >= threshold)),
    }


def speedup_ratio(cpu_ms: float, gpu_ms: float) -> float | None:
    if cpu_ms < 0 or gpu_ms < 0:
        raise ValueError("timings must not be negative")
    if gpu_ms == 0:
        return None
    return cpu_ms / gpu_ms
