from __future__ import annotations

import numpy as np


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
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

    return {
        "mean_magnitude": float(np.mean(values)),
        "std_magnitude": float(np.std(values)),
        "max_magnitude": float(np.max(values)),
        "edge_density": float(np.mean(values >= threshold)),
    }
