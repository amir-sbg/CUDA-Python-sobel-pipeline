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


def edge_mask(edges: np.ndarray, threshold: float) -> np.ndarray:
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    values = np.asarray(edges, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("edges must be two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("edges must contain only finite values")
    return (values >= threshold).astype(np.float32)


def edge_orientation_histogram(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    bins: int = 8,
) -> list[dict[str, float | int]]:
    gx, gy = _validated_pair(horizontal, vertical)
    _, mag = _validated_pair(horizontal, magnitude)
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    if bins < 1:
        raise ValueError("bins must be positive")

    active = mag >= threshold
    if not np.any(active):
        return [
            {
                "bin": index + 1,
                "start_degrees": float(index * 180.0 / bins),
                "end_degrees": float((index + 1) * 180.0 / bins),
                "count": 0,
                "fraction": 0.0,
            }
            for index in range(bins)
        ]

    angles = np.degrees(np.arctan2(gy[active], gx[active]))
    angles = np.mod(angles, 180.0)
    counts, edges = np.histogram(angles, bins=bins, range=(0.0, 180.0))
    total = int(np.sum(counts))
    return [
        {
            "bin": index + 1,
            "start_degrees": float(edges[index]),
            "end_degrees": float(edges[index + 1]),
            "count": int(count),
            "fraction": float(count / total) if total else 0.0,
        }
        for index, count in enumerate(counts)
    ]


def non_maximum_suppression(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    gx, gy = _validated_pair(horizontal, vertical)
    _, mag = _validated_pair(horizontal, magnitude)
    if gx.ndim != 2:
        raise ValueError("gradient inputs must be two-dimensional")

    output = np.zeros_like(mag, dtype=np.float32)
    angles = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    height, width = mag.shape

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            angle = angles[row, col]
            current = mag[row, col]
            if angle < 22.5 or angle >= 157.5:
                before, after = mag[row, col - 1], mag[row, col + 1]
            elif angle < 67.5:
                before, after = mag[row - 1, col + 1], mag[row + 1, col - 1]
            elif angle < 112.5:
                before, after = mag[row - 1, col], mag[row + 1, col]
            else:
                before, after = mag[row - 1, col - 1], mag[row + 1, col + 1]

            if current >= before and current >= after:
                output[row, col] = current

    return output


def adaptive_threshold(edges: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("edges must be two-dimensional")
    if values.size == 0:
        raise ValueError("edges must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("edges must contain only finite values")
    return float(np.quantile(values, quantile))


def speedup_ratio(cpu_ms: float, gpu_ms: float) -> float | None:
    if cpu_ms < 0 or gpu_ms < 0:
        raise ValueError("timings must not be negative")
    if gpu_ms == 0:
        return None
    return cpu_ms / gpu_ms


def throughput_mpix_per_second(height: int, width: int, elapsed_ms: float) -> float | None:
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    if elapsed_ms < 0:
        raise ValueError("elapsed_ms must not be negative")
    if elapsed_ms == 0:
        return None
    pixels = height * width
    return pixels / 1_000_000.0 / (elapsed_ms / 1000.0)
