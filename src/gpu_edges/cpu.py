from __future__ import annotations

import numpy as np


def sobel_edges(image: np.ndarray) -> np.ndarray:
    _, _, edges = sobel_components(image)
    return edges.astype(np.float32)


def sobel_components(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Sobel input must be a two-dimensional grayscale image")
    if not np.all(np.isfinite(values)):
        raise ValueError("Sobel input must contain only finite values")
    padded = np.pad(values, 1, mode="constant")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]
    horizontal = -top_left + top_right - 2 * left + 2 * right - bottom_left + bottom_right
    vertical = -top_left - 2 * top - top_right + bottom_left + 2 * bottom + bottom_right
    edges = np.sqrt(horizontal**2 + vertical**2)
    horizontal[[0, -1], :] = 0.0
    horizontal[:, [0, -1]] = 0.0
    vertical[[0, -1], :] = 0.0
    vertical[:, [0, -1]] = 0.0
    edges[[0, -1], :] = 0.0
    edges[:, [0, -1]] = 0.0
    return (
        horizontal.astype(np.float32),
        vertical.astype(np.float32),
        edges.astype(np.float32),
    )
