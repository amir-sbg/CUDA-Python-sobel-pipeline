from __future__ import annotations

import numpy as np


def sobel_edges(image: np.ndarray, operator: str = "sobel") -> np.ndarray:
    _, _, edges = sobel_components(image, operator=operator)
    return edges.astype(np.float32)


def sobel_components(
    image: np.ndarray,
    operator: str = "sobel",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Sobel input must be a two-dimensional grayscale image")
    if not np.all(np.isfinite(values)):
        raise ValueError("Sobel input must contain only finite values")
    if operator not in {"sobel", "scharr"}:
        raise ValueError("operator must be 'sobel' or 'scharr'")
    corner_weight, axial_weight = (1.0, 2.0) if operator == "sobel" else (3.0, 10.0)
    padded = np.pad(values, 1, mode="constant")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]
    horizontal = (
        -corner_weight * top_left + corner_weight * top_right
        - axial_weight * left + axial_weight * right
        - corner_weight * bottom_left + corner_weight * bottom_right
    )
    vertical = (
        -corner_weight * top_left - axial_weight * top - corner_weight * top_right
        + corner_weight * bottom_left + axial_weight * bottom + corner_weight * bottom_right
    )
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
