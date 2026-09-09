from __future__ import annotations

from math import ceil

import numpy as np


def gaussian_blur(image: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must not be negative")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if sigma == 0:
        return values.copy()

    kernel = _gaussian_kernel1d(sigma)
    blurred = _convolve_axis(values, kernel, axis=1)
    blurred = _convolve_axis(blurred, kernel, axis=0)
    return blurred.astype(np.float32)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        start = offset
        stop = start + image.shape[axis]
        if axis == 0:
            output += weight * padded[start:stop, :]
        else:
            output += weight * padded[:, start:stop]
    return output
