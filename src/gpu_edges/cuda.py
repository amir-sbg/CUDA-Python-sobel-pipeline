from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from numbers import Integral
from time import perf_counter

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def cuda_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _sobel_kernel():
    if cp is None:
        raise RuntimeError("CuPy is not installed")
    source = files("gpu_edges").joinpath("kernels", "sobel.cu").read_text()
    return cp.RawKernel(source, "sobel_edges")


def _validate_launch(
    image: np.ndarray,
    block_x: int,
    block_y: int,
    iterations: int | None = None,
) -> None:
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if min(values.shape) < 3:
        raise ValueError("image dimensions must be at least 3")
    if not isinstance(block_x, Integral) or not isinstance(block_y, Integral):
        raise ValueError("block dimensions must be integers")
    if block_x < 1 or block_y < 1 or block_x * block_y > 1024:
        raise ValueError("block dimensions must be positive and use at most 1024 threads")
    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be at least 1")


def _launch(kernel, image, output, block_x: int, block_y: int) -> None:
    height, width = image.shape
    grid = (
        (width + block_x - 1) // block_x,
        (height + block_y - 1) // block_y,
    )
    kernel(
        grid,
        (block_x, block_y),
        (image, output, np.int32(height), np.int32(width)),
    )


def sobel_edges_gpu(
    image: np.ndarray,
    block_x: int = 16,
    block_y: int = 16,
) -> np.ndarray:
    _validate_launch(image, block_x, block_y)
    if not cuda_available():
        raise RuntimeError("a CUDA-enabled CuPy runtime is required")
    values = cp.asarray(image, dtype=cp.float32)
    output = cp.empty_like(values)
    _launch(_sobel_kernel(), values, output, block_x, block_y)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(output)


def benchmark_gpu(
    image: np.ndarray,
    block_x: int,
    block_y: int,
    iterations: int,
) -> tuple[np.ndarray, float]:
    _validate_launch(image, block_x, block_y, iterations)
    if not cuda_available():
        raise RuntimeError("a CUDA-enabled CuPy runtime is required")
    values = cp.asarray(image, dtype=cp.float32)
    output = cp.empty_like(values)
    kernel = _sobel_kernel()
    _launch(kernel, values, output, block_x, block_y)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(iterations):
        _launch(kernel, values, output, block_x, block_y)
    stop.record()
    stop.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, stop) / iterations
    return cp.asnumpy(output), float(elapsed_ms)
