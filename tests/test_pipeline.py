from __future__ import annotations

import numpy as np
import pytest

from gpu_edges.config import PipelineConfig
from gpu_edges.cpu import sobel_edges
from gpu_edges.cuda import benchmark_gpu, cuda_available, sobel_edges_gpu
from gpu_edges.data import generate_image
from gpu_edges.metrics import comparison_metrics
from gpu_edges.pipeline import run


def test_generated_image_is_repeatable() -> None:
    first = generate_image(32, 24, seed=4)
    second = generate_image(32, 24, seed=4)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (32, 24)
    assert first.dtype == np.float32


def test_cpu_sobel_preserves_shape_and_zero_border() -> None:
    image = generate_image(20, 18)
    edges = sobel_edges(image)
    assert edges.shape == image.shape
    assert np.all(edges[[0, -1], :] == 0)
    assert np.all(edges[:, [0, -1]] == 0)
    assert np.isfinite(edges).all()


def test_comparison_metrics_report_zero_for_matching_arrays() -> None:
    values = np.ones((3, 3), dtype=np.float32)
    assert comparison_metrics(values, values) == {
        "max_absolute_error": 0.0,
        "rmse": 0.0,
    }


def test_cpu_pipeline_writes_output_and_report(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=2,
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )
    report = run(config, cpu_only=True)
    assert report["backend"] == "cpu"
    assert report["input_source"] == "generated"
    assert report["input_dtype"] == "float32"
    assert config.output_path.exists()
    assert config.report_path.exists()


def test_config_rejects_oversized_blocks() -> None:
    with pytest.raises(ValueError, match="1024"):
        PipelineConfig(block_x=33, block_y=32)


def test_cuda_availability_returns_a_boolean() -> None:
    assert isinstance(cuda_available(), bool)


def test_gpu_entrypoints_validate_inputs_before_device_check() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sobel_edges_gpu(np.zeros((8, 8, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="1024"):
        benchmark_gpu(np.zeros((8, 8), dtype=np.float32), 33, 32, 1)
