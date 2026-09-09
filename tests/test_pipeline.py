from __future__ import annotations

import numpy as np
import pytest

from gpu_edges.config import PipelineConfig
from gpu_edges.cpu import sobel_components, sobel_edges
from gpu_edges.cuda import benchmark_gpu, cuda_available, sobel_edges_gpu
from gpu_edges.data import generate_image
from gpu_edges.filters import gaussian_blur
from gpu_edges.metrics import (
    adaptive_threshold,
    comparison_metrics,
    edge_mask,
    edge_orientation_histogram,
    edge_statistics,
    speedup_ratio,
    throughput_mpix_per_second,
)
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


def test_cpu_sobel_components_match_magnitude_output() -> None:
    image = generate_image(20, 18)
    horizontal, vertical, magnitude = sobel_components(image)

    assert horizontal.shape == image.shape
    assert vertical.shape == image.shape
    np.testing.assert_allclose(magnitude, sobel_edges(image))


def test_gaussian_blur_preserves_shape_and_reduces_impulse() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0

    blurred = gaussian_blur(image, sigma=1.0)

    assert blurred.shape == image.shape
    assert blurred.dtype == np.float32
    assert 0 < blurred[4, 4] < 1.0
    assert blurred[4, 3] > 0


def test_gaussian_blur_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_blur(np.ones((4, 4), dtype=np.float32), sigma=-0.1)
    with pytest.raises(ValueError, match="two-dimensional"):
        gaussian_blur(np.ones((4, 4, 1), dtype=np.float32), sigma=1.0)


def test_comparison_metrics_report_zero_for_matching_arrays() -> None:
    values = np.ones((3, 3), dtype=np.float32)
    assert comparison_metrics(values, values) == {
        "max_absolute_error": 0.0,
        "rmse": 0.0,
    }


def test_comparison_metrics_reject_bad_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        comparison_metrics(np.ones((3, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        comparison_metrics(np.ones((3, 3)), np.full((3, 3), np.nan))


def test_edge_statistics_report_density_and_magnitude() -> None:
    edges = np.array(
        [
            [0.0, 0.1, 0.3],
            [0.4, 0.0, 0.8],
        ],
        dtype=np.float32,
    )

    stats = edge_statistics(edges, threshold=0.25)

    assert stats["edge_density"] == 0.5
    assert stats["max_magnitude"] == pytest.approx(0.8)
    assert stats["mean_magnitude"] == pytest.approx(float(edges.mean()))


def test_edge_mask_thresholds_magnitude_image() -> None:
    edges = np.array([[0.0, 0.2], [0.5, 0.7]], dtype=np.float32)

    mask = edge_mask(edges, threshold=0.5)

    np.testing.assert_array_equal(mask, np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_edge_statistics_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        edge_statistics(np.array([[0.0, np.inf]], dtype=np.float32))


def test_edge_orientation_histogram_counts_active_edges() -> None:
    horizontal = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    vertical = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    magnitude = np.ones((2, 2), dtype=np.float32)

    rows = edge_orientation_histogram(horizontal, vertical, magnitude, threshold=0.5, bins=2)

    assert len(rows) == 2
    assert sum(row["count"] for row in rows) == 4
    assert sum(row["fraction"] for row in rows) == pytest.approx(1.0)


def test_adaptive_threshold_uses_requested_quantile() -> None:
    edges = np.array([[0.0, 0.1], [0.4, 0.8]], dtype=np.float32)

    assert adaptive_threshold(edges, 0.75) == pytest.approx(float(np.quantile(edges, 0.75)))


def test_adaptive_threshold_rejects_bad_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        adaptive_threshold(np.ones((3, 3)), 1.0)


def test_speedup_ratio_handles_zero_gpu_time() -> None:
    assert speedup_ratio(4.0, 2.0) == 2.0
    assert speedup_ratio(4.0, 0.0) is None
    with pytest.raises(ValueError, match="timings"):
        speedup_ratio(4.0, -0.1)


def test_throughput_reports_megapixels_per_second() -> None:
    assert throughput_mpix_per_second(1000, 1000, 2.0) == pytest.approx(500.0)
    assert throughput_mpix_per_second(1000, 1000, 0.0) is None
    with pytest.raises(ValueError, match="height"):
        throughput_mpix_per_second(0, 100, 1.0)


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
    assert report["cpu_throughput_mpix_per_s"] > 0
    assert report["edge_threshold"] == config.edge_threshold
    assert "edge_density" in report["edge_statistics"]
    assert len(report["edge_orientation_histogram"]) == 8
    assert config.output_path.exists()
    assert config.report_path.exists()


def test_cpu_pipeline_reports_adaptive_threshold(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=2,
        edge_threshold=0.1,
        edge_quantile=0.80,
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["edge_quantile"] == 0.80
    assert report["edge_threshold"] != 0.1
    assert 0.0 <= report["edge_statistics"]["edge_density"] <= 1.0


def test_cpu_pipeline_reports_blur_setting(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=2,
        blur_sigma=0.8,
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["blur_sigma"] == 0.8
    assert config.report_path.exists()


def test_cpu_pipeline_can_write_binary_edge_mask(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=2,
        output_path=tmp_path / "edges.png",
        mask_output_path=tmp_path / "mask.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["mask_output"] == str(config.mask_output_path)
    assert config.mask_output_path.exists()


def test_config_rejects_oversized_blocks() -> None:
    with pytest.raises(ValueError, match="1024"):
        PipelineConfig(block_x=33, block_y=32)


def test_config_rejects_negative_edge_threshold() -> None:
    with pytest.raises(ValueError, match="edge_threshold"):
        PipelineConfig(edge_threshold=-0.1)


def test_config_rejects_negative_blur_sigma() -> None:
    with pytest.raises(ValueError, match="blur_sigma"):
        PipelineConfig(blur_sigma=-0.1)


def test_config_rejects_bad_edge_quantile() -> None:
    with pytest.raises(ValueError, match="edge_quantile"):
        PipelineConfig(edge_quantile=1.0)


def test_cuda_availability_returns_a_boolean() -> None:
    assert isinstance(cuda_available(), bool)


def test_gpu_entrypoints_validate_inputs_before_device_check() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sobel_edges_gpu(np.zeros((8, 8, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="1024"):
        benchmark_gpu(np.zeros((8, 8), dtype=np.float32), 33, 32, 1)
