from __future__ import annotations

import numpy as np
import pytest

from gpu_edges.config import PipelineConfig
from gpu_edges.cpu import sobel_components, sobel_edges
from gpu_edges.cuda import benchmark_gpu, cuda_available, launch_shape, sobel_edges_gpu
from gpu_edges.data import generate_image
from gpu_edges.filters import gaussian_blur
from gpu_edges.metrics import (
    adaptive_threshold,
    binary_edge_metrics,
    comparison_metrics,
    edge_mask,
    edge_component_statistics,
    edge_magnitude_quantiles,
    edge_localization_distance,
    edge_orientation_coherence,
    edge_orientation_histogram,
    edge_quality_flags,
    edge_separation_metrics,
    edge_statistics,
    edge_threshold_curve,
    edge_topology_comparison,
    gradient_energy,
    hysteresis_edge_mask,
    non_maximum_suppression,
    orientation_distribution_metrics,
    pratt_figure_of_merit,
    nms_reduction_ratio,
    minimum_memory_bandwidth_gbps,
    multiscale_edge_persistence,
    scale_space_consistency,
    speedup_ratio,
    spatial_edge_metrics,
    throughput_mpix_per_second,
    threshold_sensitivity,
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


def test_scharr_operator_detects_step_edge() -> None:
    image = np.zeros((7, 7), dtype=np.float32)
    image[:, 4:] = 1.0

    horizontal, vertical, magnitude = sobel_components(image, operator="scharr")

    assert magnitude[3, 3] > sobel_edges(image)[3, 3]
    assert horizontal[3, 3] > 0
    assert vertical[3, 3] == pytest.approx(0.0)


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
    invalid = np.ones((4, 4), dtype=np.float32)
    invalid[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        gaussian_blur(invalid, sigma=1.0)


def test_cpu_sobel_rejects_non_finite_inputs() -> None:
    image = np.ones((4, 4), dtype=np.float32)
    image[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        sobel_edges(image)


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


def test_scale_space_consistency_is_one_for_rescaled_maps() -> None:
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    report = scale_space_consistency(reference, reference * 4)

    assert report["cosine_similarity"] == pytest.approx(1.0)
    assert report["pearson_correlation"] == pytest.approx(1.0)
    assert report["normalized_mae"] == pytest.approx(0.0)


def test_multiscale_persistence_tracks_strong_edge_overlap() -> None:
    reference = np.array([[0.0, 0.2, 0.8], [0.1, 0.4, 1.0]], dtype=np.float32)
    identical = multiscale_edge_persistence(reference, reference * 0.5, quantile=0.6)
    changed = reference.copy()
    changed[1, 2] = 0.0
    changed[0, 0] = 1.0
    shifted = multiscale_edge_persistence(reference, changed, quantile=0.6)

    assert identical["retained_edge_fraction"] == pytest.approx(1.0)
    assert identical["edge_iou"] == pytest.approx(1.0)
    assert shifted["retained_edge_fraction"] < 1.0
    assert shifted["introduced_edge_fraction"] > 0.0


def test_binary_edge_metrics_report_overlap() -> None:
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    candidate = np.array([[0.0, 1.0], [0.0, 1.0]])

    metrics = binary_edge_metrics(reference, candidate, threshold=0.5)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)


def test_binary_edge_metrics_handles_empty_edge_maps() -> None:
    metrics = binary_edge_metrics(np.zeros((2, 2)), np.zeros((2, 2)), 0.5)
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}


def test_binary_edge_metrics_report_zero_f1_for_disjoint_edges() -> None:
    reference = np.array([[1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0]], dtype=np.float32)

    assert binary_edge_metrics(reference, candidate, threshold=0.5)["f1"] == 0.0


def test_edge_threshold_curve_recovers_rescaled_edge_map() -> None:
    reference = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    candidate = reference * 0.5

    report = edge_threshold_curve(
        reference,
        candidate,
        reference_threshold=0.8,
        candidate_thresholds=(0.2, 0.5, 0.8),
    )

    assert report["best_threshold"] == pytest.approx(0.2)
    assert report["best_f1"] == pytest.approx(1.0)


def test_spatial_edge_metrics_tolerate_one_pixel_shift() -> None:
    reference = np.zeros((5, 5), dtype=np.float32)
    candidate = np.zeros((5, 5), dtype=np.float32)
    reference[2, 2] = 1.0
    candidate[2, 3] = 1.0

    strict = spatial_edge_metrics(reference, candidate, threshold=0.5, radius=0)
    tolerant = spatial_edge_metrics(reference, candidate, threshold=0.5, radius=1)

    assert strict["f1"] == 0.0
    assert tolerant["f1"] == 1.0


def test_edge_localization_distance_measures_pixel_shift() -> None:
    reference = np.zeros((5, 5), dtype=np.float32)
    candidate = np.zeros((5, 5), dtype=np.float32)
    reference[2, 2] = 1.0
    candidate[2, 3] = 1.0

    shifted = edge_localization_distance(reference, candidate, threshold=0.5)
    aligned = edge_localization_distance(reference, reference, threshold=0.5)

    assert shifted["symmetric_mean_pixels"] == pytest.approx(1.0)
    assert aligned["symmetric_mean_pixels"] == pytest.approx(0.0)


def test_pratt_score_penalizes_localization_and_false_edges() -> None:
    reference = np.zeros((7, 7), dtype=np.float32)
    shifted = np.zeros((7, 7), dtype=np.float32)
    noisy = np.zeros((7, 7), dtype=np.float32)
    reference[3, 2:5] = 1.0
    shifted[4, 2:5] = 1.0
    noisy[4, 2:5] = 1.0
    noisy[0, 0] = 1.0

    assert pratt_figure_of_merit(reference, reference, 0.5) == pytest.approx(1.0)
    assert pratt_figure_of_merit(reference, shifted, 0.5) < 1.0
    assert pratt_figure_of_merit(reference, noisy, 0.5) < pratt_figure_of_merit(
        reference, shifted, 0.5
    )


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


def test_threshold_sensitivity_tracks_density_changes() -> None:
    edges = np.array([[0.46, 0.50, 0.56, 0.90]], dtype=np.float32)

    report = threshold_sensitivity(edges, threshold=0.5, relative_delta=0.1)

    assert report["low_density"] == pytest.approx(1.0)
    assert report["center_density"] == pytest.approx(0.75)
    assert report["high_density"] == pytest.approx(0.5)
    assert report["density_span"] == pytest.approx(0.5)


def test_edge_separation_reports_gradient_contrast() -> None:
    edges = np.array([[0.05, 0.1, 0.8, 1.0]], dtype=np.float32)

    report = edge_separation_metrics(edges, threshold=0.5)

    assert report["edge_pixels"] == 2
    assert report["background_pixels"] == 2
    assert report["edge_mean"] > report["background_mean"]
    assert report["standardized_gap"] > 0


def test_edge_magnitude_quantiles_ignore_background_pixels() -> None:
    edges = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]], dtype=np.float32)

    summary = edge_magnitude_quantiles(edges, threshold=0.5)

    assert summary["active_pixels"] == 3
    assert summary["p50"] == pytest.approx(0.8)
    assert summary["p90"] > summary["p50"]


def test_gradient_energy_reports_directional_fraction() -> None:
    horizontal = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    vertical = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    report = gradient_energy(horizontal, vertical)

    assert report["mean_gradient_energy"] == pytest.approx(1.75)
    assert report["max_gradient_energy"] == pytest.approx(4.0)
    assert report["horizontal_energy_fraction"] == pytest.approx(5 / 7)


def test_orientation_coherence_distinguishes_aligned_gradients() -> None:
    horizontal = np.ones((3, 3), dtype=np.float32)
    vertical = np.zeros((3, 3), dtype=np.float32)
    magnitude = np.ones((3, 3), dtype=np.float32)

    assert edge_orientation_coherence(horizontal, vertical, magnitude, 0.5) == pytest.approx(1.0)

    mixed_horizontal = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    mixed_vertical = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    assert edge_orientation_coherence(
        mixed_horizontal,
        mixed_vertical,
        np.ones((2, 2), dtype=np.float32),
        0.5,
    ) == pytest.approx(0.0, abs=1e-7)


def test_orientation_distribution_reports_directional_entropy() -> None:
    magnitude = np.ones((2, 2), dtype=np.float32)
    aligned = orientation_distribution_metrics(
        np.ones((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        magnitude,
        threshold=0.5,
        bins=4,
    )
    mixed = orientation_distribution_metrics(
        np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32),
        magnitude,
        threshold=0.5,
        bins=4,
    )

    assert aligned["normalized_entropy"] == pytest.approx(0.0)
    assert aligned["dominant_orientation_fraction"] == pytest.approx(1.0)
    assert mixed["normalized_entropy"] > aligned["normalized_entropy"]


def test_edge_quality_flags_warn_on_bad_thresholds() -> None:
    assert edge_quality_flags({"edge_density": 0.0, "max_magnitude": 0.0}) == [
        "very_sparse_edges",
        "zero_gradient_image",
    ]
    assert edge_quality_flags({"edge_density": 0.75, "max_magnitude": 1.0}) == [
        "very_dense_edges"
    ]


def test_edge_mask_thresholds_magnitude_image() -> None:
    edges = np.array([[0.0, 0.2], [0.5, 0.7]], dtype=np.float32)

    mask = edge_mask(edges, threshold=0.5)

    np.testing.assert_array_equal(mask, np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_component_statistics_detect_connected_and_isolated_edges() -> None:
    edges = np.zeros((5, 5), dtype=np.float32)
    edges[1, 1:4] = 1.0
    edges[4, 4] = 1.0

    report = edge_component_statistics(edges, threshold=0.5)

    assert report["components"] == 2
    assert report["largest_component_pixels"] == 3
    assert report["largest_component_fraction"] == pytest.approx(0.75)
    assert report["isolated_component_fraction"] == pytest.approx(0.5)


def test_edge_topology_comparison_detects_fragmentation() -> None:
    reference = np.zeros((5, 5), dtype=np.float32)
    candidate = np.zeros((5, 5), dtype=np.float32)
    reference[2, 1:4] = 1.0
    candidate[1, 1] = candidate[2, 3] = candidate[4, 4] = 1.0

    report = edge_topology_comparison(reference, candidate, threshold=0.5)

    assert report["reference_components"] == 1
    assert report["candidate_components"] == 3
    assert report["component_count_delta"] == 2
    assert report["isolated_component_fraction_gap"] > 0


def test_hysteresis_keeps_weak_edges_connected_to_strong_edges() -> None:
    edges = np.zeros((5, 6), dtype=np.float32)
    edges[2, 1:4] = [1.0, 0.6, 0.55]
    edges[4, 4] = 0.7

    mask = hysteresis_edge_mask(edges, high_threshold=0.9, low_ratio=0.5)

    np.testing.assert_array_equal(mask[2, 1:4], np.ones(3))
    assert mask[4, 4] == 0.0


def test_edge_statistics_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        edge_statistics(np.array([[0.0, np.inf]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        edge_statistics(np.ones((2, 2)), threshold=np.nan)


def test_edge_orientation_histogram_counts_active_edges() -> None:
    horizontal = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    vertical = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    magnitude = np.ones((2, 2), dtype=np.float32)

    rows = edge_orientation_histogram(horizontal, vertical, magnitude, threshold=0.5, bins=2)

    assert len(rows) == 2
    assert sum(row["count"] for row in rows) == 4
    assert sum(row["fraction"] for row in rows) == pytest.approx(1.0)


def test_non_maximum_suppression_keeps_local_gradient_maxima() -> None:
    horizontal = np.ones((5, 5), dtype=np.float32)
    vertical = np.zeros((5, 5), dtype=np.float32)
    magnitude = np.zeros((5, 5), dtype=np.float32)
    magnitude[2, 1:4] = [0.4, 1.0, 0.6]

    thinned = non_maximum_suppression(horizontal, vertical, magnitude)

    assert thinned[2, 2] == pytest.approx(1.0)
    assert thinned[2, 1] == 0.0
    assert thinned[2, 3] == 0.0


def test_nms_reduction_ratio_counts_removed_edges() -> None:
    before = np.array([[0.0, 0.5], [0.7, 0.9]], dtype=np.float32)
    after = np.array([[0.0, 0.0], [0.7, 0.0]], dtype=np.float32)

    assert nms_reduction_ratio(before, after, threshold=0.5) == pytest.approx(2 / 3)


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


def test_launch_shape_reports_grid_geometry() -> None:
    shape = launch_shape(height=33, width=65, block_x=16, block_y=8)

    assert shape["blocks_x"] == 5
    assert shape["blocks_y"] == 5
    assert shape["threads_per_block"] == 128
    assert shape["total_blocks"] == 25
    assert shape["scheduled_threads"] == 3200
    assert shape["useful_threads"] == 2145
    assert shape["boundary_threads"] == 1055
    assert shape["launch_utilization"] == pytest.approx(2145 / 3200)


def test_throughput_reports_megapixels_per_second() -> None:
    assert throughput_mpix_per_second(1000, 1000, 2.0) == pytest.approx(500.0)
    assert throughput_mpix_per_second(1000, 1000, 0.0) is None
    with pytest.raises(ValueError, match="height"):
        throughput_mpix_per_second(0, 100, 1.0)


def test_memory_bandwidth_reports_minimum_image_traffic() -> None:
    bandwidth = minimum_memory_bandwidth_gbps(
        1000,
        1000,
        elapsed_ms=2.0,
        bytes_per_pixel=4,
        arrays_read=1,
        arrays_written=1,
    )

    assert bandwidth == pytest.approx(4.0)
    assert minimum_memory_bandwidth_gbps(10, 10, 0.0) is None


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
    assert report["launch"]["threads_per_block"] == config.block_x * config.block_y
    assert report["launch"]["total_blocks"] == 4
    assert report["cpu_throughput_mpix_per_s"] > 0
    assert report["cpu_minimum_memory_bandwidth_gbps"] > 0
    assert report["edge_threshold"] == config.edge_threshold
    assert "edge_density" in report["edge_statistics"]
    assert report["threshold_sensitivity"]["low_density"] >= report["threshold_sensitivity"]["high_density"]
    assert report["edge_separation"]["edge_pixels"] >= 0
    assert report["edge_components"]["components"] >= 0
    assert isinstance(report["edge_quality_flags"], list)
    assert "mean_gradient_energy" in report["gradient_energy"]
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


def test_pipeline_reports_scale_space_stability(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=1,
        scale_check_sigma=1.2,
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["scale_space"]["additional_sigma"] == 1.2
    assert -1.0 <= report["scale_space"]["pearson_correlation"] <= 1.0


def test_pipeline_reports_noise_robustness(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=1,
        noise_check_std=0.02,
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["noise_robustness"]["noise_std"] == pytest.approx(0.02)
    assert 0.0 <= report["noise_robustness"]["binary_f1"] <= 1.0


def test_cpu_pipeline_reports_gradient_operator(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=1,
        gradient_operator="scharr",
        output_path=tmp_path / "edges.png",
        report_path=tmp_path / "run.json",
    )

    assert run(config, cpu_only=True)["gradient_operator"] == "scharr"


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


def test_pipeline_can_write_hysteresis_mask(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=1,
        hysteresis_ratio=0.5,
        output_path=tmp_path / "edges.png",
        mask_output_path=tmp_path / "mask.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["hysteresis_ratio"] == 0.5
    assert report["mask_edge_pixels"] >= 0


def test_cpu_pipeline_can_write_nms_edge_map(tmp_path) -> None:
    config = PipelineConfig(
        height=32,
        width=32,
        iterations=2,
        output_path=tmp_path / "edges.png",
        nms_output_path=tmp_path / "nms.png",
        report_path=tmp_path / "run.json",
    )

    report = run(config, cpu_only=True)

    assert report["nms_output"] == str(config.nms_output_path)
    assert "nms_edge_statistics" in report
    assert 0.0 <= report["nms_reduction_ratio"] <= 1.0
    assert config.nms_output_path.exists()


def test_config_rejects_oversized_blocks() -> None:
    with pytest.raises(ValueError, match="1024"):
        PipelineConfig(block_x=33, block_y=32)


def test_config_rejects_negative_edge_threshold() -> None:
    with pytest.raises(ValueError, match="edge_threshold"):
        PipelineConfig(edge_threshold=-0.1)


def test_config_rejects_negative_blur_sigma() -> None:
    with pytest.raises(ValueError, match="blur_sigma"):
        PipelineConfig(blur_sigma=-0.1)


def test_config_rejects_bad_noise_check_std() -> None:
    with pytest.raises(ValueError, match="noise_check_std"):
        PipelineConfig(noise_check_std=0.0)


def test_config_rejects_bad_edge_quantile() -> None:
    with pytest.raises(ValueError, match="edge_quantile"):
        PipelineConfig(edge_quantile=1.0)
    with pytest.raises(ValueError, match="edge_threshold"):
        PipelineConfig(edge_threshold=np.nan)


def test_cuda_availability_returns_a_boolean() -> None:
    assert isinstance(cuda_available(), bool)


def test_gpu_entrypoints_validate_inputs_before_device_check() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sobel_edges_gpu(np.zeros((8, 8, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="1024"):
        benchmark_gpu(np.zeros((8, 8), dtype=np.float32), 33, 32, 1)
    invalid = np.ones((8, 8), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sobel_edges_gpu(invalid)
