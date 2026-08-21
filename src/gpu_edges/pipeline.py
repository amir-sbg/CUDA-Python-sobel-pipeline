from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .config import PipelineConfig
from .cpu import sobel_edges
from .cuda import benchmark_gpu, cuda_available
from .data import generate_image
from .io import load_grayscale, save_grayscale
from .metrics import comparison_metrics, edge_statistics, speedup_ratio


def _benchmark_cpu(image, iterations: int) -> tuple[object, float]:
    output = sobel_edges(image)
    start = perf_counter()
    for _ in range(iterations):
        output = sobel_edges(image)
    elapsed_ms = (perf_counter() - start) * 1000.0 / iterations
    return output, elapsed_ms


def _normalize(image):
    maximum = float(image.max())
    return image / maximum if maximum > 0 else image


def run(
    config: PipelineConfig,
    input_path: Path | None = None,
    cpu_only: bool = False,
) -> dict:
    image = load_grayscale(input_path) if input_path else generate_image(
        config.height,
        config.width,
        config.seed,
    )
    cpu_output, cpu_ms = _benchmark_cpu(image, config.iterations)
    report = {
        "backend": "cpu",
        "input_source": str(input_path) if input_path else "generated",
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "block": [config.block_x, config.block_y],
        "iterations": config.iterations,
        "cpu_average_ms": cpu_ms,
    }

    if not cpu_only and not cuda_available():
        raise RuntimeError("CUDA is unavailable; run with --cpu-only for a CPU check")
    if cpu_only:
        output = cpu_output
    else:
        output, gpu_ms = benchmark_gpu(
            image,
            config.block_x,
            config.block_y,
            config.iterations,
        )
        report["backend"] = "cuda"
        report["gpu_kernel_average_ms"] = gpu_ms
        report["speedup"] = speedup_ratio(cpu_ms, gpu_ms)
        report["comparison"] = comparison_metrics(cpu_output, output)

    report["edge_threshold"] = config.edge_threshold
    report["edge_statistics"] = edge_statistics(output, config.edge_threshold)
    save_grayscale(_normalize(output), config.output_path)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python/CUDA Sobel pipeline.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-x", type=int, default=16)
    parser.add_argument("--block-y", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--edge-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("outputs/sobel_edges.png"))
    parser.add_argument("--report", type=Path, default=Path("reports/run.json"))
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    values = vars(args).copy()
    values.pop("input", None)
    values.pop("cpu_only", None)
    values["output_path"] = values.pop("output")
    values["report_path"] = values.pop("report")
    return PipelineConfig(**values)


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        config_from_args(arguments),
        input_path=arguments.input,
        cpu_only=arguments.cpu_only,
    )
