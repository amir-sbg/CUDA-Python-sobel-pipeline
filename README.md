# Python/CUDA Sobel Pipeline

A Python image-processing pipeline with a custom CUDA Sobel edge detector. The project uses NumPy for the reference implementation, CuPy for CUDA memory and kernel execution, and Pillow for image input/output.

## Overview

The pipeline accepts a grayscale image or creates a deterministic synthetic image, can apply a small Gaussian prefilter, computes Sobel edge magnitude on the CPU and GPU, compares the results, and writes edge artifacts with a JSON timing report.

The CUDA kernel assigns one thread to each pixel, uses a two-dimensional grid, and reads the 3×3 neighborhood required by the horizontal and vertical Sobel filters. The Python layer controls the input, launch dimensions, synchronization, benchmarking, and output handling.

## Requirements

- Python 3.10+
- NVIDIA GPU and CUDA Toolkit
- CUDA 12.x for the default `cupy-cuda12x` dependency

For a CUDA 11 installation, replace `cupy-cuda12x` with the matching CuPy package. The CPU path does not require CuPy or a GPU.

## Installation

```bash
git clone https://github.com/amir-sbg/cuda.git
cd cuda
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Run

Run the CUDA pipeline with a generated 1024×1024 image:

```bash
python -m gpu_edges.pipeline
```

Use an input image and choose the CUDA launch configuration explicitly:

```bash
python -m gpu_edges.pipeline \
  --input path/to/image.png \
  --block-x 16 \
  --block-y 16 \
  --iterations 50 \
  --blur-sigma 0.8 \
  --edge-threshold 0.20 \
  --edge-quantile 0.85 \
  --output outputs/edges.png \
  --mask-output outputs/edge_mask.png \
  --nms-output outputs/edges_nms.png \
  --report reports/run.json
```

The CPU reference can be run without CuPy or a GPU:

```bash
python -m gpu_edges.pipeline \
  --cpu-only \
  --height 256 \
  --width 256
```

The report contains the input shape, CUDA block/grid geometry, CPU time, GPU kernel time, megapixels/sec throughput, speedup, CPU/GPU comparison error, and edge statistics such as mean magnitude, edge density, and an orientation histogram. Use `--blur-sigma` when the input is noisy, `--edge-threshold` for a fixed magnitude cutoff, or `--edge-quantile` to choose the cutoff from the output distribution. Add `--mask-output` for a binary edge map and `--nms-output` for a thinned non-maximum-suppressed edge map when the output is feeding a CV/ML preprocessing stage. Device transfers are outside the GPU timing region so the reported GPU value measures kernel execution. On very small inputs, `speedup` may be `null` if the CUDA event timer reports a zero-duration kernel average.

## Project structure

```text
.
├── src/gpu_edges/
│   ├── kernels/sobel.cu  # CUDA Sobel kernel
│   ├── cuda.py           # CuPy RawKernel bridge and timing
│   ├── cpu.py            # NumPy reference implementation and Sobel components
│   ├── data.py           # deterministic synthetic input
│   ├── filters.py        # Gaussian prefiltering
│   ├── io.py             # grayscale image loading and saving
│   ├── metrics.py        # CPU/GPU comparison, NMS, and edge-analysis metrics
│   └── pipeline.py       # command-line orchestration
├── tests/test_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```
