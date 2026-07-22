# Python/CUDA Sobel Pipeline

A Python image-processing pipeline with a custom CUDA Sobel edge detector. The project uses NumPy for the reference implementation, CuPy for CUDA memory and kernel execution, and Pillow for image input/output.

## Overview

The pipeline accepts a grayscale image or creates a deterministic synthetic image, computes Sobel edge magnitude on the CPU and GPU, compares the results, and writes an edge image with a JSON timing report.

The CUDA kernel assigns one thread to each pixel, uses a two-dimensional grid, and reads the 3×3 neighborhood required by the horizontal and vertical Sobel filters. The Python layer controls the input, launch dimensions, synchronization, benchmarking, and output handling.

## Requirements

- Python 3.10+
- NVIDIA GPU and CUDA Toolkit
- CUDA 12.x for the default `cupy-cuda12x` dependency

For a CUDA 11 installation, replace `cupy-cuda12x` with the matching CuPy package. The CPU path does not require CuPy or a GPU.

## Installation

```bash
cd python-cupy-sobel
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
  --output outputs/edges.png \
  --report reports/run.json
```

The CPU reference can be run without CuPy or a GPU:

```bash
python -m gpu_edges.pipeline \
  --cpu-only \
  --height 256 \
  --width 256
```

The report contains the input shape, block dimensions, CPU time, GPU kernel time, speedup, and CPU/GPU comparison error. Device transfers are outside the GPU timing region so the reported GPU value measures kernel execution.

## Project structure

```text
python-cupy-sobel/
├── src/gpu_edges/
│   ├── kernels/sobel.cu  # CUDA Sobel kernel
│   ├── cuda.py           # CuPy RawKernel bridge and timing
│   ├── cpu.py            # NumPy reference implementation
│   ├── data.py           # deterministic synthetic input
│   ├── io.py             # grayscale image loading and saving
│   ├── metrics.py        # CPU/GPU comparison metrics
│   └── pipeline.py       # command-line orchestration
├── tests/test_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```
