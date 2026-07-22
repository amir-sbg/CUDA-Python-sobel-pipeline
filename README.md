# CUDA Python Projects

This repository contains a Python image-processing pipeline that combines NumPy, CuPy, and a custom CUDA Sobel kernel.

## Project

See [`python-cupy-sobel/`](python-cupy-sobel/) for installation, usage, the CUDA kernel implementation, CPU reference path, benchmarking, and tests.

```bash
cd python-cupy-sobel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m gpu_edges.pipeline
```
