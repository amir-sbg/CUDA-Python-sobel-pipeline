# Python/CUDA Sobel Pipeline

A small, reproducible edge-detection pipeline with matching NumPy and CUDA implementations. It is useful for checking a custom GPU kernel against a reference implementation while keeping the image-processing and benchmarking steps easy to inspect.

The pipeline can generate a deterministic test image or read a grayscale image, optionally smooth it with a Gaussian filter, run the Sobel operator, and save the edge map. On CUDA systems, CuPy launches one thread per pixel; the CPU path remains available for development and comparison.

## What it reports

- CPU and GPU kernel time, throughput, and speedup
- Maximum error and RMSE between CPU and CUDA magnitudes
- Binary edge precision, recall, F1, and IoU
- Edge density, orientation histogram, and optional non-maximum suppression output

## Setup

Python 3.10+ is required. CUDA 12.x systems use `cupy-cuda12x`; the CPU path does not require a GPU.

```bash
git clone https://github.com/amir-sbg/CUDA-Python-sobel-pipeline.git
cd CUDA-Python-sobel-pipeline
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For CUDA 11, install the matching CuPy package instead. Run the test suite with:

```bash
python -m pytest -q
```

## Usage

Run a CPU-only check with a deterministic synthetic image:

```bash
python -m gpu_edges.pipeline --cpu-only --height 256 --width 256
```

Run the CUDA benchmark and write the edge map plus JSON report:

```bash
python -m gpu_edges.pipeline \
  --input path/to/image.png \
  --block-x 16 --block-y 16 --iterations 50 \
  --blur-sigma 0.8 --edge-quantile 0.85 \
  --output outputs/edges.png \
  --mask-output outputs/edge_mask.png \
  --nms-output outputs/edges_nms.png \
  --report reports/run.json
```

Use `--edge-threshold` for a fixed cutoff. The adaptive quantile option is helpful when images have different contrast levels. GPU timing covers kernel execution only; host-device transfers are outside the measured region.

## Layout

```text
src/gpu_edges/
├── kernels/sobel.cu  # CUDA kernel
├── cpu.py             # NumPy reference and gradient components
├── cuda.py            # CuPy kernel launch and timing
├── filters.py         # Gaussian prefilter
├── metrics.py         # Error, edge, and orientation metrics
├── io.py              # Grayscale image I/O
└── pipeline.py        # Command-line entry point
tests/test_pipeline.py
```
