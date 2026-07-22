from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def save_grayscale(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
