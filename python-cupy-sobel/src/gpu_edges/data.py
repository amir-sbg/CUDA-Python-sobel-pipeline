from __future__ import annotations

import numpy as np


def generate_image(height: int, width: int, seed: int = 7) -> np.ndarray:
    if height < 3 or width < 3:
        raise ValueError("height and width must be at least 3")
    generator = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)
    image = 0.15 + 0.35 * x + 0.20 * y
    image += 0.40 * (((x - 0.30) ** 2 + (y - 0.35) ** 2) < 0.12**2)
    image += 0.35 * (((x - 0.70) ** 2 + (y - 0.65) ** 2) < 0.18**2)
    image += generator.normal(0.0, 0.015, size=(height, width))
    return np.clip(image, 0.0, 1.0).astype(np.float32)
