from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    height: int = 1024
    width: int = 1024
    seed: int = 7
    block_x: int = 16
    block_y: int = 16
    iterations: int = 50
    output_path: Path = Path("outputs/sobel_edges.png")
    report_path: Path = Path("reports/run.json")

    def __post_init__(self) -> None:
        if self.height < 3 or self.width < 3:
            raise ValueError("height and width must be at least 3")
        if self.block_x < 1 or self.block_y < 1:
            raise ValueError("CUDA block dimensions must be positive")
        if self.block_x * self.block_y > 1024:
            raise ValueError("CUDA blocks cannot contain more than 1024 threads")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
