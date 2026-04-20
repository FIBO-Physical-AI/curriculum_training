from __future__ import annotations

from pathlib import Path

import numpy as np


class PerBinReturnLogger:
    def __init__(self, num_bins: int, out_dir: Path):
        self.num_bins = num_bins
        self.out_dir = Path(out_dir)
        self.history: list[tuple[int, np.ndarray]] = []

    def log(self, iteration: int, per_bin_return: np.ndarray) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError
