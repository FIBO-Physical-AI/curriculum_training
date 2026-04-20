from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MASTERY_THRESHOLD_DEFAULT = 0.7


@dataclass
class IterationsToMasteryLogger:
    num_bins: int
    out_dir: Path
    mastery_threshold: float = MASTERY_THRESHOLD_DEFAULT
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    first_crossing: list[int | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.first_crossing:
            self.first_crossing = [None] * self.num_bins

    def log(self, iteration: int, per_bin_return: np.ndarray) -> None:
        if per_bin_return.shape != (self.num_bins,):
            raise ValueError(f"expected ({self.num_bins},), got {per_bin_return.shape}")
        self.history.append((iteration, per_bin_return.copy()))
        for bin_idx in range(self.num_bins):
            if self.first_crossing[bin_idx] is None and per_bin_return[bin_idx] >= self.mastery_threshold:
                self.first_crossing[bin_idx] = iteration

    def result(self) -> dict[int, int | float]:
        return {i: (it if it is not None else math.inf) for i, it in enumerate(self.first_crossing)}

    def save(self) -> None:
        raise NotImplementedError


def iterations_to_mastery_from_curves(
    curves: np.ndarray,
    iterations: np.ndarray,
    mastery_threshold: float = MASTERY_THRESHOLD_DEFAULT,
) -> np.ndarray:
    if curves.ndim != 2 or curves.shape[0] != iterations.shape[0]:
        raise ValueError("curves must be (T, B) aligned with iterations (T,)")
    num_bins = curves.shape[1]
    out = np.full(num_bins, np.inf, dtype=np.float64)
    for bin_idx in range(num_bins):
        crossings = np.where(curves[:, bin_idx] >= mastery_threshold)[0]
        if crossings.size > 0:
            out[bin_idx] = iterations[crossings[0]]
    return out
