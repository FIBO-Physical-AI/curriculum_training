from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CurriculumBase(ABC):
    def __init__(self, num_bins: int, v_max: float):
        self.num_bins = num_bins
        self.v_max = v_max
        self.bin_width = v_max / num_bins
        self.weights: np.ndarray = np.ones(num_bins, dtype=np.float64) / num_bins

    @property
    def bin_centers(self) -> np.ndarray:
        return (np.arange(self.num_bins) + 0.5) * self.bin_width

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        bins = rng.choice(self.num_bins, size=n, p=self._normalized_weights())
        return self.bin_centers[bins]

    def _normalized_weights(self) -> np.ndarray:
        total = float(self.weights.sum())
        return self.weights / total if total > 0 else np.ones_like(self.weights) / self.num_bins

    @abstractmethod
    def update(self, bin_rewards: np.ndarray, step: int) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_bins": self.num_bins,
            "v_max": self.v_max,
            "weights": self.weights.tolist(),
        }
