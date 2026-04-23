from __future__ import annotations

import numpy as np

from curriculum_rl.curricula.base import CurriculumBase


class TaskSpecificCurriculum(CurriculumBase):
    def __init__(
        self,
        num_bins: int,
        v_max: float,
        gamma: float = 0.7,
        seed_bin: int = 0,
        min_episodes_per_bin: int = 50,
    ):
        super().__init__(num_bins=num_bins, v_max=v_max)
        self.gamma = gamma
        self.seed_bin = seed_bin
        self.min_episodes_per_bin = int(min_episodes_per_bin)
        self.weights = np.zeros(num_bins, dtype=np.float64)
        self.weights[seed_bin] = 1.0
        self.fired: np.ndarray = np.zeros(num_bins, dtype=bool)
        self._episode_counts: np.ndarray = np.zeros(num_bins, dtype=np.int64)

    def update(self, bin_rewards: np.ndarray, step: int, bin_counts: np.ndarray | None = None) -> None:
        if bin_rewards.shape != (self.num_bins,):
            raise ValueError(f"expected ({self.num_bins},), got {bin_rewards.shape}")
        if bin_counts is not None:
            self._episode_counts += bin_counts.astype(np.int64)
        new_weights = self.weights.copy()
        for b in range(self.num_bins):
            if self.weights[b] <= 0.0 or self.fired[b]:
                continue
            if self._episode_counts[b] < self.min_episodes_per_bin:
                continue
            if bin_rewards[b] >= self.gamma:
                self.fired[b] = True
                if b - 1 >= 0:
                    new_weights[b - 1] = 1.0
                if b + 1 < self.num_bins:
                    new_weights[b + 1] = 1.0
        self.weights = new_weights
