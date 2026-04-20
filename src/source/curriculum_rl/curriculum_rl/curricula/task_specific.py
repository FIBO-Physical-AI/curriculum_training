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
    ):
        super().__init__(num_bins=num_bins, v_max=v_max)
        self.gamma = gamma
        self.seed_bin = seed_bin
        self.weights = np.zeros(num_bins, dtype=np.float64)
        self.weights[seed_bin] = 1.0

    def update(self, bin_rewards: np.ndarray, step: int) -> None:
        raise NotImplementedError
