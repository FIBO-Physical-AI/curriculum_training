from __future__ import annotations

import numpy as np

from curriculum_rl.curricula.base import CurriculumBase


class UniformCurriculum(CurriculumBase):
    def __init__(self, num_bins: int, v_max: float):
        super().__init__(num_bins=num_bins, v_max=v_max)

    def update(self, bin_rewards: np.ndarray, step: int) -> None:
        return
