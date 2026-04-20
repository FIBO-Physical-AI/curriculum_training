from __future__ import annotations

import numpy as np

from curriculum_rl.curricula.base import CurriculumBase


class TeacherGuidedCurriculum(CurriculumBase):
    def __init__(
        self,
        num_bins: int,
        v_max: float,
        beta: float = 1.0,
        stage_length: int = 100,
    ):
        super().__init__(num_bins=num_bins, v_max=v_max)
        self.beta = beta
        self.stage_length = stage_length
        self.prev_rewards: np.ndarray = np.zeros(num_bins, dtype=np.float64)
        self.last_update_step: int = 0

    def update(self, bin_rewards: np.ndarray, step: int) -> None:
        raise NotImplementedError
