from __future__ import annotations

import os

import numpy as np

from curriculum_rl.curricula.base import CurriculumBase


def _steps_per_ppo_iteration() -> int:
    val = os.environ.get("CURRICULUM_STEPS_PER_ITER")
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            pass
    return 48


class TeacherGuidedCurriculum(CurriculumBase):
    def __init__(
        self,
        num_bins: int,
        v_max: float,
        beta: float = 0.05,
        stage_length: int = 50,
        eps: float = 0.05,
        seed_bin: int = 0,
    ):
        super().__init__(num_bins=num_bins, v_max=v_max)
        self.beta = beta
        self.stage_length = stage_length
        self.eps = eps
        self.seed_bin = int(seed_bin)
        self.weights = np.zeros(num_bins, dtype=np.float64)
        self.weights[self.seed_bin] = 1.0
        if self.seed_bin + 1 < num_bins:
            self.weights[self.seed_bin + 1] = 0.5
        self.weights /= self.weights.sum()
        self.prev_rewards: np.ndarray = np.zeros(num_bins, dtype=np.float64)
        self.last_update_step: int = -1
        self._stage_sum: np.ndarray = np.zeros(num_bins, dtype=np.float64)
        self._stage_count: int = 0
        self._warmed_up: bool = False

    def update(self, bin_rewards: np.ndarray, step: int) -> None:
        if bin_rewards.shape != (self.num_bins,):
            raise ValueError(f"expected ({self.num_bins},), got {bin_rewards.shape}")
        self._stage_sum += bin_rewards
        self._stage_count += 1
        if self.last_update_step < 0:
            self.last_update_step = step
            return
        trigger_step = self.last_update_step + self.stage_length * _steps_per_ppo_iteration()
        if step < trigger_step:
            return
        stage_avg = self._stage_sum / max(self._stage_count, 1)
        self._stage_sum[:] = 0.0
        self._stage_count = 0
        self.last_update_step = step
        if not self._warmed_up:
            self.prev_rewards = stage_avg.copy()
            self._warmed_up = True
            return
        lp = np.maximum(stage_avg - self.prev_rewards, 0.0)
        scaled = lp / max(self.beta, 1e-8)
        scaled = scaled - scaled.max()
        exp = np.exp(scaled)
        softmax = exp / exp.sum()
        uniform = np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float64)
        self.weights = (1.0 - self.eps) * softmax + self.eps * uniform
        self.prev_rewards = stage_avg.copy()
