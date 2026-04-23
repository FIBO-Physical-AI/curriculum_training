from __future__ import annotations

import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.mdp.commands import UniformLevelVelocityCommandCfg

from curriculum_rl.curricula.base import CurriculumBase
from curriculum_rl.curricula.task_specific import TaskSpecificCurriculum
from curriculum_rl.curricula.teacher_guided import TeacherGuidedCurriculum
from curriculum_rl.curricula.uniform import UniformCurriculum

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


CURRICULUM_KINDS = ("uniform", "task_specific", "teacher")


class BinnedVelocityCommand(UniformVelocityCommand):
    cfg: "BinnedVelocityCommandCfg"

    def __init__(self, cfg: "BinnedVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.num_bins = cfg.num_bins
        self.v_max = cfg.v_max
        self.bin_width = cfg.v_max / cfg.num_bins
        self.bin_centers = (torch.arange(self.num_bins, device=self.device) + 0.5) * self.bin_width
        self.env_bin_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curriculum: CurriculumBase = self._build_curriculum(cfg)
        self._weights_gpu = torch.from_numpy(np.ascontiguousarray(self.curriculum.weights)).float().to(self.device)

    def _build_curriculum(self, cfg: "BinnedVelocityCommandCfg") -> CurriculumBase:
        if cfg.curriculum_kind == "uniform":
            return UniformCurriculum(num_bins=cfg.num_bins, v_max=cfg.v_max)
        if cfg.curriculum_kind == "task_specific":
            return TaskSpecificCurriculum(
                num_bins=cfg.num_bins,
                v_max=cfg.v_max,
                gamma=cfg.gamma,
                seed_bin=cfg.seed_bin,
                min_episodes_per_bin=cfg.min_episodes_per_bin,
            )
        if cfg.curriculum_kind == "teacher":
            return TeacherGuidedCurriculum(
                num_bins=cfg.num_bins,
                v_max=cfg.v_max,
                beta=cfg.beta,
                stage_length=cfg.stage_length,
                eps=cfg.eps,
                seed_bin=cfg.seed_bin,
            )
        raise ValueError(f"unknown curriculum_kind: {cfg.curriculum_kind}")

    @property
    def weights(self) -> torch.Tensor:
        return self._weights_gpu

    def sync_weights_from_curriculum(self) -> None:
        self._weights_gpu = torch.from_numpy(np.ascontiguousarray(self.curriculum.weights)).float().to(self.device)

    def _resample_command(self, env_ids):
        n = len(env_ids)
        forced = os.environ.get("CURRICULUM_PLAY_BIN")
        if forced is not None and forced != "":
            b = max(0, min(self.num_bins - 1, int(forced)))
            bin_idx = torch.full((n,), b, dtype=torch.long, device=self.device)
        else:
            w = self._weights_gpu.clamp(min=1e-12)
            probs = w / w.sum()
            bin_idx = torch.multinomial(probs, n, replacement=True)
        self.env_bin_idx[env_ids] = bin_idx
        self.vel_command_b[env_ids, 0] = self.bin_centers[bin_idx]
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0
        r = torch.empty(n, device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


@configclass
class BinnedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = BinnedVelocityCommand
    num_bins: int = MISSING
    v_max: float = MISSING
    curriculum_kind: str = "uniform"
    gamma: float = 0.7
    seed_bin: int = 0
    min_episodes_per_bin: int = 50
    beta: float = 0.05
    stage_length: int = 50
    eps: float = 0.05
