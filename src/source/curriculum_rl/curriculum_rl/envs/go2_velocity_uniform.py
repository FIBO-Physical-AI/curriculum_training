from __future__ import annotations

from isaaclab.utils import configclass

from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg, Go2VelocityBasePlayEnvCfg


@configclass
class UniformCurriculumEnvCfg(Go2VelocityBaseEnvCfg):
    curriculum_kind: str = "uniform"


@configclass
class UniformCurriculumPlayEnvCfg(Go2VelocityBasePlayEnvCfg):
    curriculum_kind: str = "uniform"
