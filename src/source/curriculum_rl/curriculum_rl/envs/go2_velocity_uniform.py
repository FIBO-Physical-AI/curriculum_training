from __future__ import annotations

from isaaclab.utils import configclass

from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg, Go2VelocityBasePlayEnvCfg


@configclass
class UniformCurriculumEnvCfg(Go2VelocityBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class UniformCurriculumPlayEnvCfg(Go2VelocityBasePlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
