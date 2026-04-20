from __future__ import annotations

from isaaclab.utils import configclass

from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg, Go2VelocityBasePlayEnvCfg


@configclass
class TaskSpecificCurriculumEnvCfg(Go2VelocityBaseEnvCfg):
    gamma: float = 0.7
    seed_bin: int = 0

    def __post_init__(self):
        super().__post_init__()


@configclass
class TaskSpecificCurriculumPlayEnvCfg(Go2VelocityBasePlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
