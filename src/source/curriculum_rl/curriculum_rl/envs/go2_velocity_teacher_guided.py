from __future__ import annotations

from isaaclab.utils import configclass

from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg, Go2VelocityBasePlayEnvCfg


@configclass
class TeacherGuidedCurriculumEnvCfg(Go2VelocityBaseEnvCfg):
    beta: float = 1.0
    stage_length: int = 100

    def __post_init__(self):
        super().__post_init__()


@configclass
class TeacherGuidedCurriculumPlayEnvCfg(Go2VelocityBasePlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
