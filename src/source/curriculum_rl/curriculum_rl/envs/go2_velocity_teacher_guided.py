from __future__ import annotations

from isaaclab.utils import configclass

from curriculum_rl.envs.go2_velocity_base import Go2VelocityBaseEnvCfg, Go2VelocityBasePlayEnvCfg


@configclass
class TeacherGuidedCurriculumEnvCfg(Go2VelocityBaseEnvCfg):
    curriculum_kind: str = "teacher"


@configclass
class TeacherGuidedCurriculumPlayEnvCfg(Go2VelocityBasePlayEnvCfg):
    curriculum_kind: str = "teacher"
