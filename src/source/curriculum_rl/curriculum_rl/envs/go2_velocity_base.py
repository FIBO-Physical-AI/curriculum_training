from __future__ import annotations

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg,
    RobotPlayEnvCfg,
)

V_MAX = 3.5
NUM_BINS = 7
BIN_WIDTH = V_MAX / NUM_BINS


@configclass
class Go2VelocityBaseEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, V_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, V_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class Go2VelocityBasePlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, V_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
