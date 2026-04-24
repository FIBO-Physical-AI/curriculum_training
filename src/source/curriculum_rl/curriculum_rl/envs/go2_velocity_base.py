from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg,
    RobotPlayEnvCfg,
)

from curriculum_rl.envs import mdp as curriculum_mdp
from curriculum_rl.envs.commands import BinnedVelocityCommandCfg


V_MAX = 3.0
NUM_BINS = 8
BIN_WIDTH = V_MAX / NUM_BINS


def _make_binned_cmd(curriculum_kind: str, **kwargs) -> BinnedVelocityCommandCfg:
    return BinnedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=BinnedVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, V_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=BinnedVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, V_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        num_bins=NUM_BINS,
        v_max=V_MAX,
        curriculum_kind=curriculum_kind,
        **kwargs,
    )


def _flatten_terrain(cfg) -> None:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    if hasattr(cfg.curriculum, "terrain_levels"):
        cfg.curriculum.terrain_levels = None


@configclass
class Go2VelocityBaseEnvCfg(RobotEnvCfg):
    curriculum_kind: str = "uniform"

    def __post_init__(self):
        super().__post_init__()
        _flatten_terrain(self)
        self.commands.base_velocity = _make_binned_cmd(self.curriculum_kind)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.velocity_curriculum = CurrTerm(
            func=curriculum_mdp.velocity_curriculum_step,
            params={"command_name": "base_velocity", "reward_term_name": "track_lin_vel_xy"},
        )


@configclass
class Go2VelocityBasePlayEnvCfg(RobotPlayEnvCfg):
    curriculum_kind: str = "uniform"

    def __post_init__(self):
        super().__post_init__()
        _flatten_terrain(self)
        self.commands.base_velocity = _make_binned_cmd(self.curriculum_kind)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(self.curriculum, "lin_vel_cmd_levels"):
            self.curriculum.lin_vel_cmd_levels = None
