from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg,
    RobotPlayEnvCfg,
)

from curriculum_rl.envs import mdp as curriculum_mdp
from curriculum_rl.envs.commands import BinnedVelocityCommandCfg


V_MAX = 4.0
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


def _apply_sprint_retune(cfg) -> None:
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.5
    cfg.rewards.action_rate.weight = -0.005
    cfg.rewards.joint_acc.weight = -1e-7
    cfg.rewards.joint_torques.weight = -2e-5
    cfg.rewards.joint_vel.weight = -1e-4
    cfg.rewards.feet_air_time.params["threshold"] = 0.1
    cfg.actions.JointPositionAction.scale = 0.35


def _apply_play_camera(cfg) -> None:
    if hasattr(cfg, "viewer"):
        cfg.viewer.eye = (1.6, 1.4, 0.25)
        cfg.viewer.lookat = (0.0, 0.0, 0.15)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.resolution = (1280, 720)


def _lock_play_pose(cfg) -> None:
    if hasattr(cfg.events, "reset_base"):
        cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }


@configclass
class Go2VelocityBaseEnvCfg(RobotEnvCfg):
    curriculum_kind: str = "uniform"

    def __post_init__(self):
        super().__post_init__()
        _flatten_terrain(self)
        _apply_sprint_retune(self)
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
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
        _apply_sprint_retune(self)
        _lock_play_pose(self)
        _apply_play_camera(self)
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.commands.base_velocity = _make_binned_cmd(self.curriculum_kind)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(self.curriculum, "lin_vel_cmd_levels"):
            self.curriculum.lin_vel_cmd_levels = None
