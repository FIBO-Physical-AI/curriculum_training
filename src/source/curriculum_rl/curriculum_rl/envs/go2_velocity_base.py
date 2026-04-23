from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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


def _apply_perf_trims(cfg) -> None:
    if hasattr(cfg.events, "push_robot"):
        cfg.events.push_robot = None
    if hasattr(cfg.events, "base_external_force_torque"):
        cfg.events.base_external_force_torque = None
    if hasattr(cfg.scene, "contact_forces") and cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.history_length = 1


def _remove_yaw_tracking_reward(cfg) -> None:
    if hasattr(cfg.rewards, "track_ang_vel_z"):
        cfg.rewards.track_ang_vel_z.weight = 0.0
    cfg.rewards.ang_vel_z_penalty = RewTerm(
        func=curriculum_mdp.ang_vel_z_l2,
        weight=-0.05,
    )


def _tune_feet_air_time_for_fast_gaits(cfg) -> None:
    if hasattr(cfg.rewards, "feet_air_time"):
        cfg.rewards.feet_air_time.params["threshold"] = 0.3


def _trim_velocity_command_obs(cfg) -> None:
    term = ObsTerm(
        func=curriculum_mdp.forward_velocity_command,
        clip=(-100, 100),
        params={"command_name": "base_velocity"},
    )
    if hasattr(cfg.observations, "policy") and hasattr(cfg.observations.policy, "velocity_commands"):
        cfg.observations.policy.velocity_commands = term
    if hasattr(cfg.observations, "critic") and hasattr(cfg.observations.critic, "velocity_commands"):
        cfg.observations.critic.velocity_commands = term


@configclass
class Go2VelocityBaseEnvCfg(RobotEnvCfg):
    curriculum_kind: str = "uniform"

    def __post_init__(self):
        super().__post_init__()
        _flatten_terrain(self)
        _remove_yaw_tracking_reward(self)
        _tune_feet_air_time_for_fast_gaits(self)
        _trim_velocity_command_obs(self)
        _apply_perf_trims(self)
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
        _remove_yaw_tracking_reward(self)
        _tune_feet_air_time_for_fast_gaits(self)
        _trim_velocity_command_obs(self)
        _apply_perf_trims(self)
        self.commands.base_velocity = _make_binned_cmd(self.curriculum_kind)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(self.curriculum, "lin_vel_cmd_levels"):
            self.curriculum.lin_vel_cmd_levels = None
