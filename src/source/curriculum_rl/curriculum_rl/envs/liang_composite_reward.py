from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


W_DOF_POS_LIMITS = 10.0
W_ACTION_RATE = 0.1
W_UNDESIRED_CONTACTS = 1.0
W_FLAT_ORIENTATION = 2.5
W_BASE_LIN_VEL_Z = 2.0
W_BASE_ANG_VEL_XY = 0.05
W_FEET_SLIDE = 0.1


def composite_liang_energy_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    aux_undesired_contact_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
    ),
    aux_feet_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    aux_feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    sigma_v: float = 0.25,
    sigma_w: float = 0.25,
    alpha_ang: float = 0.5,
    sigma_en_x: float = 1000.0,
    sigma_en_z: float = 500.0,
    alpha_en: float = 1.0,
    eps: float = 0.1,
    r_aux_clip: float = 10.0,
    aux_undesired_threshold: float = 1.0,
    return_components: bool = False,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    v_b = asset.data.root_lin_vel_b
    w_b = asset.data.root_ang_vel_b
    v_x = v_b[:, 0]
    v_y = v_b[:, 1]
    w_z = w_b[:, 2]

    v_err_sq = (v_x - cmd[:, 0]) ** 2 + (v_y - cmd[:, 1]) ** 2
    R_lin = torch.exp(-v_err_sq / sigma_v)

    w_err_sq = (w_z - cmd[:, 2]) ** 2
    R_ang = torch.exp(-w_err_sq / sigma_w)

    R_motion = R_lin + alpha_ang * R_ang

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    power = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

    denom = torch.clamp(sigma_en_x * torch.abs(v_x) + sigma_en_z * torch.abs(w_z), min=eps)
    R_en = torch.exp(-power / denom)

    q_dof_pos_limits = mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
    q_action_rate = mdp.action_rate_l2(env)
    q_undesired_contacts = mdp.undesired_contacts(
        env, threshold=aux_undesired_threshold, sensor_cfg=aux_undesired_contact_cfg
    )
    q_flat_orientation = mdp.flat_orientation_l2(env, asset_cfg=asset_cfg)
    q_base_lin_vel_z = mdp.lin_vel_z_l2(env, asset_cfg=asset_cfg)
    q_base_ang_vel_xy = mdp.ang_vel_xy_l2(env, asset_cfg=asset_cfg)
    q_feet_slide = mdp.feet_slide(env, asset_cfg=aux_feet_asset_cfg, sensor_cfg=aux_feet_sensor_cfg)

    R_aux = (
        W_DOF_POS_LIMITS * q_dof_pos_limits
        + W_ACTION_RATE * q_action_rate
        + W_UNDESIRED_CONTACTS * q_undesired_contacts.float()
        + W_FLAT_ORIENTATION * q_flat_orientation
        + W_BASE_LIN_VEL_Z * q_base_lin_vel_z
        + W_BASE_ANG_VEL_XY * q_base_ang_vel_xy
        + W_FEET_SLIDE * q_feet_slide
    )
    R_aux = torch.clamp(R_aux, min=0.0, max=r_aux_clip)

    R = (R_motion + alpha_en * R_en) * torch.exp(-R_aux)

    if return_components:
        return {
            "R": R,
            "R_motion": R_motion,
            "R_lin": R_lin,
            "R_ang": R_ang,
            "R_en": R_en,
            "R_aux": R_aux,
            "denom": denom,
            "power": power,
        }
    return R
