from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--num_stab_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import math

import gymnasium as gym
import torch

import curriculum_rl  # noqa: F401
from curriculum_rl.envs.liang_composite_reward import (
    W_DOF_POS_LIMITS,
    W_ACTION_RATE,
    W_UNDESIRED_CONTACTS,
    W_FLAT_ORIENTATION,
    W_BASE_LIN_VEL_Z,
    W_BASE_ANG_VEL_XY,
    W_FEET_SLIDE,
    composite_liang_energy_reward,
)


TASK = "Curriculum-Go2-Velocity-Uniform"


def _check(label: str, ok: bool, detail: str = "") -> int:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}{(' :: ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    failures = 0

    env_cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]
    module_path, _, cls_name = env_cfg.rpartition(":")
    cfg_module = __import__(module_path, fromlist=[cls_name])
    cfg = getattr(cfg_module, cls_name)()
    cfg.scene.num_envs = args.num_envs

    print("=" * 70)
    print("V1 / V2: static cfg checks")
    print("=" * 70)

    failures += _check(
        "V1 import + signature",
        callable(composite_liang_energy_reward),
        f"function: {composite_liang_energy_reward.__qualname__}",
    )

    rew = cfg.rewards
    failures += _check(
        "V2 composite_liang.weight == 1.0",
        rew.composite_liang.weight == 1.0,
        f"got {rew.composite_liang.weight}",
    )
    failures += _check(
        "V2 track_lin_vel_xy.weight == 1e-8",
        rew.track_lin_vel_xy.weight == 1e-8,
        f"got {rew.track_lin_vel_xy.weight}",
    )
    failures += _check(
        "V2 track_ang_vel_z.weight == 1e-8",
        rew.track_ang_vel_z.weight == 1e-8,
        f"got {rew.track_ang_vel_z.weight}",
    )
    for name in (
        "feet_air_time",
        "air_time_variance",
        "joint_vel",
        "joint_acc",
        "joint_torques",
        "joint_pos",
        "energy",
    ):
        w = getattr(rew, name).weight
        failures += _check(f"V2 dropped: {name}.weight == 0.0", w == 0.0, f"got {w}")
    print()
    print("R_aux additive cfg weights (must be 0.0 - composite computes R_aux internally):")
    for name in (
        "dof_pos_limits",
        "action_rate",
        "undesired_contacts",
        "flat_orientation_l2",
        "base_linear_velocity",
        "base_angular_velocity",
        "feet_slide",
    ):
        w = getattr(rew, name).weight
        failures += _check(f"V2 R_aux additive: {name}.weight == 0.0", w == 0.0, f"got {w}")
    print()
    print("R_aux internal weights (used by composite for R_aux sum):")
    for name, val in (
        ("W_DOF_POS_LIMITS", W_DOF_POS_LIMITS),
        ("W_ACTION_RATE", W_ACTION_RATE),
        ("W_UNDESIRED_CONTACTS", W_UNDESIRED_CONTACTS),
        ("W_FLAT_ORIENTATION", W_FLAT_ORIENTATION),
        ("W_BASE_LIN_VEL_Z", W_BASE_LIN_VEL_Z),
        ("W_BASE_ANG_VEL_XY", W_BASE_ANG_VEL_XY),
        ("W_FEET_SLIDE", W_FEET_SLIDE),
    ):
        print(f"  {name:25s} = {val:>10g}")

    print()
    print("=" * 70)
    print("V3 - V12: runtime checks")
    print("=" * 70)

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    env.reset(seed=0)

    inner = env.unwrapped
    num_envs = inner.num_envs
    device = inner.device
    action_space = env.action_space
    act_dim = action_space.shape[-1]

    R_motion_max = 1.5 + 1e-6
    R_en_max = 1.0 + 1e-6
    r_aux_clip = 10.0

    R_motion_log, R_en_log, R_aux_log, R_log = [], [], [], []
    nan_seen = False

    print(f"Stepping {args.num_steps} steps with random actions on {num_envs} envs")
    for step in range(args.num_steps):
        action = (torch.rand(num_envs, act_dim, device=device) * 2 - 1).clamp(-1, 1)
        env.step(action)

        comps = composite_liang_energy_reward(inner, return_components=True)
        R = comps["R"]
        R_motion = comps["R_motion"]
        R_en = comps["R_en"]
        R_aux = comps["R_aux"]

        if torch.any(R < 0):
            failures += _check("V3 R >= 0", False, f"step {step}: min(R)={R.min().item()}")
            break
        if torch.any(R_motion > R_motion_max) or torch.any(R_motion < 0):
            failures += _check(
                "V4 R_motion in [0, 1.5]", False,
                f"step {step}: min={R_motion.min().item()} max={R_motion.max().item()}",
            )
            break
        if torch.any(R_en > R_en_max) or torch.any(R_en < 0):
            failures += _check(
                "V5 R_en in [0, 1]", False,
                f"step {step}: min={R_en.min().item()} max={R_en.max().item()}",
            )
            break
        if torch.any(R_aux < 0) or torch.any(R_aux > r_aux_clip + 1e-6):
            failures += _check(
                "V6 R_aux in [0, clip]", False,
                f"step {step}: min={R_aux.min().item()} max={R_aux.max().item()}",
            )
            break
        if (
            torch.isnan(R).any()
            or torch.isnan(R_motion).any()
            or torch.isnan(R_en).any()
            or torch.isnan(R_aux).any()
        ):
            nan_seen = True
            break

        R_motion_log.append(R_motion.detach().clone())
        R_en_log.append(R_en.detach().clone())
        R_aux_log.append(R_aux.detach().clone())
        R_log.append(R.detach().clone())
    else:
        failures += _check("V3 R >= 0 (all steps)", True)
        failures += _check("V4 R_motion in [0, 1.5] (all steps)", True)
        failures += _check("V5 R_en in [0, 1] (all steps)", True)
        failures += _check("V6 R_aux in [0, 10] (all steps)", True)

    if R_motion_log:
        Rm = torch.stack(R_motion_log)
        Re = torch.stack(R_en_log)
        Ra = torch.stack(R_aux_log)
        Rt = torch.stack(R_log)
        print()
        print("V7 component summaries (mean / max over runtime window):")
        print(f"  R_motion : mean={Rm.mean().item():.4f}   max={Rm.max().item():.4f}")
        print(f"  R_en     : mean={Re.mean().item():.4f}   max={Re.max().item():.4f}")
        print(f"  R_aux    : mean={Ra.mean().item():.4f}   max={Ra.max().item():.4f}")
        print(f"  R        : mean={Rt.mean().item():.4f}   max={Rt.max().item():.4f}")

    print()
    print("V8 standing-still / zero-cmd edge case (env 0):")
    asset = inner.scene["robot"]
    cmd = inner.command_manager.get_command("base_velocity")
    cmd[0] = 0.0
    asset.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=device), env_ids=torch.tensor([0], device=device)
    )
    inner.sim.forward()
    comps_v8 = composite_liang_energy_reward(inner, return_components=True)
    print(f"  R_lin    [env 0] = {comps_v8['R_lin'][0].item():.6f}  (expect ~1)")
    print(f"  R_ang    [env 0] = {comps_v8['R_ang'][0].item():.6f}  (expect ~1)")
    print(f"  R_motion [env 0] = {comps_v8['R_motion'][0].item():.6f}  (expect ~1.5)")
    print(f"  denom    [env 0] = {comps_v8['denom'][0].item():.6f}  (expect 0.1 floor)")
    print(f"  power    [env 0] = {comps_v8['power'][0].item():.6f}")
    print(f"  R_en     [env 0] = {comps_v8['R_en'][0].item():.6f}")
    print(f"  R        [env 0] = {comps_v8['R'][0].item():.6f}")

    print()
    print("V9 sprint-cmd / zero-motion edge case (env 0):")
    cmd = inner.command_manager.get_command("base_velocity")
    cmd[0, 0] = 4.0
    cmd[0, 1] = 0.0
    cmd[0, 2] = 0.0
    asset.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=device), env_ids=torch.tensor([0], device=device)
    )
    inner.sim.forward()
    comps_v9 = composite_liang_energy_reward(inner, return_components=True)
    print(f"  R_lin    [env 0] = {comps_v9['R_lin'][0].item():.6e}  (expect ~0, exp(-64))")
    print(f"  R_ang    [env 0] = {comps_v9['R_ang'][0].item():.6f}  (expect ~1)")
    print(f"  R_motion [env 0] = {comps_v9['R_motion'][0].item():.6f}  (expect ~0.5)")
    print(f"  R        [env 0] = {comps_v9['R'][0].item():.6f}")

    print()
    print("V10 hardware-violation case: forcing joint past limit on env 0")
    soft_limits = asset.data.soft_joint_pos_limits
    new_pos = asset.data.joint_pos.clone()
    new_pos[0, 0] = soft_limits[0, 0, 1].item() + 0.5
    new_vel = asset.data.joint_vel.clone()
    asset.write_joint_state_to_sim(
        position=new_pos[:1].clone(),
        velocity=new_vel[:1].clone(),
        env_ids=torch.tensor([0], device=device),
    )
    inner.sim.forward()
    comps_v10 = composite_liang_energy_reward(inner, return_components=True)
    print(f"  R_aux [env 0] = {comps_v10['R_aux'][0].item():.4f}  (expect > baseline)")
    print(f"  R     [env 0] = {comps_v10['R'][0].item():.4f}  (expect attenuated)")

    print()
    print(f"V11 / V12 stability sweep: {args.num_stab_steps} random steps")
    floor_hits = 0
    floor_total = 0
    nan_count = 0
    for step in range(args.num_stab_steps):
        action = (torch.rand(num_envs, act_dim, device=device) * 2 - 1).clamp(-1, 1)
        env.step(action)
        comps = composite_liang_energy_reward(inner, return_components=True)
        R = comps["R"]
        R_motion = comps["R_motion"]
        R_en = comps["R_en"]
        R_aux = comps["R_aux"]
        denom = comps["denom"]
        if (
            torch.isnan(R).any() or torch.isinf(R).any()
            or torch.isnan(R_motion).any() or torch.isinf(R_motion).any()
            or torch.isnan(R_en).any() or torch.isinf(R_en).any()
            or torch.isnan(R_aux).any() or torch.isinf(R_aux).any()
        ):
            nan_count += 1
        floor_hits += (denom <= 0.1 + 1e-9).sum().item()
        floor_total += denom.numel()

    failures += _check("V11 no NaN/Inf in 1000 steps", nan_count == 0, f"{nan_count} steps with NaN/Inf")
    frac = floor_hits / max(floor_total, 1)
    failures += _check(
        "V12 denom floor engages",
        frac > 0.0,
        f"{frac*100:.2f}% of (env, step) samples had denom == eps",
    )

    print()
    print("=" * 70)
    print(f"VERIFICATION DONE: {failures} failure(s)")
    print("=" * 70)

    env.close()
    simulation_app.close()
    return failures


if __name__ == "__main__":
    sys.exit(main())
