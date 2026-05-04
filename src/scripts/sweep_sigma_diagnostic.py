from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

print("[diag] script start", flush=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT / "unitree_rl_lab")
print(f"[diag] cwd = {os.getcwd()}", flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--task", type=str, default="Curriculum-Go2-Velocity-Uniform")
AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()
args_cli.headless = True

print("[diag] launching AppLauncher (headless)...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[diag] AppLauncher up", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
import curriculum_rl  # noqa: F401

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

SIGMA_X_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]
SIGMA_Z_RATIO = 0.5
EPS = 0.1


def main() -> int:
    print(f"[diag] parsing env cfg for {args_cli.task} (play entry)...", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    print(f"[diag] env_cfg.scene.num_envs = {env_cfg.scene.num_envs}", flush=True)

    print("[diag] creating env...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    inner = env.unwrapped
    device = inner.device
    print(f"[diag] env created, device={device}", flush=True)

    asset = inner.scene["robot"]
    act_dim = asset.data.joint_pos.shape[-1]
    print(f"[diag] act_dim={act_dim}", flush=True)

    print("[diag] reset()...", flush=True)
    env.reset(seed=0)
    print("[diag] reset done; starting random-action rollout", flush=True)

    r_en_acc: dict[int, list[torch.Tensor]] = {s: [] for s in SIGMA_X_VALUES}
    power_acc: list[float] = []
    vx_acc: list[float] = []
    wz_acc: list[float] = []

    for step in range(args_cli.num_steps):
        action = (torch.rand(args_cli.num_envs, act_dim, device=device) * 2.0 - 1.0).clamp(-1.0, 1.0)
        env.step(action)

        v_x = asset.data.root_lin_vel_b[:, 0]
        w_z = asset.data.root_ang_vel_b[:, 2]
        qvel = asset.data.joint_vel
        qfrc = asset.data.applied_torque
        power = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

        power_acc.append(power.mean().item())
        vx_acc.append(torch.abs(v_x).mean().item())
        wz_acc.append(torch.abs(w_z).mean().item())

        for s in SIGMA_X_VALUES:
            s_z = s * SIGMA_Z_RATIO
            denom = torch.clamp(s * torch.abs(v_x) + s_z * torch.abs(w_z), min=EPS)
            r_en = torch.exp(-power / denom)
            r_en_acc[s].append(r_en.detach().cpu())

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args_cli.num_steps}", flush=True)

    print("[diag] rollout complete; closing env", flush=True)
    env.close()

    print()
    print(f"mean power : {np.mean(power_acc):.3f} W")
    print(f"mean |v_x| : {np.mean(vx_acc):.4f} m/s")
    print(f"mean |w_z| : {np.mean(wz_acc):.4f} rad/s")
    print()
    print(f"{'sigma_en_x':>10}  {'mean(R_en)':>10}  {'std(R_en)':>9}  {'min':>7}  {'max':>7}  gradient regime")
    print("-" * 80)

    for s in SIGMA_X_VALUES:
        data = torch.cat(r_en_acc[s]).numpy()
        m = float(data.mean())
        std = float(data.std())
        mn = float(data.min())
        mx = float(data.max())
        if m < 0.05:
            regime = "too strong (saturated near 0)"
        elif m < 0.30:
            regime = "strong"
        elif m < 0.75:
            regime = "useful gradient"
        elif m < 0.95:
            regime = "weak gradient"
        else:
            regime = "saturated near 1, no gradient"
        print(f"{s:>10}  {m:>10.3f}  {std:>9.3f}  {mn:>7.3f}  {mx:>7.3f}  {regime}")

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except SystemExit:
        raise
    except BaseException:
        print("[diag] FATAL exception in main():", flush=True)
        traceback.print_exc()
        rc = 1
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(rc)
