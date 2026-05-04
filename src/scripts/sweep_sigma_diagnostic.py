from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import argparse

SIGMA_X_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]
SIGMA_Z_RATIO = 0.5
EPS = 0.1
DEFAULT_NUM_ENVS = 512
DEFAULT_NUM_STEPS = 1000

REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import curriculum_rl  # noqa: F401

TASK = "Curriculum-Go2-Velocity-Uniform"


def main() -> int:
    env_cfg_entry = gym.spec(TASK).kwargs["env_cfg_entry_point"]
    module_path, _, cls_name = env_cfg_entry.rpartition(":")
    cfg_module = __import__(module_path, fromlist=[cls_name])
    cfg = getattr(cfg_module, cls_name)()
    cfg.scene.num_envs = args.num_envs

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    env.reset(seed=0)
    inner = env.unwrapped
    device = inner.device
    act_dim = env.action_space.shape[-1]

    r_en_acc: dict[int, list[torch.Tensor]] = {s: [] for s in SIGMA_X_VALUES}
    power_acc: list[float] = []
    vx_acc: list[float] = []
    wz_acc: list[float] = []

    for step in range(args.num_steps):
        action = (torch.rand(args.num_envs, act_dim, device=device) * 2.0 - 1.0).clamp(-1.0, 1.0)
        env.step(action)

        asset = inner.scene["robot"]
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

        if (step + 1) % 200 == 0:
            print(f"  step {step + 1}/{args.num_steps}")

    env.close()
    simulation_app.close()

    print()
    print(f"mean power : {np.mean(power_acc):.3f} W")
    print(f"mean |v_x| : {np.mean(vx_acc):.4f} m/s")
    print(f"mean |w_z| : {np.mean(wz_acc):.4f} rad/s")
    print()
    print(f"{'σ_en_x':>8}  {'mean(R_en)':>10}  {'std(R_en)':>9}  {'min':>7}  {'max':>7}  gradient regime")
    print("─" * 80)

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
            regime = "useful gradient ✓"
        elif m < 0.95:
            regime = "weak gradient"
        else:
            regime = "saturated near 1, no gradient"
        print(f"{s:>8}  {m:>10.3f}  {std:>9.3f}  {mn:>7.3f}  {mx:>7.3f}  {regime}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
