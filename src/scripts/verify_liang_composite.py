from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import curriculum_rl  # noqa: F401
from curriculum_rl.envs.liang_composite_reward import energy_cot


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
    print("V1-V7: static cfg checks")
    print("=" * 70)

    rew = cfg.rewards
    failures += _check("V1 energy.func is energy_cot", rew.energy.func is energy_cot)
    failures += _check("V2 energy.weight == 1.0", rew.energy.weight == 1.0, f"got {rew.energy.weight}")
    failures += _check(
        "V3 track_lin_vel_xy.weight == 1.5", rew.track_lin_vel_xy.weight == 1.5,
        f"got {rew.track_lin_vel_xy.weight}",
    )
    failures += _check(
        "V4 track_ang_vel_z.weight == 0.75", rew.track_ang_vel_z.weight == 0.75,
        f"got {rew.track_ang_vel_z.weight}",
    )
    failures += _check(
        "V5 feet_air_time.weight == 0.0", rew.feet_air_time.weight == 0.0,
        f"got {rew.feet_air_time.weight}",
    )
    failures += _check(
        "V6 air_time_variance.weight == 0.0", rew.air_time_variance.weight == 0.0,
        f"got {rew.air_time_variance.weight}",
    )
    failures += _check("V7 composite_liang absent", not hasattr(rew, "composite_liang"))

    print()
    print("=" * 70)
    print("V8-V11: runtime energy_cot checks")
    print("=" * 70)

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    env.reset(seed=0)
    inner = env.unwrapped
    num_envs = inner.num_envs
    device = inner.device
    act_dim = env.action_space.shape[-1]

    vals = []
    nan_seen = False
    print(f"Stepping {args.num_steps} steps on {num_envs} envs")
    for _ in range(args.num_steps):
        action = (torch.rand(num_envs, act_dim, device=device) * 2 - 1).clamp(-1, 1)
        env.step(action)
        v = energy_cot(inner)
        if torch.isnan(v).any() or torch.isinf(v).any():
            nan_seen = True
            break
        vals.append(v.detach().clone())

    failures += _check("V8 no NaN/Inf", not nan_seen)

    if vals:
        all_vals = torch.stack(vals)
        in_range = (all_vals > 0.0).all() and (all_vals <= 1.0 + 1e-6).all()
        failures += _check(
            "V9 values in (0, 1]",
            bool(in_range),
            f"min={all_vals.min().item():.4f}  max={all_vals.max().item():.4f}",
        )
        mean_val = all_vals.mean().item()
        failures += _check(
            "V10 shape (num_steps, num_envs)",
            all_vals.shape == (len(vals), num_envs),
            f"got {all_vals.shape}",
        )
        failures += _check(
            "V11 mean in (0.05, 0.95) — not stuck at extremes",
            0.05 < mean_val < 0.95,
            f"mean={mean_val:.4f}",
        )
        print(f"\n  mean={mean_val:.4f}  min={all_vals.min().item():.4f}  max={all_vals.max().item():.4f}")

    print()
    print("=" * 70)
    print(f"VERIFICATION DONE: {failures} failure(s)")
    print("=" * 70)

    env.close()
    simulation_app.close()
    return failures


if __name__ == "__main__":
    sys.exit(main())
