from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--n_calls", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch

import curriculum_rl  # noqa: F401
from curriculum_rl.envs.liang_composite_reward import composite_liang_energy_reward
from unitree_rl_lab.tasks.locomotion import mdp
from isaaclab.managers import SceneEntityCfg


TASK = "Curriculum-Go2-Velocity-Uniform"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn, n: int) -> float:
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        if isinstance(out, torch.Tensor):
            pass
    sync()
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> int:
    spec = gym.spec(TASK)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].rpartition(":")
    cfg = getattr(__import__(mod, fromlist=[cls]), cls)()
    cfg.scene.num_envs = args.num_envs

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    env.reset(seed=0)
    inner = env.unwrapped
    device = inner.device
    act_dim = env.action_space.shape[-1]
    num_envs = inner.num_envs

    print(f"num_envs={num_envs}  device={device}  act_dim={act_dim}")
    print(f"warmup steps: {args.warmup}")
    for _ in range(args.warmup):
        action = (torch.rand(num_envs, act_dim, device=device) * 2 - 1).clamp(-1, 1)
        env.step(action)

    print()
    print("=" * 60)
    print(f"Timing over {args.n_calls} calls")
    print("=" * 60)

    def step_call():
        action = (torch.rand(num_envs, act_dim, device=device) * 2 - 1).clamp(-1, 1)
        return env.step(action)

    ms_step = time_callable(step_call, args.n_calls)
    ms_composite = time_callable(lambda: composite_liang_energy_reward(inner), args.n_calls)

    print(f"  env.step()                 : {ms_step:7.3f} ms / call")
    print(f"  composite_liang_energy_reward : {ms_composite:7.3f} ms / call")
    print(f"  composite / step ratio        : {ms_composite / ms_step:.3f}x")

    if ms_composite > 5.0:
        print()
        print("=" * 60)
        print("composite > 5ms — profiling internal mdp.* calls")
        print("=" * 60)
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.resolve(inner.scene)
        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
        )
        contact_cfg.resolve(inner.scene)
        feet_asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
        feet_asset_cfg.resolve(inner.scene)
        feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
        feet_sensor_cfg.resolve(inner.scene)

        sub_calls = [
            ("mdp.joint_pos_limits", lambda: mdp.joint_pos_limits(inner, asset_cfg=asset_cfg)),
            ("mdp.action_rate_l2", lambda: mdp.action_rate_l2(inner)),
            ("mdp.undesired_contacts", lambda: mdp.undesired_contacts(inner, threshold=1.0, sensor_cfg=contact_cfg)),
            ("mdp.flat_orientation_l2", lambda: mdp.flat_orientation_l2(inner, asset_cfg=asset_cfg)),
            ("mdp.lin_vel_z_l2", lambda: mdp.lin_vel_z_l2(inner, asset_cfg=asset_cfg)),
            ("mdp.ang_vel_xy_l2", lambda: mdp.ang_vel_xy_l2(inner, asset_cfg=asset_cfg)),
            ("mdp.feet_slide", lambda: mdp.feet_slide(inner, asset_cfg=feet_asset_cfg, sensor_cfg=feet_sensor_cfg)),
        ]
        for name, fn in sub_calls:
            try:
                ms = time_callable(fn, args.n_calls)
                print(f"  {name:30s} : {ms:7.3f} ms / call")
            except Exception as exc:
                print(f"  {name:30s} : ERROR {exc}")

    print()
    print("=" * 60)
    print("Reward Manager term count (additive loop cost driver):")
    print("=" * 60)
    rm = inner.reward_manager
    n_total = len(rm.active_terms)
    n_zero = sum(1 for n in rm.active_terms if rm.get_term_cfg(n).weight == 0.0)
    print(f"  total active reward terms : {n_total}")
    print(f"  zero-weight terms         : {n_zero}  (still computed each step)")

    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
