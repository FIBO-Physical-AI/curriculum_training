from __future__ import annotations
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import curriculum_rl  # noqa: F401

TASK = "Curriculum-Go2-Velocity-Uniform"
spec = gym.spec(TASK)
mod, _, cls = spec.kwargs["env_cfg_entry_point"].rpartition(":")
cfg = getattr(__import__(mod, fromlist=[cls]), cls)()

print("=" * 60)
print("Reward term weights at training time:")
print("=" * 60)
for name in [
    "track_lin_vel_xy", "track_ang_vel_z",
    "base_linear_velocity", "base_angular_velocity",
    "joint_vel", "joint_acc", "joint_torques", "action_rate",
    "dof_pos_limits", "energy",
    "flat_orientation_l2", "joint_pos",
    "feet_air_time", "air_time_variance", "feet_slide",
    "undesired_contacts", "composite_liang",
]:
    term = getattr(cfg.rewards, name, None)
    if term is None:
        print(f"  {name:25s} = MISSING")
    else:
        print(f"  {name:25s} = {term.weight:>12g}")
print("=" * 60)
print("All RewardsCfg fields (in case anything new appeared):")
for k, v in vars(cfg.rewards).items():
    w = getattr(v, "weight", None)
    print(f"  {k:30s} weight={w}")
print("=" * 60)
app.close()
