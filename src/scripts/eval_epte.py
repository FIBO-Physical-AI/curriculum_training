from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


CONDITION_TO_TASK = {
    "uniform": "Curriculum-Go2-Velocity-Uniform",
    "task_specific": "Curriculum-Go2-Velocity-TaskSpec",
    "teacher": "Curriculum-Go2-Velocity-Teacher",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=list(CONDITION_TO_TASK.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument("--rollouts-per-bin", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "src/results/epte_sp.csv")
    return parser


def run(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_cfg = argparse.Namespace(
        headless=True,
        enable_cameras=False,
        device="cuda:0",
        num_gpus=1,
        experience="",
        renderer="RaytracedLighting",
        livestream=-1,
        kit_args="",
    )
    app = AppLauncher(app_cfg).app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    import curriculum_rl  # noqa: F401  (registers gym ids)
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    task_id = CONDITION_TO_TASK[args.condition]

    env_cfg = parse_env_cfg(
        task_id,
        device=app_cfg.device,
        num_envs=args.rollouts_per_bin,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(retrieve_file_path(str(args.checkpoint))))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    bin_width = args.v_max / args.num_bins
    reward_cfg = env.unwrapped.reward_manager.get_term_cfg("track_lin_vel_xy")
    reward_weight = float(reward_cfg.weight)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out_csv.exists() or args.out_csv.stat().st_size == 0
    with args.out_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["condition", "seed", "bin_idx", "rollout", "fall_step", "tracking_error", "epte_sp", "v_x_mean", "v_x_signed_mean"])

        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        for b in range(args.num_bins):
            v = (b + 0.5) * bin_width
            fall_step = torch.full((args.rollouts_per_bin,), args.episode_steps, dtype=torch.long, device=env.unwrapped.device)
            err_sum = torch.zeros(args.rollouts_per_bin, device=env.unwrapped.device)
            v_abs_sum = torch.zeros(args.rollouts_per_bin, device=env.unwrapped.device)
            v_signed_sum = torch.zeros(args.rollouts_per_bin, device=env.unwrapped.device)
            alive = torch.ones(args.rollouts_per_bin, dtype=torch.bool, device=env.unwrapped.device)

            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            cmd_term.vel_command_b[:, 0] = v
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0
            cmd_term.env_bin_idx[:] = b

            for step in range(args.episode_steps):
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _r, dones, _info = env.step(actions)
                cmd_term.vel_command_b[:, 0] = v
                cmd_term.vel_command_b[:, 1] = 0.0
                cmd_term.vel_command_b[:, 2] = 0.0

                v_act = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, 0]
                lin_err = torch.abs(v_act - v)
                err_sum[alive] += lin_err[alive]
                v_abs_sum[alive] += torch.abs(v_act[alive])
                v_signed_sum[alive] += v_act[alive]

                fall_mask = dones.bool() & alive
                fall_step[fall_mask] = step
                alive = alive & ~dones.bool()
                if not alive.any():
                    break

            K = args.episode_steps
            denom = fall_step.clamp(min=1).float()
            err_norm = (err_sum / denom / max(v, 1e-6)).clamp(0.0, 1.0)
            v_x_mean = v_abs_sum / denom
            v_x_signed_mean = v_signed_sum / denom
            k_f = fall_step.clone().clamp(0, K)
            epte = (err_norm * k_f.float() + (K - k_f).float()) / K
            for i in range(args.rollouts_per_bin):
                writer.writerow([
                    args.condition, args.seed, b, i,
                    int(k_f[i].item()),
                    float(err_norm[i].item()),
                    float(epte[i].item()),
                    float(v_x_mean[i].item()),
                    float(v_x_signed_mean[i].item()),
                ])

            early_term = int((k_f < K - 1).sum())
            print(
                f"[eval_epte] bin {b}  v_cmd={v:.2f}  v_act={float(v_x_signed_mean.mean()):+.2f}  "
                f"EPTE={float(epte.mean()):.3f}  early={early_term}/{args.rollouts_per_bin}"
            )

    env.close()
    app.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    os.chdir(REPO_ROOT / "unitree_rl_lab")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
