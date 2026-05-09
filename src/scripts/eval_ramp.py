from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


CONDITION_TO_TASK = {
    "uniform": "Curriculum-Go2-Velocity-Uniform",
    "task_specific": "Curriculum-Go2-Velocity-TaskSpec",
    "teacher": "Curriculum-Go2-Velocity-Teacher",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="teacher", choices=list(CONDITION_TO_TASK.keys()))
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v-start", type=float, default=0.0)
    parser.add_argument("--v-end", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=30.0, help="ramp duration in seconds")
    parser.add_argument("--hold-duration", type=float, default=10.0,
                        help="seconds held at v_end after the ramp completes")
    parser.add_argument("--out", type=Path, default=None,
                        help="output npz path; defaults to src/results/ramp_<condition>_seed<N>.npz")
    parser.add_argument("--video", action="store_true",
                        help="record a video of the ramp rollout")
    parser.add_argument("--video-dir", type=Path, default=None,
                        help="directory to write ramp.mp4 into; defaults to <out_parent>/ramp_<cond>_seed<N>_video")
    parser.add_argument("--video-length", type=int, default=0,
                        help="max frames to record (0 = full duration)")
    return parser


def run(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    total_duration = args.duration + max(0.0, args.hold_duration)
    sim_dt_guess = 0.02
    total_steps_guess = int(total_duration / sim_dt_guess)
    video_steps = args.video_length if args.video_length > 0 else total_steps_guess

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_cfg = launcher_parser.parse_args([])
    app_cfg.headless = True
    app_cfg.enable_cameras = bool(args.video)
    app_cfg.device = "cuda:0"

    app = AppLauncher(app_cfg).app

    import torch
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    import curriculum_rl  # noqa: F401
    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    task_id = CONDITION_TO_TASK[args.condition]

    env_cfg = parse_env_cfg(
        task_id,
        device=app_cfg.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.episode_length_s = float(total_duration) + 5.0
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        if args.video_dir is not None:
            video_dir = args.video_dir
        else:
            video_dir = args.out.parent / f"ramp_{args.condition}_seed{args.seed}_video"
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                name_prefix="ramp",
                step_trigger=lambda step: step == 0,
                video_length=video_steps,
                disable_logger=True,
            )
            print(f"[eval_ramp] video -> {video_dir} (length={video_steps} frames)")
        except Exception as e:
            print(f"[eval_ramp] WARN: RecordVideo failed ({e}), continuing without video")

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg: RslRlOnPolicyRunnerCfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(retrieve_file_path(str(args.checkpoint))))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    sim_dt = float(env.unwrapped.step_dt)
    total_steps = int(total_duration / sim_dt)
    ramp_steps = int(args.duration / sim_dt)

    contact_sensor = env.unwrapped.scene["contact_forces"]
    foot_indices, foot_names = contact_sensor.find_bodies(".*_foot")
    foot_indices_t = torch.tensor(foot_indices, device=env.unwrapped.device, dtype=torch.long)
    n_feet = len(foot_indices)

    t_arr = np.zeros(total_steps, dtype=np.float32)
    vcmd_arr = np.zeros(total_steps, dtype=np.float32)
    vx_arr = np.zeros(total_steps, dtype=np.float32)
    vy_arr = np.zeros(total_steps, dtype=np.float32)
    contact_arr = np.zeros((total_steps, n_feet), dtype=np.bool_)
    force_arr = np.zeros((total_steps, n_feet), dtype=np.float32)
    n_recorded = total_steps

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    for step in range(total_steps):
        t = step * sim_dt
        if step < ramp_steps:
            frac = t / max(args.duration, 1e-6)
            v_cmd = args.v_start + frac * (args.v_end - args.v_start)
        else:
            v_cmd = args.v_end

        cmd_term.vel_command_b[:, 0] = v_cmd
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0

        with torch.inference_mode():
            actions = policy(obs)
        obs, _r, dones, _info = env.step(actions)

        cmd_term.vel_command_b[:, 0] = v_cmd
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0

        robot_data = env.unwrapped.scene["robot"].data
        v_b = robot_data.root_lin_vel_b[0]

        forces = contact_sensor.data.net_forces_w[:, foot_indices_t, :].norm(dim=-1)
        in_contact = forces > 1.0

        t_arr[step] = t
        vcmd_arr[step] = v_cmd
        vx_arr[step] = float(v_b[0].item())
        vy_arr[step] = float(v_b[1].item())
        contact_arr[step] = in_contact[0].detach().cpu().numpy()
        force_arr[step] = forces[0].detach().cpu().numpy()

        if bool(dones[0].item()):
            print(f"[eval_ramp] WARN: physical termination at step={step} t={t:.2f}s vcmd={v_cmd:.2f} vx={vx_arr[step]:.2f}")
            n_recorded = step + 1
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.out),
        t=t_arr[:n_recorded],
        vcmd=vcmd_arr[:n_recorded],
        vx=vx_arr[:n_recorded],
        vy=vy_arr[:n_recorded],
        contact=contact_arr[:n_recorded],
        force=force_arr[:n_recorded],
        sim_dt=np.array(sim_dt, dtype=np.float32),
        foot_names=np.array(foot_names),
        condition=np.array(args.condition),
        seed=np.array(args.seed),
    )
    print(f"[eval_ramp] saved {n_recorded} steps -> {args.out}")

    env.close()

    if args.video:
        try:
            target_video_dir = args.video_dir if args.video_dir is not None \
                else args.out.parent / f"ramp_{args.condition}_seed{args.seed}_video"
            mp4s = sorted(target_video_dir.glob("ramp*.mp4"))
            if mp4s:
                src = mp4s[0]
                dst = target_video_dir / "ramp.mp4"
                if src != dst:
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
                for extra in mp4s[1:]:
                    if extra.exists():
                        extra.unlink()
                print(f"[eval_ramp] video -> {dst}")
        except Exception as e:
            print(f"[eval_ramp] WARN: post-rename failed ({e})")

    app.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.out is None:
        args.out = REPO_ROOT / "src" / "results" / f"ramp_{args.condition}_seed{args.seed}.npz"
    args.out = Path(args.out).resolve()
    if args.video_dir is not None:
        args.video_dir = Path(args.video_dir).resolve()
    args.checkpoint = Path(args.checkpoint).resolve()
    os.chdir(REPO_ROOT / "unitree_rl_lab")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
