from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--condition", required=True, choices=list(CONDITION_TO_TASK.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument("--rollouts-per-bin", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--joint-trace-rollouts", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "src/results/epte_sp.csv")
    parser.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "src/results/eval_traces")
    return parser


def _term_done_dict(tm) -> dict[str, "object"]:
    out = {}
    raw = getattr(tm, "_term_dones", None)
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[k] = v
    return out


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

    import curriculum_rl  # noqa: F401
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

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.traces_dir.mkdir(parents=True, exist_ok=True)
    new_file = not args.out_csv.exists() or args.out_csv.stat().st_size == 0
    fh = args.out_csv.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow([
            "condition", "seed", "bin_idx", "rollout",
            "fall_step", "tracking_error", "epte_sp",
            "v_x_mean", "v_x_signed_mean",
            "vfc", "action_rate", "term_reason",
            "duty_factor", "stride_freq_hz", "stride_length_m",
        ])

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    term_mgr = env.unwrapped.termination_manager

    contact_sensor = env.unwrapped.scene["contact_forces"]
    foot_indices, foot_names = contact_sensor.find_bodies(".*_foot")
    foot_indices_t = torch.tensor(foot_indices, device=env.unwrapped.device, dtype=torch.long)
    n_feet = len(foot_indices)
    sim_dt = float(env.unwrapped.step_dt)

    traces_per_bin: dict[int, np.ndarray] = {}
    vcmd_per_bin: dict[int, float] = {}
    contact_traces_per_bin: dict[int, np.ndarray] = {}
    joint_vel_traces_per_bin: dict[int, np.ndarray] = {}

    robot_data = env.unwrapped.scene["robot"].data
    joint_names = list(robot_data.joint_names)
    n_joints = len(joint_names)
    n_jt = max(1, min(args.joint_trace_rollouts, args.rollouts_per_bin))

    for b in range(args.num_bins):
        v = (b + 0.5) * bin_width
        N = args.rollouts_per_bin
        K = args.episode_steps
        device = env.unwrapped.device

        fall_step = torch.full((N,), K, dtype=torch.long, device=device)
        err_sum = torch.zeros(N, device=device)
        v_abs_sum = torch.zeros(N, device=device)
        v_signed_sum = torch.zeros(N, device=device)
        action_rate_sum = torch.zeros(N, device=device)
        action_rate_count = torch.zeros(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        term_code = torch.zeros(N, dtype=torch.long, device=device)
        prev_actions: torch.Tensor | None = None
        vx_trace = np.zeros((N, K), dtype=np.float32)
        contact_trace = np.zeros((N, K, n_feet), dtype=np.bool_)
        joint_vel_trace = np.zeros((n_jt, K, n_joints), dtype=np.float32)
        contact_steps = torch.zeros(N, n_feet, device=device)
        touchdown_count = torch.zeros(N, n_feet, device=device)
        prev_in_contact: torch.Tensor | None = None
        steps_alive = torch.zeros(N, device=device)

        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        cmd_term.vel_command_b[:, 0] = v
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0
        cmd_term.env_bin_idx[:] = b

        REASON_NAMES = ["alive", "time_out", "bad_orientation", "base_contact", "other"]

        for step in range(K):
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
            vx_trace[:, step] = v_act.detach().cpu().numpy()

            forces = contact_sensor.data.net_forces_w[:, foot_indices_t, :].norm(dim=-1)
            in_contact = forces > 1.0
            contact_trace[:, step, :] = in_contact.detach().cpu().numpy()
            jv = robot_data.joint_vel[:n_jt].detach().cpu().numpy()
            joint_vel_trace[:, step, :] = jv
            contact_steps[alive] += in_contact[alive].float()
            if prev_in_contact is not None:
                touchdown = in_contact & ~prev_in_contact
                touchdown_count[alive] += touchdown[alive].float()
            prev_in_contact = in_contact.clone()
            steps_alive[alive] += 1.0

            if prev_actions is not None:
                a_diff = (actions - prev_actions).abs().mean(dim=-1)
                action_rate_sum[alive] += a_diff[alive]
                action_rate_count[alive] += 1
            prev_actions = actions.detach().clone()

            done_bool = dones.bool()
            fall_mask = done_bool & alive
            if fall_mask.any():
                fall_step[fall_mask] = step
                td = _term_done_dict(term_mgr)
                if td:
                    time_out_t = td.get("time_out")
                    bad_ori_t = td.get("bad_orientation")
                    base_con_t = td.get("base_contact")
                    if time_out_t is not None:
                        m = fall_mask & time_out_t.bool()
                        term_code[m] = 1
                    if bad_ori_t is not None:
                        m = fall_mask & bad_ori_t.bool()
                        term_code[m] = 2
                    if base_con_t is not None:
                        m = fall_mask & base_con_t.bool()
                        term_code[m] = 3
                    rest = fall_mask & (term_code == 0)
                    if rest.any():
                        for k_name, t in td.items():
                            if k_name in ("time_out", "bad_orientation", "base_contact"):
                                continue
                            m = rest & t.bool()
                            term_code[m] = 4
                            rest = rest & ~m
                else:
                    term_code[fall_mask] = 4
            alive = alive & ~done_bool
            if not alive.any():
                vx_trace[:, step + 1:] = 0.0
                break

        denom = fall_step.clamp(min=1).float()
        err_norm = (err_sum / denom / max(v, 1e-6)).clamp(0.0, 1.0)
        v_x_mean = v_abs_sum / denom
        v_x_signed_mean = v_signed_sum / denom
        action_rate_denom = action_rate_count.clamp(min=1).float()
        action_rate_mean = action_rate_sum / action_rate_denom
        k_f = fall_step.clone().clamp(0, K)
        epte = (err_norm * k_f.float() + (K - k_f).float()) / K
        vfc = v_x_signed_mean / max(v, 1e-6)

        steps_alive_denom = steps_alive.clamp(min=1).unsqueeze(-1)
        per_foot_duty = contact_steps / steps_alive_denom
        duty_factor = per_foot_duty.mean(dim=-1)
        seconds_alive = (steps_alive.clamp(min=1).float()) * sim_dt
        per_foot_freq = touchdown_count / seconds_alive.unsqueeze(-1)
        stride_freq = per_foot_freq.mean(dim=-1)
        stride_length = torch.where(
            stride_freq > 1e-3,
            v_x_signed_mean / stride_freq.clamp(min=1e-3),
            torch.zeros_like(v_x_signed_mean),
        )

        for i in range(N):
            writer.writerow([
                args.condition, args.seed, b, i,
                int(k_f[i].item()),
                float(err_norm[i].item()),
                float(epte[i].item()),
                float(v_x_mean[i].item()),
                float(v_x_signed_mean[i].item()),
                float(vfc[i].item()),
                float(action_rate_mean[i].item()),
                REASON_NAMES[int(term_code[i].item())],
                float(duty_factor[i].item()),
                float(stride_freq[i].item()),
                float(stride_length[i].item()),
            ])

        traces_per_bin[b] = vx_trace
        vcmd_per_bin[b] = float(v)
        contact_traces_per_bin[b] = contact_trace
        joint_vel_traces_per_bin[b] = joint_vel_trace

        early_term = int((k_f < K - 1).sum())
        print(
            f"[eval_epte] bin {b}  v_cmd={v:.2f}  v_act={float(v_x_signed_mean.mean()):+.2f}  "
            f"VFC={float(vfc.mean()):+.2f}  EPTE={float(epte.mean()):.3f}  "
            f"a_rate={float(action_rate_mean.mean()):.3f}  duty={float(duty_factor.mean()):.2f}  "
            f"freq={float(stride_freq.mean()):.2f}Hz  stride={float(stride_length.mean()):.2f}m  "
            f"early={early_term}/{N}"
        )

    fh.close()

    trace_path = args.traces_dir / f"{args.condition}_seed{args.seed}.npz"
    save_dict = {}
    for b in range(args.num_bins):
        save_dict[f"vx_b{b}"] = traces_per_bin[b]
        save_dict[f"vcmd_b{b}"] = np.array(vcmd_per_bin[b], dtype=np.float32)
        save_dict[f"contact_b{b}"] = contact_traces_per_bin[b]
        save_dict[f"joint_vel_b{b}"] = joint_vel_traces_per_bin[b]
    save_dict["sim_dt"] = np.array(sim_dt, dtype=np.float32)
    save_dict["foot_names"] = np.array(foot_names)
    save_dict["joint_names"] = np.array(joint_names)
    np.savez_compressed(trace_path, **save_dict)
    print(f"[eval_epte] saved traces -> {trace_path}")

    env.close()
    app.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    os.chdir(REPO_ROOT / "unitree_rl_lab")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
