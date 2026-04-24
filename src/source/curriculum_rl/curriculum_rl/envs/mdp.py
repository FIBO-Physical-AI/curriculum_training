from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_csv_path: Path | None = None
_csv_handle = None

_per_bin_reward_sum: torch.Tensor | None = None
_per_bin_reward_count: torch.Tensor | None = None


def ang_vel_z_l2(env: "ManagerBasedRLEnv", asset_cfg=None) -> torch.Tensor:
    from isaaclab.managers import SceneEntityCfg
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def track_lin_vel_x_linear(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    v_max: float = 4.0,
    asset_cfg=None,
) -> torch.Tensor:
    from isaaclab.managers import SceneEntityCfg
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    v_cmd = cmd[:, 0]
    v_act = asset.data.root_lin_vel_b[:, 0]
    err = torch.abs(v_act - v_cmd)
    return torch.clamp(1.0 - err / v_max, min=0.0)


def forward_velocity_command(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 0:1]


def _resolve_csv_path(env: "ManagerBasedRLEnv") -> Path:
    global _csv_path
    if _csv_path is not None:
        return _csv_path
    override = os.environ.get("CURRICULUM_LOG_PATH")
    if override:
        _csv_path = Path(override)
    else:
        log_dir = getattr(env, "_log_dir", None) or getattr(env.cfg, "log_dir", None)
        if log_dir:
            _csv_path = Path(log_dir) / "curriculum.csv"
        else:
            candidates = [p for p in Path("logs/rsl_rl").glob("*/*") if p.is_dir()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                _csv_path = latest / "curriculum.csv"
            else:
                _csv_path = Path.cwd() / f"curriculum_fallback_{os.getpid()}.csv"
    _csv_path.parent.mkdir(parents=True, exist_ok=True)
    return _csv_path


def _append_rows(env: "ManagerBasedRLEnv", step: int, weights, per_bin_reward, counts) -> None:
    global _csv_handle
    path = _resolve_csv_path(env)
    if _csv_handle is None:
        new_file = not path.exists() or path.stat().st_size == 0
        _csv_handle = open(path, "a", buffering=1)
        if new_file:
            _csv_handle.write("step,bin_idx,weight,mean_reward,n_samples\n")
    for b in range(len(weights)):
        _csv_handle.write(
            f"{step},{b},{float(weights[b]):.6f},{float(per_bin_reward[b]):.6f},{int(counts[b])}\n"
        )


def _get_steps_per_ppo_iteration(env: "ManagerBasedRLEnv") -> int:
    val = os.environ.get("CURRICULUM_STEPS_PER_ITER")
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            pass
    return 48


def velocity_curriculum_step(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    global _per_bin_reward_sum, _per_bin_reward_count
    cmd = env.command_manager.get_term(command_name)

    if _per_bin_reward_sum is None or _per_bin_reward_sum.shape[0] != cmd.num_bins:
        _per_bin_reward_sum = torch.zeros(cmd.num_bins, device=env.device)
        _per_bin_reward_count = torch.zeros(cmd.num_bins, dtype=torch.long, device=env.device)

    if not isinstance(env_ids, torch.Tensor):
        env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
    else:
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)

    if env_ids_t.numel() > 0:
        reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids_t]
        ep_len = env.episode_length_buf[env_ids_t].clamp(min=1).float()
        per_env = (
            episode_sums
            / ep_len
            / max(float(reward_cfg.weight), 1e-6)
        )
        bins = cmd.env_bin_idx[env_ids_t]
        standing = getattr(cmd, "is_standing_env", None)
        if standing is not None:
            keep = ~standing[env_ids_t]
            per_env = per_env[keep]
            bins = bins[keep]
        if per_env.numel() > 0:
            per_env = per_env.clamp(0.0, 1.0)
            _per_bin_reward_sum.scatter_add_(0, bins, per_env)
            _per_bin_reward_count.scatter_add_(0, bins, torch.ones_like(bins))

    steps_per_iter = _get_steps_per_ppo_iteration(env)
    if env.common_step_counter == 0 or env.common_step_counter % steps_per_iter != 0:
        return cmd.weights.max()

    counts = _per_bin_reward_count
    sums = _per_bin_reward_sum
    per_bin = torch.where(counts > 0, sums / counts.clamp(min=1).float(), torch.zeros_like(sums))

    per_bin_np = per_bin.detach().cpu().numpy()
    counts_np = counts.detach().cpu().numpy()

    try:
        cmd.curriculum.update(per_bin_np, int(env.common_step_counter), bin_counts=counts_np)
    except TypeError:
        cmd.curriculum.update(per_bin_np, int(env.common_step_counter))
    cmd.sync_weights_from_curriculum()

    weights_np = cmd.weights.detach().cpu().numpy()
    _append_rows(env, int(env.common_step_counter), weights_np, per_bin_np, counts_np)

    _per_bin_reward_sum.zero_()
    _per_bin_reward_count.zero_()

    return cmd.weights.max()
