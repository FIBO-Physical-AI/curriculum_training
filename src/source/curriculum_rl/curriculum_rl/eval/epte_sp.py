from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeResult:
    tracking_error: float
    fall_step: int
    episode_length: int


def epte_sp_one(ep: EpisodeResult) -> float:
    K = int(ep.episode_length)
    if K <= 0:
        raise ValueError("episode_length must be > 0")
    k_f = int(ep.fall_step) if ep.fall_step is not None else K
    k_f = max(0, min(k_f, K))
    eps = float(np.clip(ep.tracking_error, 0.0, 1.0))
    return (eps * k_f + (K - k_f)) / K


def compute_epte_sp(episodes: list[EpisodeResult]) -> np.ndarray:
    return np.array([epte_sp_one(e) for e in episodes], dtype=np.float64)


def mean_epte_sp_per_bin(results: dict[int, list[EpisodeResult]]) -> dict[int, float]:
    return {bin_idx: float(compute_epte_sp(eps).mean()) for bin_idx, eps in results.items()}
