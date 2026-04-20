from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeResult:
    tracking_error: float
    fall_step: int
    episode_length: int


def compute_epte_sp(episodes: list[EpisodeResult]) -> np.ndarray:
    raise NotImplementedError


def mean_epte_sp_per_bin(
    results: dict[int, list[EpisodeResult]],
) -> dict[int, float]:
    raise NotImplementedError
