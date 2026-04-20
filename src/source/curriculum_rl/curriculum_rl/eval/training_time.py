from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingTimeLogger:
    out_dir: Path
    start_time: float = field(default_factory=time.monotonic)
    per_iteration_seconds: list[tuple[int, float]] = field(default_factory=list)
    _last_tick: float = field(default_factory=time.monotonic)

    def tick(self, iteration: int) -> float:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        self.per_iteration_seconds.append((iteration, dt))
        return dt

    def total_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def save(self) -> None:
        raise NotImplementedError
