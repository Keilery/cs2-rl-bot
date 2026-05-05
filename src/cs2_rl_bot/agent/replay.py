"""Round-based experience buffer.

CS2 episodes correspond to rounds, which are typically 30-115 seconds long. We
store transitions in a ring buffer and expose them as numpy arrays on demand.
For PPO we don't actually use this -- SB3 manages its own rollout buffer -- but
it's useful for off-policy algorithms (DQN, SAC) and for offline analysis of a
match.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Transition:
    obs: dict[str, np.ndarray]
    action: dict[str, np.ndarray | int | float]
    reward: float
    next_obs: dict[str, np.ndarray]
    terminated: bool
    info: dict[str, Any]


class RoundBuffer:
    """Bounded buffer that flushes when a round ends."""

    def __init__(self, max_size: int = 100_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=max_size)
        self._round_starts: list[int] = []

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)
        if transition.terminated:
            self._round_starts.append(len(self._buffer))

    def latest_round(self) -> list[Transition]:
        if not self._round_starts:
            return list(self._buffer)
        if len(self._round_starts) == 1:
            return list(self._buffer)[: self._round_starts[-1]]
        start = self._round_starts[-2]
        end = self._round_starts[-1]
        return list(self._buffer)[start:end]

    def clear(self) -> None:
        self._buffer.clear()
        self._round_starts.clear()
