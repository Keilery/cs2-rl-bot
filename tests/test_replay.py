"""Tests for the round-based replay buffer."""

from __future__ import annotations

import numpy as np

from cs2_rl_bot.agent.replay import RoundBuffer, Transition


def _t(terminated: bool, idx: int = 0) -> Transition:
    obs = {"frame": np.zeros((1, 1, 1), dtype=np.uint8), "scalars": np.zeros(2, dtype=np.float32)}
    return Transition(
        obs=obs,
        action={"movement": idx},
        reward=float(idx),
        next_obs=obs,
        terminated=terminated,
        info={},
    )


def test_buffer_appends() -> None:
    buf = RoundBuffer()
    buf.add(_t(False))
    buf.add(_t(True))
    assert len(buf) == 2


def test_latest_round_returns_last_round() -> None:
    buf = RoundBuffer()
    buf.add(_t(False, 0))
    buf.add(_t(True, 1))
    buf.add(_t(False, 2))
    buf.add(_t(False, 3))
    buf.add(_t(True, 4))
    last = buf.latest_round()
    assert [t.action["movement"] for t in last] == [2, 3, 4]


def test_clear_resets() -> None:
    buf = RoundBuffer()
    buf.add(_t(True))
    buf.clear()
    assert len(buf) == 0
    assert buf.latest_round() == []
