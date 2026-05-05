"""End-to-end env smoke test using a stub observation provider."""

from __future__ import annotations

import time

import numpy as np

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.action.controller import Controller
from cs2_rl_bot.env.cs2_env import CS2Env
from cs2_rl_bot.observation.state import (
    Frame,
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.utils.config import AppConfig


class _StubProvider:
    def __init__(self) -> None:
        self.calls = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest(self) -> Observation:
        self.calls += 1
        # Toggle round phase between LIVE and OVER so episodes terminate.
        phase = RoundPhase.LIVE if self.calls < 5 else RoundPhase.OVER
        win_team = Team.T if phase is RoundPhase.OVER else Team.UNKNOWN
        return Observation(
            player=PlayerState(health=100, team=Team.T, round_kills=self.calls),
            round=RoundState(phase=phase, win_team=win_team, round_number=1),
            frame=Frame(image=np.zeros((84, 84, 3), dtype=np.uint8), timestamp=time.time()),
        )


def test_env_step_loop_runs_to_termination() -> None:
    config = AppConfig()
    config.capture.target_fps = 240  # speed up sleeps
    config.dry_run = True

    provider = _StubProvider()
    controller = Controller(config.controller, dry_run=True)
    env = CS2Env(config, observation_provider=provider, controller=controller)

    obs, info = env.reset()
    assert "frame" in obs
    assert "scalars" in obs
    assert obs["frame"].shape == (84, 84, 3)
    assert obs["scalars"].shape == (14,)
    assert "raw" in info

    terminated = False
    steps = 0
    action = {
        "movement": int(Movement.STAND),
        "mouse": np.zeros(2, dtype=np.float32),
        "attack": 0,
        "jump": 0,
        "crouch": 0,
        "reload": 0,
        "weapon": int(WeaponSlot.NONE),
    }
    while not terminated and steps < 20:
        obs, reward, terminated, truncated, info = env.step(action)
        assert truncated is False
        assert isinstance(reward, float)
        steps += 1

    assert terminated is True
    env.close()
