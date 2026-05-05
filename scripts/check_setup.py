"""Smoke test that exercises imports + a stub env step without a running game.

Run from the repo root after `pip install -e '.[dev]'`:

    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

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
from cs2_rl_bot.utils.logging import configure_logging, logger


class _FakeProvider:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def latest(self) -> Observation:
        return Observation(
            player=PlayerState(health=100, team=Team.T),
            round=RoundState(phase=RoundPhase.LIVE, round_number=1),
            frame=Frame(image=np.zeros((84, 84, 3), dtype=np.uint8), timestamp=time.time()),
        )


def main() -> int:
    cfg = AppConfig()
    cfg.capture.target_fps = 240  # don't actually wait
    cfg.dry_run = True
    configure_logging(cfg.log_level)

    env = CS2Env(
        cfg,
        observation_provider=_FakeProvider(),
        controller=Controller(cfg.controller, dry_run=True),
    )

    obs, _ = env.reset()
    logger.info(
        "obs.frame.shape={} obs.scalars.shape={}",
        obs["frame"].shape,
        obs["scalars"].shape,
    )

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    logger.info(
        "step ok | reward={:+.4f} terminated={} truncated={} keys={}",
        reward,
        terminated,
        truncated,
        sorted(info.keys()),
    )
    env.close()
    logger.info("smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
