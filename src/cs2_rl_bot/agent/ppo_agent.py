"""PPO agent skeleton built on top of stable-baselines3.

This module is a thin wrapper that:

1. Builds a stable-baselines3 PPO model bound to a :class:`CS2Env`.
2. Exposes ``predict`` for inference and ``learn`` for training.
3. Saves/loads checkpoints to a configurable directory.

We deliberately do **not** trigger any real training in this scaffold — that is
the user's responsibility. The ``learn`` method is provided for completeness;
calling it without a running CS2 instance will block on environment steps.

The "scripted" and "random" agents are convenience baselines so that the
end-to-end pipeline can be smoke-tested without installing torch or having a
trained policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.utils.config import AgentConfig
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from cs2_rl_bot.env.cs2_env import CS2Env


class Agent(Protocol):
    def predict(self, obs: dict[str, np.ndarray]) -> dict[str, Any]: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...


class RandomAgent:
    """Uniform-random baseline. Useful for smoke tests."""

    def __init__(self, env: CS2Env) -> None:
        self._env = env

    def predict(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        return self._env.action_space.sample()

    def save(self, path: str | Path) -> None:
        return None

    def load(self, path: str | Path) -> None:
        return None


class ScriptedAgent:
    """Hand-coded baseline: walks forward and fires when health drops.

    Demonstrates how to plug in classical bot logic alongside the RL stack.
    """

    def __init__(self, env: CS2Env) -> None:
        self._env = env
        self._last_health = 100

    def predict(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        scalars = obs.get("scalars")
        health = float(scalars[0]) * 100.0 if scalars is not None else 100.0
        attack = 1 if health < self._last_health else 0
        self._last_health = int(health)
        return {
            "movement": int(Movement.FORWARD),
            "mouse": np.zeros(2, dtype=np.float32),
            "attack": attack,
            "jump": 0,
            "crouch": 0,
            "reload": 0,
            "weapon": int(WeaponSlot.PRIMARY),
        }

    def save(self, path: str | Path) -> None:
        return None

    def load(self, path: str | Path) -> None:
        return None


class PPOAgent:
    """stable-baselines3 PPO agent with a Dict observation/action space."""

    def __init__(self, env: CS2Env, config: AgentConfig) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "stable-baselines3 is required for PPOAgent — install it via `pip install -e .`"
            ) from exc

        self._env = env
        self._config = config
        self._PPO = PPO
        self._model = PPO(
            policy=config.policy,
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=config.tensorboard_log,
            verbose=1,
        )

    def predict(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        action, _state = self._model.predict(obs, deterministic=False)
        return action  # type: ignore[no-any-return]

    @property
    def model(self) -> Any:
        """SB3 model handle, for advanced consumers (callbacks, learn args)."""
        return self._model

    def learn(
        self,
        total_timesteps: int,
        *,
        callback: Any | None = None,
        reset_num_timesteps: bool = True,
    ) -> None:
        logger.warning(
            "Starting PPO training for {} timesteps -- this is a real training run.",
            total_timesteps,
        )
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    def load(self, path: str | Path) -> None:
        self._model = self._PPO.load(str(path), env=self._env)


def build_agent(name: str, env: CS2Env, config: AgentConfig) -> Agent:
    if name == "random":
        return RandomAgent(env)
    if name == "scripted":
        return ScriptedAgent(env)
    if name == "ppo":
        return PPOAgent(env, config)
    raise ValueError(f"unknown agent: {name!r}")
