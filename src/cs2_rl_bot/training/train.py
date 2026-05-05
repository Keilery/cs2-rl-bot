"""Training entry point. Intentionally guarded behind an explicit flag.

Running this against a live CS2 instance on a VAC-protected server WILL get
your account banned. The :func:`train` function refuses to proceed unless the
environment is configured with ``dry_run=False`` and the operator passes
``--i-understand-the-risks``.
"""

from __future__ import annotations

from pathlib import Path

from cs2_rl_bot.agent.ppo_agent import PPOAgent
from cs2_rl_bot.env.cs2_env import CS2Env
from cs2_rl_bot.utils.config import AppConfig
from cs2_rl_bot.utils.logging import configure_logging, logger


class TrainingRefusedError(RuntimeError):
    """Raised when the operator hasn't acknowledged the VAC ban risk."""


def train(
    config: AppConfig,
    *,
    total_timesteps: int = 100_000,
    acknowledged_risks: bool = False,
    checkpoint_every: int = 10_000,
) -> None:
    configure_logging(config.log_level)

    if not acknowledged_risks:
        raise TrainingRefusedError(
            "Refusing to train: pass `acknowledged_risks=True` after reading "
            "docs/legal_and_safety.md. Online matchmaking will VAC-ban you."
        )
    if config.dry_run:
        raise TrainingRefusedError(
            "Refusing to train with dry_run=True. Set CS2BOT_DRY_RUN=false in "
            "config to enable controller output."
        )
    if config.agent.algo != "ppo":
        raise TrainingRefusedError(f"train() only supports algo='ppo', got {config.agent.algo!r}.")

    env = CS2Env(config)
    agent = PPOAgent(env, config.agent)

    checkpoint_dir = Path(config.agent.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training PPO for {} timesteps -> {}", total_timesteps, checkpoint_dir)

    remaining = total_timesteps
    iteration = 0
    try:
        while remaining > 0:
            chunk = min(checkpoint_every, remaining)
            agent.learn(chunk)
            remaining -= chunk
            iteration += 1
            ckpt = checkpoint_dir / f"ppo_iter_{iteration:04d}.zip"
            agent.save(ckpt)
            logger.info("checkpoint saved -> {}", ckpt)
    finally:
        env.close()
