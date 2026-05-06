"""Run a trained (or untrained) agent against a live CS2 client.

This module *does not* perform any optimisation — it just steps the environment
in a loop and prints diagnostics. Useful as a smoke test before unlocking
training.
"""

from __future__ import annotations

from cs2_rl_bot.agent.ppo_agent import build_agent
from cs2_rl_bot.env.cs2_env import CS2Env
from cs2_rl_bot.env.flat_action import FlatActionWrapper
from cs2_rl_bot.utils.config import AppConfig
from cs2_rl_bot.utils.logging import configure_logging, logger


def run_inference(config: AppConfig, *, max_steps: int = 1000) -> None:
    configure_logging(config.log_level)
    base_env = CS2Env(config)
    # SB3 PPO does not accept Dict action spaces — flatten only for that path.
    env = FlatActionWrapper(base_env) if config.agent.algo == "ppo" else base_env
    agent = build_agent(config.agent.algo, env, config.agent)

    logger.info(
        "Starting inference loop (algo={}, dry_run={}, max_steps={})",
        config.agent.algo,
        config.dry_run,
        max_steps,
    )

    obs, _info = env.reset()
    try:
        for step in range(max_steps):
            action = agent.predict(obs)
            obs, reward, terminated, _truncated, info = env.step(action)
            if step % 50 == 0:
                breakdown = info.get("reward_breakdown")
                logger.info(
                    "step={} reward={:+.3f} hp={} round_phase={} kill_switch={}",
                    step,
                    reward,
                    int(info["raw"].player.health),
                    info["raw"].round.phase.value,
                    info.get("kill_switch", False),
                )
                if breakdown is not None:
                    logger.debug("breakdown={}", breakdown)
            if terminated:
                logger.info("Round terminated, resetting.")
                obs, _info = env.reset()
    finally:
        env.close()
