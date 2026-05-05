"""Training entry point with run directories, callbacks, and resume support.

Running this against a live CS2 instance on a VAC-protected server WILL get
your account banned. The :func:`train` function refuses to proceed unless the
environment is configured with ``dry_run=False`` and the operator passes
``acknowledged_risks=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CallbackList

from cs2_rl_bot.agent.ppo_agent import PPOAgent
from cs2_rl_bot.env.cs2_env import CS2Env
from cs2_rl_bot.training.callbacks import (
    MetricsCallback,
    RoundCheckpointCallback,
    TrainingControlCallback,
)
from cs2_rl_bot.training.control import ControlChannel, reset_control
from cs2_rl_bot.training.run_dir import RunDir, RunRegistry
from cs2_rl_bot.utils.config import AppConfig
from cs2_rl_bot.utils.logging import configure_logging, logger


class TrainingRefusedError(RuntimeError):
    """Raised when the operator hasn't acknowledged the VAC ban risk."""


@dataclass(slots=True)
class TrainingHandle:
    """Returned by :func:`train` so callers (e.g. the CLI) can introspect."""

    run: RunDir
    total_timesteps: int
    final_checkpoint: Path | None


def train(
    config: AppConfig,
    *,
    total_timesteps: int = 100_000,
    acknowledged_risks: bool = False,
    save_every_steps: int = 10_000,
    resume_run_id: str | None = None,
    runs_base: Path | None = None,
) -> TrainingHandle:
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

    registry = RunRegistry(base=runs_base or Path("runs"))
    if resume_run_id is not None:
        run = registry.get(resume_run_id)
        if not run.root.exists():
            raise FileNotFoundError(f"run not found: {resume_run_id}")
        logger.info("resuming run {}", run.run_id)
    else:
        run = registry.create(kind="ppo")
        _snapshot_config(run, config)
        logger.info("created run {}", run.run_id)

    reset_control(run)
    run.update_status("running")

    env = CS2Env(config)
    env_config = _patch_agent_paths(config, run)
    agent = PPOAgent(env, env_config.agent)

    latest = run.latest_checkpoint()
    if latest is not None:
        logger.info("loading checkpoint {}", latest)
        agent.load(latest)

    channel = ControlChannel(run)
    channel.status.total_timesteps = total_timesteps
    channel.status.state = "running"
    channel.write_status()

    control_cb = TrainingControlCallback(channel)
    checkpoint_cb = RoundCheckpointCallback(
        run,
        save_every_steps=save_every_steps,
        save_request_provider=control_cb,
    )
    metrics_cb = MetricsCallback(channel)
    callbacks = CallbackList([control_cb, metrics_cb, checkpoint_cb])

    final: Path | None = None
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_run_id is None,
        )
        run.update_status("finished")
        final = run.checkpoint_dir / f"ppo_final_step_{channel.status.current_step:09d}.zip"
        agent.save(final)
        logger.info("training complete; final checkpoint -> {}", final)
    except KeyboardInterrupt:  # pragma: no cover — manual abort
        run.update_status("stopped")
        logger.warning("training interrupted; saving final state")
        final = run.checkpoint_dir / f"ppo_interrupt_step_{channel.status.current_step:09d}.zip"
        agent.save(final)
    except Exception as exc:
        run.update_status("errored")
        channel.status.state = "errored"
        channel.status.error = repr(exc)
        channel.write_status()
        raise
    finally:
        env.close()

    channel.status.last_checkpoint = str(final) if final else channel.status.last_checkpoint
    channel.status.state = run.read_meta().status
    channel.write_status()

    return TrainingHandle(run=run, total_timesteps=total_timesteps, final_checkpoint=final)


def _snapshot_config(run: RunDir, config: AppConfig) -> None:
    data = json.loads(config.model_dump_json())
    run.config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _patch_agent_paths(config: AppConfig, run: RunDir) -> AppConfig:
    """Redirect tensorboard + checkpoint paths into the run directory."""
    patched = config.model_copy(deep=True)
    patched.agent.checkpoint_dir = str(run.checkpoint_dir)
    patched.agent.tensorboard_log = str(run.tensorboard_dir)
    return patched
