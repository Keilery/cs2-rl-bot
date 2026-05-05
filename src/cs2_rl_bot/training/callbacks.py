"""SB3 callbacks for control, checkpointing, and metric logging.

These callbacks use a :class:`ControlChannel` to surface live state to a
dashboard process and to react to pause / stop / save commands.

Callbacks must be cheap — they run inside the PPO inner loop. The metrics
collector keeps O(1) state and writes ``status.json`` at most once per second.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from stable_baselines3.common.callbacks import BaseCallback

from cs2_rl_bot.observation.state import RoundPhase, Team
from cs2_rl_bot.training.control import ControlChannel
from cs2_rl_bot.training.run_dir import RunDir
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    pass


class TrainingControlCallback(BaseCallback):
    """Polls the control channel; pauses, stops, or saves on demand.

    * ``paused=True`` → busy-waits inside ``_on_step`` until cleared.
    * ``stop_requested=True`` → returns ``False``, ending ``learn()``.
    * ``save_now=True`` → triggers a save via the parent run's checkpoint
      callback (we just set a flag; the checkpoint callback owns disk writes).
    """

    def __init__(
        self,
        channel: ControlChannel,
        *,
        poll_every_steps: int = 16,
        pause_sleep_s: float = 0.1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._channel = channel
        self._poll_every = poll_every_steps
        self._pause_sleep = pause_sleep_s
        self._save_requested = False

    @property
    def save_requested(self) -> bool:
        """One-shot flag set when a save was requested via control.json."""
        flag = self._save_requested
        self._save_requested = False
        return flag

    def _on_step(self) -> bool:
        if self.num_timesteps % self._poll_every != 0:
            return True

        signal = self._channel.fresh_signal()
        if signal is None:
            return True

        if signal.save_now:
            self._save_requested = True
            self._channel.status.state = "running"

        if signal.paused:
            self._channel.status.state = "paused"
            self._channel.write_status()
            logger.info("training paused via dashboard")
            while True:
                time.sleep(self._pause_sleep)
                live = self._channel.read_signal()
                if live.stop_requested:
                    self._channel.status.state = "stopped"
                    self._channel.write_status()
                    return False
                if not live.paused:
                    self._channel.status.state = "running"
                    self._channel.write_status()
                    logger.info("training resumed")
                    break

        if signal.stop_requested:
            self._channel.status.state = "stopped"
            self._channel.write_status()
            logger.info("training stopped via dashboard")
            return False

        return True


class RoundCheckpointCallback(BaseCallback):
    """Save model on round termination + every ``save_every_steps`` steps."""

    def __init__(
        self,
        run: RunDir,
        *,
        save_every_steps: int = 10_000,
        keep_last: int = 10,
        save_on_round_end: bool = True,
        save_request_provider: TrainingControlCallback | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._run = run
        self._save_every = save_every_steps
        self._keep_last = keep_last
        self._save_on_round_end = save_on_round_end
        self._save_request_provider = save_request_provider
        self._next_step_target = save_every_steps

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [])
        if not isinstance(dones, list | tuple):
            dones = list(dones)

        # Manual save request from the dashboard.
        if self._save_request_provider is not None and self._save_request_provider.save_requested:
            self._save_checkpoint(reason="manual")

        if self._save_on_round_end:
            for info, done in zip(infos, dones, strict=False):
                raw = info.get("raw") if isinstance(info, dict) else None
                if (
                    done
                    and raw is not None
                    and getattr(getattr(raw, "round", None), "phase", None) is RoundPhase.OVER
                ):
                    self._save_checkpoint(reason="round_end")
                    break

        if self.num_timesteps >= self._next_step_target:
            self._save_checkpoint(reason="step")
            self._next_step_target += self._save_every

        return True

    def _save_checkpoint(self, *, reason: str) -> None:
        path = self._run.checkpoint_dir / f"ppo_step_{self.num_timesteps:09d}.zip"
        self.model.save(str(path))
        self._prune_old()
        logger.info("saved checkpoint ({}) -> {}", reason, path.name)

    def _prune_old(self) -> None:
        ckpts = self._run.list_checkpoints()
        if len(ckpts) <= self._keep_last:
            return
        for old in ckpts[: -self._keep_last]:
            try:
                old.unlink()
            except OSError as exc:  # pragma: no cover — best effort
                logger.warning("failed to prune {}: {}", old, exc)


class MetricsCallback(BaseCallback):
    """Aggregate per-round metrics into the run's status.json."""

    def __init__(
        self,
        channel: ControlChannel,
        *,
        write_every_steps: int = 32,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._channel = channel
        self._write_every = write_every_steps
        self._round_kills = 0
        self._round_deaths = 0
        self._round_damage = 0
        self._round_reward = 0.0
        self._capture_window: list[float] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", []) or []
        dones = list(self.locals.get("dones", []) or [])

        for info, reward, done in zip(infos, rewards, dones, strict=False):
            self._round_reward += float(reward)
            raw = info.get("raw") if isinstance(info, dict) else None
            if raw is not None:
                player = getattr(raw, "player", None)
                if player is not None and getattr(player, "team", Team.UNKNOWN) is not Team.UNKNOWN:
                    self._round_kills = int(getattr(player, "round_kills", 0))
                    self._round_deaths = int(getattr(player, "round_deaths", 0))
                    self._round_damage = int(getattr(player, "round_damage", 0))

                frame = getattr(raw, "frame", None)
                if frame is not None and self._capture_window and frame.timestamp > 0:
                    last = self._capture_window[-1]
                    if frame.timestamp > last:
                        self._capture_window.append(frame.timestamp)
                elif frame is not None and frame.timestamp > 0:
                    self._capture_window.append(frame.timestamp)
                if len(self._capture_window) > 60:
                    self._capture_window = self._capture_window[-60:]

            if done:
                self._channel.push_round_reward(self._round_reward)
                self._channel.status.last_kills = self._round_kills
                self._channel.status.last_deaths = self._round_deaths
                self._channel.status.last_damage = self._round_damage
                if self.logger is not None:
                    self.logger.record("round/reward", self._round_reward)
                    self.logger.record("round/kills", self._round_kills)
                    self.logger.record("round/damage", self._round_damage)
                self._round_kills = 0
                self._round_deaths = 0
                self._round_damage = 0
                self._round_reward = 0.0

        self._channel.status.current_step = int(self.num_timesteps)
        if len(self._capture_window) >= 2:
            span = self._capture_window[-1] - self._capture_window[0]
            self._channel.status.capture_fps = (
                (len(self._capture_window) - 1) / span if span > 0 else 0.0
            )

        if self.num_timesteps % self._write_every == 0:
            self._channel.write_status()
        return True

    def _on_training_end(self) -> None:
        self._channel.status.state = self._channel.status.state or "finished"
        self._channel.write_status()
