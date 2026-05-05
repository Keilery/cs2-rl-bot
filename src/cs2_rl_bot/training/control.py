"""File-based control protocol between the trainer and the dashboard.

The trainer and the dashboard run as **separate processes** (you might attach
the dashboard from a second terminal, an SSH session, or even a different
machine that has the runs directory mounted). Communication happens via two
JSON files inside the run directory:

* ``status.json`` — written by the trainer roughly once per second; never
  modified by the dashboard.
* ``control.json`` — written by the dashboard; the trainer polls it inside
  its training callback and acts on the flags.

Both are written atomically (via ``atomic_write_json``) so a partial read
returns the previous payload rather than parse errors.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from cs2_rl_bot.training.run_dir import RunDir, atomic_write_json


@dataclass(slots=True)
class TrainingStatus:
    """Live trainer state, written to ``status.json`` periodically."""

    run_id: str = ""
    state: str = "pending"  # pending | running | paused | stopped | finished | errored
    started_at: float = 0.0
    updated_at: float = 0.0
    total_timesteps: int = 0
    current_step: int = 0
    rounds_completed: int = 0
    last_round_reward: float = 0.0
    rolling_reward: float = 0.0  # mean over the last N rounds
    last_kills: int = 0
    last_deaths: int = 0
    last_damage: int = 0
    capture_fps: float = 0.0
    gsi_updates: int = 0
    kill_switch: bool = False
    last_checkpoint: str | None = None
    error: str | None = None

    # Rolling buffers — not persisted by reference; we serialize the contents.
    recent_rewards: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recent_rewards"] = list(self.recent_rewards)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingStatus:
        recent = list(data.get("recent_rewards", []))
        d = {k: v for k, v in data.items() if k != "recent_rewards"}
        return cls(recent_rewards=recent, **d)


@dataclass(slots=True)
class ControlSignal:
    """Commands sent from the dashboard to the trainer."""

    paused: bool = False
    stop_requested: bool = False
    save_now: bool = False
    # Monotonic counter — incremented by the dashboard on each new command so
    # the trainer can ignore duplicates.
    revision: int = 0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlSignal:
        return cls(
            paused=bool(data.get("paused", False)),
            stop_requested=bool(data.get("stop_requested", False)),
            save_now=bool(data.get("save_now", False)),
            revision=int(data.get("revision", 0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


class ControlChannel:
    """Trainer-side helper for reading control signals + writing status."""

    def __init__(self, run: RunDir, *, rolling_window: int = 50) -> None:
        self._run = run
        self._status = TrainingStatus(run_id=run.run_id, started_at=time.time())
        self._reward_window: deque[float] = deque(maxlen=rolling_window)
        # 0 is the "no command yet" sentinel — the dashboard's first command
        # bumps it to 1.
        self._last_signal_revision = 0

    @property
    def status(self) -> TrainingStatus:
        return self._status

    def write_status(self) -> None:
        self._status.updated_at = time.time()
        self._status.recent_rewards = list(self._reward_window)
        if self._reward_window:
            self._status.rolling_reward = sum(self._reward_window) / len(self._reward_window)
        atomic_write_json(self._run.status_path, self._status.to_dict())

    def push_round_reward(self, reward: float) -> None:
        self._reward_window.append(float(reward))
        self._status.last_round_reward = float(reward)
        self._status.rounds_completed += 1

    def read_signal(self) -> ControlSignal:
        if not self._run.control_path.exists():
            return ControlSignal()
        try:
            with self._run.control_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return ControlSignal()
        return ControlSignal.from_dict(data)

    def fresh_signal(self) -> ControlSignal | None:
        """Return the signal only if its revision changed since last poll."""
        sig = self.read_signal()
        if sig.revision == self._last_signal_revision:
            return None
        self._last_signal_revision = sig.revision
        return sig


class DashboardClient:
    """Dashboard-side helper for reading status + writing control signals."""

    def __init__(self, run: RunDir) -> None:
        self._run = run
        self._signal = ControlSignal()

    @property
    def signal(self) -> ControlSignal:
        return self._signal

    def read_status(self) -> TrainingStatus | None:
        path = self._run.status_path
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return TrainingStatus.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError):
            return None

    def _bump(self, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(self._signal, k, v)
        self._signal.revision += 1
        self._signal.updated_at = time.time()
        atomic_write_json(self._run.control_path, self._signal.to_dict())

    def pause(self) -> None:
        self._bump(paused=True)

    def resume(self) -> None:
        self._bump(paused=False)

    def request_save(self) -> None:
        self._bump(save_now=True)

    def request_stop(self) -> None:
        self._bump(stop_requested=True)


def load_status(run: RunDir) -> TrainingStatus | None:
    return DashboardClient(run).read_status()


def load_control(run: RunDir) -> ControlSignal:
    return ControlChannel(run).read_signal()


def reset_control(run: RunDir) -> None:
    """Wipe the control file at the start of a new training run."""
    if run.control_path.exists():
        run.control_path.unlink()
