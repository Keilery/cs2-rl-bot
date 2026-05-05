"""Tests for cs2_rl_bot.training.callbacks.

These tests use lightweight stubs in place of a real SB3 model so they run
fast and don't require torch or a Gym environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cs2_rl_bot.observation.state import (
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.training.callbacks import (
    MetricsCallback,
    RoundCheckpointCallback,
    TrainingControlCallback,
)
from cs2_rl_bot.training.control import ControlChannel, DashboardClient
from cs2_rl_bot.training.run_dir import RunRegistry


@dataclass(slots=True)
class _StubLogger:
    records: list[tuple[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []

    def record(self, key: str, value: float) -> None:
        self.records.append((key, float(value)))


@dataclass(slots=True)
class _StubModel:
    """The bare minimum a SB3 callback needs to function."""

    save_count: int = 0
    saved_paths: list[Path] = None  # type: ignore[assignment]
    logger: _StubLogger = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.saved_paths is None:
            self.saved_paths = []
        if self.logger is None:
            self.logger = _StubLogger()

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake")
        self.saved_paths.append(Path(path))
        self.save_count += 1


def _make_obs(phase: RoundPhase, kills: int = 0, damage: int = 0) -> Observation:
    return Observation(
        player=PlayerState(
            health=100,
            team=Team.T,
            round_kills=kills,
            round_deaths=0,
            round_damage=damage,
        ),
        round=RoundState(phase=phase),
        frame=None,
        raw_gsi={},
    )


def _bind_callback(cb: object, model: _StubModel) -> None:
    """Inject the minimum globals SB3 callbacks require for ``_on_step``."""
    cb.model = model  # type: ignore[attr-defined]
    cb.locals = {}  # type: ignore[attr-defined]
    cb.globals = {}  # type: ignore[attr-defined]
    cb.num_timesteps = 0  # type: ignore[attr-defined]
    cb.parent = None  # type: ignore[attr-defined]


def test_control_callback_sees_pause_then_stop(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    channel = ControlChannel(run)
    dash = DashboardClient(run)

    cb = TrainingControlCallback(channel, poll_every_steps=1, pause_sleep_s=0.0)
    _bind_callback(cb, _StubModel())

    # No signal: callback returns True.
    assert cb._on_step() is True

    # Stop signal: callback returns False.
    dash.request_stop()
    assert cb._on_step() is False
    assert channel.status.state == "stopped"


def test_control_callback_save_request(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    channel = ControlChannel(run)
    DashboardClient(run).request_save()

    cb = TrainingControlCallback(channel, poll_every_steps=1)
    _bind_callback(cb, _StubModel())
    assert cb._on_step() is True
    assert cb.save_requested is True
    # The flag is one-shot.
    assert cb.save_requested is False


def test_round_checkpoint_callback_step_based(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    model = _StubModel()
    cb = RoundCheckpointCallback(run, save_every_steps=10, keep_last=3)
    _bind_callback(cb, model)
    cb.locals = {"infos": [{}], "dones": [False]}

    cb.num_timesteps = 5
    cb._on_step()
    assert model.save_count == 0  # not yet
    cb.num_timesteps = 10
    cb._on_step()
    assert model.save_count == 1
    cb.num_timesteps = 25
    cb._on_step()
    assert model.save_count == 2

    # Pruning kicks in once we exceed keep_last=3.
    for step in (30, 40, 50):
        cb.num_timesteps = step
        cb._on_step()
    assert len(run.list_checkpoints()) == 3


def test_round_checkpoint_callback_round_end(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    model = _StubModel()
    cb = RoundCheckpointCallback(run, save_every_steps=10_000, keep_last=10)
    _bind_callback(cb, model)
    cb.num_timesteps = 1
    cb.locals = {
        "infos": [{"raw": _make_obs(RoundPhase.OVER, kills=2)}],
        "dones": [True],
    }
    cb._on_step()
    assert model.save_count == 1


def test_metrics_callback_round_aggregation(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    channel = ControlChannel(run)
    cb = MetricsCallback(channel, write_every_steps=1)
    _bind_callback(cb, _StubModel())

    cb.num_timesteps = 1
    cb.locals = {
        "infos": [{"raw": _make_obs(RoundPhase.LIVE, kills=1, damage=50)}],
        "rewards": [0.25],
        "dones": [False],
    }
    cb._on_step()
    assert channel.status.current_step == 1

    cb.num_timesteps = 2
    cb.locals = {
        "infos": [{"raw": _make_obs(RoundPhase.OVER, kills=2, damage=120)}],
        "rewards": [1.5],
        "dones": [True],
    }
    cb._on_step()

    assert channel.status.rounds_completed == 1
    assert channel.status.last_round_reward == pytest.approx(0.25 + 1.5)
    assert channel.status.last_kills == 2
    assert channel.status.last_damage == 120
    assert run.status_path.exists()
