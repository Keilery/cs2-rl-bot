"""Tests for cs2_rl_bot.training.control."""

from __future__ import annotations

from pathlib import Path

from cs2_rl_bot.training.control import (
    ControlChannel,
    ControlSignal,
    DashboardClient,
    TrainingStatus,
    load_control,
    load_status,
    reset_control,
)
from cs2_rl_bot.training.run_dir import RunRegistry


def test_status_round_trip() -> None:
    s = TrainingStatus(run_id="ppo-x", state="running", current_step=5, recent_rewards=[1.0, 2.0])
    rebuilt = TrainingStatus.from_dict(s.to_dict())
    assert rebuilt == s


def test_control_signal_round_trip() -> None:
    sig = ControlSignal(paused=True, save_now=True, revision=4, updated_at=1.5)
    rebuilt = ControlSignal.from_dict(sig.to_dict())
    assert rebuilt == sig


def test_dashboard_client_writes_control(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")

    dash = DashboardClient(run)
    dash.pause()
    sig = load_control(run)
    assert sig.paused is True
    assert sig.revision == 1

    dash.request_save()
    sig = load_control(run)
    assert sig.save_now is True
    assert sig.revision == 2

    dash.request_stop()
    sig = load_control(run)
    assert sig.stop_requested is True


def test_channel_writes_status(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    channel = ControlChannel(run, rolling_window=3)
    channel.status.current_step = 100
    channel.status.total_timesteps = 1000
    channel.push_round_reward(1.0)
    channel.push_round_reward(2.0)
    channel.push_round_reward(3.0)
    channel.push_round_reward(4.0)  # falls outside window
    channel.write_status()

    status = load_status(run)
    assert status is not None
    assert status.current_step == 100
    assert status.rounds_completed == 4
    assert status.recent_rewards == [2.0, 3.0, 4.0]
    assert status.rolling_reward == 3.0


def test_fresh_signal_dedupes(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    dash = DashboardClient(run)
    channel = ControlChannel(run)

    assert channel.fresh_signal() is None  # no signal file yet

    dash.pause()
    sig = channel.fresh_signal()
    assert sig is not None and sig.paused

    # Second poll without dashboard changes should return None.
    assert channel.fresh_signal() is None

    dash.resume()
    sig = channel.fresh_signal()
    assert sig is not None and sig.paused is False


def test_reset_control_removes_file(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    DashboardClient(run).pause()
    assert run.control_path.exists()
    reset_control(run)
    assert not run.control_path.exists()
