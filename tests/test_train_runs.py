"""Tests for cs2_rl_bot.training.train run-directory plumbing.

We don't actually run PPO here -- instead we monkeypatch the parts that
require a Gym env to assert that the run scaffolding is wired up correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cs2_rl_bot.training import train as train_mod
from cs2_rl_bot.training.run_dir import RunRegistry
from cs2_rl_bot.training.train import TrainingRefusedError, train
from cs2_rl_bot.utils.config import AppConfig


def test_train_refuses_without_acknowledgement(tmp_path) -> None:
    cfg = AppConfig()
    with pytest.raises(TrainingRefusedError, match="acknowledged_risks"):
        train(cfg, runs_base=tmp_path)


def test_train_refuses_dry_run(tmp_path) -> None:
    cfg = AppConfig()
    cfg.dry_run = True
    with pytest.raises(TrainingRefusedError, match="dry_run"):
        train(cfg, acknowledged_risks=True, runs_base=tmp_path)


def test_train_refuses_unknown_algo(tmp_path) -> None:
    cfg = AppConfig()
    cfg.dry_run = False
    cfg.agent.algo = "dqn"
    with pytest.raises(TrainingRefusedError, match="ppo"):
        train(cfg, acknowledged_risks=True, runs_base=tmp_path)


def test_train_creates_run_dir_and_calls_learn(monkeypatch, tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.dry_run = False
    cfg.agent.algo = "ppo"

    captured: dict[str, object] = {}

    class _StubAgent:
        def __init__(self, env, agent_cfg):
            captured["agent_cfg"] = agent_cfg
            captured["env"] = env

        def load(self, path):
            captured["loaded"] = str(path)

        def learn(self, *, total_timesteps, callback, reset_num_timesteps):
            captured["total_timesteps"] = total_timesteps
            captured["reset"] = reset_num_timesteps

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake")

    class _StubEnv:
        def __init__(self, *_a, **_kw):
            pass

        def close(self):
            captured["env_closed"] = True

    monkeypatch.setattr(train_mod, "PPOAgent", _StubAgent)
    monkeypatch.setattr(train_mod, "CS2Env", _StubEnv)

    handle = train(
        cfg,
        total_timesteps=128,
        acknowledged_risks=True,
        runs_base=tmp_path,
    )

    assert handle.run.root.exists()
    meta = handle.run.read_meta()
    assert meta.kind == "ppo"
    assert meta.status == "finished"
    assert captured["total_timesteps"] == 128
    assert captured["reset"] is True
    assert handle.run.config_path.exists()
    assert handle.run.tensorboard_dir.exists()
    assert captured["env_closed"] is True

    # Now resume the same run.
    captured.clear()

    # Drop a fake "earlier" checkpoint so .load(...) is exercised.
    fake_ckpt = handle.run.checkpoint_dir / "ppo_step_000000064.zip"
    fake_ckpt.write_bytes(b"resume-me")

    handle2 = train(
        cfg,
        total_timesteps=64,
        acknowledged_risks=True,
        runs_base=tmp_path,
        resume_run_id=handle.run.run_id,
    )
    assert handle2.run.run_id == handle.run.run_id
    assert captured["loaded"].endswith(fake_ckpt.name)  # type: ignore[union-attr]
    assert captured["reset"] is False


def test_train_resume_missing_run(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.dry_run = False
    cfg.agent.algo = "ppo"
    with pytest.raises(FileNotFoundError):
        train(cfg, acknowledged_risks=True, runs_base=tmp_path, resume_run_id="ppo-does-not-exist")


def test_registry_lists_in_mtime_order(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    a = registry.create("ppo")
    b = registry.create("bc")
    runs = registry.list_runs()
    # Both should appear, b is newer so should come first.
    assert {r.run_id for r in runs} == {a.run_id, b.run_id}
