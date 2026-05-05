"""Tests for AppConfig YAML + env-var loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cs2_rl_bot.utils.config import AppConfig


def test_load_defaults_without_yaml(tmp_path: Path) -> None:
    cfg = AppConfig.load(tmp_path / "missing.yaml")
    assert cfg.gsi.port == 3000
    assert cfg.dry_run is True
    assert cfg.agent.algo == "ppo"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("agent:\n  algo: random\n")
    cfg = AppConfig.load(cfg_path)
    assert cfg.agent.algo == "random"


def test_env_vars_override_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("agent:\n  algo: random\n")
    monkeypatch.setenv("CS2BOT_AGENT__ALGO", "scripted")
    monkeypatch.setenv("CS2BOT_DRY_RUN", "false")
    cfg = AppConfig.load(cfg_path)
    assert cfg.agent.algo == "scripted"
    assert cfg.dry_run is False


def test_unknown_env_vars_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNRELATED_VAR", "value")
    cfg = AppConfig.load(Path("/nonexistent.yaml"))
    assert cfg.gsi.host == "127.0.0.1"
    # Smoke check that os module isn't shadowed by the import.
    assert os.path.sep in {"/", "\\"}
