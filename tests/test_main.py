"""Tests for the typer CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cs2_rl_bot.main import app


def test_info_prints_config(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ppo" in result.stdout


def test_dump_cfg_writes_file(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "out.yaml"
    result = runner.invoke(app, ["dump-cfg", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "auth_token" in content
