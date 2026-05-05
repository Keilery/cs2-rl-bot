"""Tests for cs2_rl_bot.tui.dashboard rendering."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from cs2_rl_bot.training.control import TrainingStatus
from cs2_rl_bot.training.run_dir import RunRegistry
from cs2_rl_bot.tui.dashboard import (
    _render_footer,
    _render_header,
    _render_history,
    _render_stats,
    _sparkline,
    render,
)


def _capture(renderable, *, width: int = 200) -> str:  # type: ignore[no-untyped-def]
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_sparkline_handles_empty() -> None:
    assert _sparkline([]) == ""


def test_sparkline_distinct_chars() -> None:
    spark = _sparkline([0.0, 1.0, 2.0, 3.0])
    assert len(spark) == 4
    assert spark[0] != spark[-1]


def test_full_layout_renders_no_status(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    layout = render(None, run)
    # the layout itself is a Rich renderable; rendering must not raise.
    text = _capture(layout)
    assert "controls" in text


def test_individual_panels_with_status(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    status = TrainingStatus(
        run_id=run.run_id,
        state="running",
        current_step=2_500,
        total_timesteps=10_000,
        rounds_completed=12,
        last_round_reward=1.5,
        rolling_reward=0.75,
        last_kills=3,
        last_deaths=1,
        last_damage=180,
        capture_fps=27.4,
        gsi_updates=412,
        kill_switch=False,
        recent_rewards=[0.0, 0.5, 1.0, 1.5, 2.0],
        last_checkpoint="ppo_step_000002500.zip",
    )
    header = _capture(_render_header(status, run))
    assert "RUNNING" in header
    assert run.run_id in header

    stats = _capture(_render_stats(status))
    assert "2,500" in stats
    assert "10,000" in stats
    assert "ppo_step_000002500.zip" in stats
    assert "27.4" in stats
    assert "3/1/180" in stats

    history = _capture(_render_history(status))
    assert "min=" in history
    assert "max=" in history

    footer = _capture(_render_footer())
    assert "pause" in footer
    assert "resume" in footer
