"""Rich-based live dashboard for a training run.

Run from a separate terminal::

    cs2-rl-bot dashboard <run_id>

The dashboard polls ``runs/<run_id>/status.json`` and renders four panels:

* **Header** — run id, kind, state.
* **Stats** — current step, rounds played, last reward, rolling reward,
  capture FPS, kill switch.
* **History** — sparkline of the last N round rewards.
* **Footer** — keyboard hotkeys for pause/resume/save/stop.

It also watches stdin for keypresses ``p`` / ``r`` / ``s`` / ``q`` and
writes commands back to ``control.json`` via :class:`DashboardClient`.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cs2_rl_bot.training.control import DashboardClient, TrainingStatus
from cs2_rl_bot.training.run_dir import RunDir, RunRegistry
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


def render(status: TrainingStatus | None, run: RunDir) -> Layout:
    """Pure rendering function — exported for tests / snapshot tooling."""
    layout = Layout(name="root")
    layout.split(
        Layout(_render_header(status, run), name="header", size=3),
        Layout(_render_stats(status), name="body", ratio=2),
        Layout(_render_history(status), name="history", size=10),
        Layout(_render_footer(), name="footer", size=3),
    )
    return layout


def _render_header(status: TrainingStatus | None, run: RunDir) -> Panel:
    state = (status.state if status else "no status yet").upper()
    color = {
        "RUNNING": "green",
        "PAUSED": "yellow",
        "STOPPED": "red",
        "FINISHED": "cyan",
        "ERRORED": "red",
        "PENDING": "magenta",
    }.get(state, "white")
    body = Text.from_markup(
        f"[bold]{run.run_id}[/bold]    state: [bold {color}]{state}[/bold {color}]"
    )
    if status and status.error:
        body.append("\nerror: " + status.error, style="red")
    return Panel(body, title="cs2-rl-bot dashboard", border_style=color)


def _render_stats(status: TrainingStatus | None) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column(justify="left", style="bold")

    if status is None:
        table.add_row("status", "(waiting for trainer)")
        return Panel(table, title="metrics", border_style="dim")

    progress = ""
    if status.total_timesteps > 0:
        pct = 100.0 * status.current_step / status.total_timesteps
        progress = f"  ({pct:.1f}%)"
    table.add_row("step", f"{status.current_step:,} / {status.total_timesteps:,}{progress}")
    table.add_row("rounds played", f"{status.rounds_completed}")
    table.add_row("last round reward", f"{status.last_round_reward:+.3f}")
    table.add_row(
        "rolling reward", f"{status.rolling_reward:+.3f} (n={len(status.recent_rewards)})"
    )
    table.add_row("last K/D/dmg", f"{status.last_kills}/{status.last_deaths}/{status.last_damage}")
    table.add_row("capture fps", f"{status.capture_fps:.1f}")
    table.add_row("gsi updates", f"{status.gsi_updates}")
    table.add_row("kill switch", "[red]ENGAGED[/red]" if status.kill_switch else "OK")
    table.add_row("last checkpoint", status.last_checkpoint or "(none yet)")
    return Panel(table, title="metrics", border_style="cyan")


def _render_history(status: TrainingStatus | None) -> Panel:
    if status is None or not status.recent_rewards:
        return Panel(
            Text("(no rounds yet)", style="dim"), title="recent rewards", border_style="dim"
        )
    spark = _sparkline(status.recent_rewards)
    payload = Text.from_markup(
        f"{spark}\n[dim]min={min(status.recent_rewards):+.2f} "
        f"max={max(status.recent_rewards):+.2f} "
        f"mean={sum(status.recent_rewards) / len(status.recent_rewards):+.2f}[/dim]"
    )
    return Panel(payload, title=f"last {len(status.recent_rewards)} rewards", border_style="green")


def _render_footer() -> Panel:
    return Panel(
        Text.from_markup(
            "[bold]p[/bold] pause   "
            "[bold]r[/bold] resume   "
            "[bold]s[/bold] save now   "
            "[bold]q[/bold] stop training & quit"
        ),
        title="controls",
        border_style="white",
    )


def _sparkline(values: list[float]) -> str:
    """Return a unicode block-character sparkline."""
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    out = []
    for v in values:
        idx = int((v - lo) / span * (len(chars) - 1))
        out.append(chars[max(0, min(len(chars) - 1, idx))])
    return "".join(out)


@contextmanager
def _raw_stdin() -> Iterator[None]:
    """Put stdin in cbreak mode so we can read single keys without enter."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key(timeout: float = 0.0) -> str | None:
    if not sys.stdin.isatty():
        return None
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None
    return sys.stdin.read(1)


def run_dashboard(run_id: str, *, runs_base: str | None = None, refresh_hz: float = 4.0) -> None:
    base = RunRegistry(base=_path_or_default(runs_base))
    run = base.get(run_id)
    if not run.root.exists():
        logger.error("run not found: {}", run_id)
        raise SystemExit(2)

    client = DashboardClient(run)
    console = Console()
    interval = 1.0 / max(refresh_hz, 0.5)

    with (
        _raw_stdin(),
        Live(
            render(None, run), console=console, refresh_per_second=refresh_hz, screen=False
        ) as live,
    ):
        try:
            while True:
                status = client.read_status()
                live.update(render(status, run))
                key = _read_key(timeout=interval)
                if key is None:
                    continue
                if key == "p":
                    client.pause()
                    logger.info("dashboard: pause requested")
                elif key == "r":
                    client.resume()
                    logger.info("dashboard: resume requested")
                elif key == "s":
                    client.request_save()
                    logger.info("dashboard: save requested")
                elif key in ("q", "Q"):
                    client.request_stop()
                    logger.info("dashboard: stop requested -- waiting for trainer to exit")
                    time.sleep(0.5)
                    break
        except KeyboardInterrupt:  # pragma: no cover -- manual abort
            logger.info("dashboard interrupted; not stopping the trainer")


def _path_or_default(value: str | None) -> Path:
    return Path(value) if value else Path("runs")
