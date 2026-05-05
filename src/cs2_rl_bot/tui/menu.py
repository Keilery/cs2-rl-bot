"""Interactive menu for managing training runs.

Run ``cs2-rl-bot menu`` to get a Rich-styled lobby that lets the operator:

* See all existing runs and their state.
* Start a new PPO training run (after acknowledging the VAC risk).
* Resume a paused / stopped run from its latest checkpoint.
* Open the live dashboard for a running run.
* Launch BC pretraining over a directory of demos.
* Launch a YOLO fine-tune.

The menu deliberately uses input() / Rich Prompts rather than rendering its
own keyboard-driven UI — this keeps it usable over SSH and in pipelines.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from cs2_rl_bot.training.run_dir import RunDir, RunRegistry
from cs2_rl_bot.utils.logging import logger

console = Console()


def main(runs_base: str | None = None, config_path: str | None = None) -> None:
    base = Path(runs_base) if runs_base else Path("runs")
    registry = RunRegistry(base=base)

    while True:
        console.clear()
        _render_runs(registry)
        action = Prompt.ask(
            "\n[bold]action[/bold]",
            choices=["start", "resume", "dashboard", "bc", "yolo", "delete", "refresh", "quit"],
            default="refresh",
        )
        try:
            if action == "start":
                _start_training(config_path)
            elif action == "resume":
                _resume_training(registry, config_path)
            elif action == "dashboard":
                _attach_dashboard(registry, base)
            elif action == "bc":
                _start_bc(registry)
            elif action == "yolo":
                _start_yolo(registry)
            elif action == "delete":
                _delete_run(registry)
            elif action == "refresh":
                continue
            elif action == "quit":
                return
        except KeyboardInterrupt:
            logger.warning("menu interrupted")
            return
        except Exception as exc:
            console.print(Panel(f"[red]error:[/red] {exc}", border_style="red"))
            time.sleep(2.0)


def _render_runs(registry: RunRegistry) -> None:
    runs = registry.list_runs()
    title = f"runs in [bold]{registry.base}[/bold]"
    table = Table(title=title)
    table.add_column("run id", style="cyan", no_wrap=True)
    table.add_column("kind", style="magenta")
    table.add_column("status", style="green")
    table.add_column("checkpoints", justify="right")
    table.add_column("notes", style="dim")
    if not runs:
        table.add_row("(none)", "—", "—", "—", "—")
    for run in runs:
        try:
            meta = run.read_meta()
        except (OSError, KeyError, ValueError):
            continue
        table.add_row(
            run.run_id,
            meta.kind,
            meta.status,
            str(len(run.list_checkpoints())),
            meta.notes,
        )
    console.print(table)


def _pick_run(registry: RunRegistry, *, prompt_text: str) -> RunDir | None:
    runs = registry.list_runs()
    if not runs:
        console.print("[yellow]no runs yet[/yellow]")
        return None
    run_id = Prompt.ask(prompt_text, default=runs[0].run_id)
    try:
        run = registry.get(run_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    if not run.root.exists():
        console.print(f"[red]run not found: {run_id}[/red]")
        return None
    return run


def _start_training(config_path: str | None) -> None:
    if not Confirm.ask(
        "[red bold]Online play will VAC-ban your account.[/red bold] "
        "Are you sure you want to start a new PPO training run?",
        default=False,
    ):
        return
    timesteps = IntPrompt.ask("total timesteps", default=100_000)
    save_every = IntPrompt.ask("save every N steps", default=10_000)
    from cs2_rl_bot.training.train import train
    from cs2_rl_bot.utils.config import AppConfig

    cfg = AppConfig.load(config_path)
    handle = train(
        cfg,
        total_timesteps=timesteps,
        acknowledged_risks=True,
        save_every_steps=save_every,
    )
    console.print(
        Panel(
            f"training finished — run [cyan]{handle.run.run_id}[/cyan]\n"
            f"final checkpoint: [bold]{handle.final_checkpoint}[/bold]",
            border_style="green",
        )
    )


def _resume_training(registry: RunRegistry, config_path: str | None) -> None:
    run = _pick_run(registry, prompt_text="run id to resume")
    if run is None:
        return
    timesteps = IntPrompt.ask("additional timesteps", default=100_000)
    save_every = IntPrompt.ask("save every N steps", default=10_000)
    from cs2_rl_bot.training.train import train
    from cs2_rl_bot.utils.config import AppConfig

    cfg = AppConfig.load(config_path)
    train(
        cfg,
        total_timesteps=timesteps,
        acknowledged_risks=True,
        save_every_steps=save_every,
        resume_run_id=run.run_id,
    )


def _attach_dashboard(registry: RunRegistry, runs_base: Path) -> None:
    run = _pick_run(registry, prompt_text="run id to attach to")
    if run is None:
        return
    from cs2_rl_bot.tui.dashboard import run_dashboard

    run_dashboard(run.run_id, runs_base=str(runs_base))


def _start_bc(registry: RunRegistry) -> None:
    demo_dir = Prompt.ask("path to a directory containing .dem files")
    epochs = IntPrompt.ask("epochs", default=5)
    target_steam_id = Prompt.ask(
        "filter to a specific steam id (leave empty for everyone)", default=""
    )
    from cs2_rl_bot.training.demo_parser import DemoFilter, parse_demos
    from cs2_rl_bot.training.imitation import BCConfig, train_bc

    paths = sorted(Path(demo_dir).glob("*.dem"))
    if not paths:
        console.print(f"[red]no .dem files found under {demo_dir}[/red]")
        return
    flt = DemoFilter(target_steam_id=target_steam_id or None)
    transitions = parse_demos(paths, demo_filter=flt)
    config = BCConfig(epochs=epochs)
    run, metrics = train_bc(transitions, config=config, runs_base=registry.base)
    console.print(
        Panel(
            f"BC done — run [cyan]{run.run_id}[/cyan]\n"
            f"epochs: {metrics.epoch}  train: {metrics.train_loss:.4f}  "
            f"val: {metrics.val_loss:.4f}  samples: {metrics.samples}",
            border_style="green",
        )
    )


def _start_yolo(registry: RunRegistry) -> None:
    dataset_yaml = Prompt.ask("path to YOLO dataset data.yaml")
    base_model = Prompt.ask("base model", default="yolov8n.pt")
    epochs = IntPrompt.ask("epochs", default=60)
    batch = IntPrompt.ask("batch size", default=16)
    from cs2_rl_bot.training.yolo_finetune import (
        YOLOFineTuneConfig,
        finetune_yolo,
    )

    cfg = YOLOFineTuneConfig(base_model=base_model, epochs=epochs, batch_size=batch)
    run = finetune_yolo(Path(dataset_yaml), config=cfg, runs_base=registry.base)
    console.print(
        Panel(
            f"YOLO fine-tune done — run [cyan]{run.run_id}[/cyan]",
            border_style="green",
        )
    )


def _delete_run(registry: RunRegistry) -> None:
    run = _pick_run(registry, prompt_text="run id to delete")
    if run is None:
        return
    if not Confirm.ask(f"[red]delete run {run.run_id}?[/red] This is irreversible."):
        return
    import shutil

    shutil.rmtree(run.root)
    console.print(f"[yellow]deleted {run.run_id}[/yellow]")
