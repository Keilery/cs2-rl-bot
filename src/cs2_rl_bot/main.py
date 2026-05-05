"""CLI entry point exposed as ``cs2-rl-bot`` after ``pip install -e .``.

Subcommands:

* ``cs2-rl-bot gsi``       -- start the GSI listener and dump payloads.
* ``cs2-rl-bot capture``   -- preview the screen capture frame size.
* ``cs2-rl-bot inference`` -- run the agent in inference mode.
* ``cs2-rl-bot train``     -- start training (requires risk ack).
* ``cs2-rl-bot info``      -- print the loaded configuration and exit.
* ``cs2-rl-bot dump-cfg``  -- write the default configuration to YAML.
* ``cs2-rl-bot dashboard`` -- attach a live monitoring dashboard to a run.
* ``cs2-rl-bot menu``      -- interactive menu for managing all runs.
* ``cs2-rl-bot pretrain-bc`` -- behaviour-clone from .dem files.
* ``cs2-rl-bot parse-demo`` -- inspect a single demo file.
* ``cs2-rl-bot train-yolo`` -- fine-tune YOLOv8 for enemy detection.
* ``cs2-rl-bot runs``      -- list training runs and their state.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from cs2_rl_bot.observation.gsi_server import GSIState, run_server
from cs2_rl_bot.utils.config import AppConfig
from cs2_rl_bot.utils.logging import configure_logging, logger

app = typer.Typer(
    add_completion=False,
    help="Reinforcement-learning agent scaffold for Counter-Strike 2.",
    no_args_is_help=True,
)
console = Console()


def _load_config(config_path: Path | None) -> AppConfig:
    cfg = AppConfig.load(config_path)
    configure_logging(cfg.log_level)
    return cfg


@app.command()
def info(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Print the loaded configuration."""
    cfg = _load_config(config)
    console.print_json(data=cfg.model_dump())


@app.command()
def gsi(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Run the GSI listener standalone — useful for verifying CS2 setup."""
    cfg = _load_config(config)
    state = GSIState()

    async def _runner() -> None:
        server_task = asyncio.create_task(run_server(cfg.gsi, state))
        last = -1
        try:
            while True:
                await state.wait_for_update(timeout=1.0)
                if state.update_count != last:
                    last = state.update_count
                    payload = await state.get()
                    logger.info("update #{}: keys={}", last, sorted(payload.keys()))
                    console.print_json(data=payload)
        finally:
            server_task.cancel()

    asyncio.run(_runner())


@app.command()
def capture(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Capture a single frame and report its shape (no preview window)."""
    from cs2_rl_bot.observation.screen_capture import ScreenCapture

    cfg = _load_config(config)
    cap = ScreenCapture(cfg.capture)
    cap.start()
    try:
        frame = cap.latest(timeout=2.0)
        if frame is None:
            console.print("[red]No frame captured — is a display attached?[/red]")
            raise typer.Exit(code=1)
        console.print(
            f"[green]Captured frame: shape={frame.shape}, dtype={frame.image.dtype}, "
            f"timestamp={frame.timestamp:.3f}[/green]"
        )
    finally:
        cap.stop()


@app.command()
def inference(
    config: Path | None = typer.Option(None, "--config", "-c"),
    max_steps: int = typer.Option(1000, "--max-steps"),
) -> None:
    """Run the agent against the running game (no learning)."""
    from cs2_rl_bot.training.inference import run_inference

    cfg = _load_config(config)
    run_inference(cfg, max_steps=max_steps)


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", "-c"),
    total_timesteps: int = typer.Option(100_000, "--steps"),
    save_every: int = typer.Option(10_000, "--save-every"),
    resume: str | None = typer.Option(None, "--resume", help="Run id to resume."),
    i_understand_the_risks: bool = typer.Option(
        False,
        "--i-understand-the-risks",
        help="Acknowledge that running against VAC servers will get you banned.",
    ),
) -> None:
    """Start PPO training. Requires explicit acknowledgement of the VAC risk."""
    from cs2_rl_bot.training.train import TrainingRefusedError
    from cs2_rl_bot.training.train import train as _train

    cfg = _load_config(config)
    try:
        handle = _train(
            cfg,
            total_timesteps=total_timesteps,
            acknowledged_risks=i_understand_the_risks,
            save_every_steps=save_every,
            resume_run_id=resume,
        )
    except TrainingRefusedError as exc:
        console.print(f"[red]Training refused:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]run {handle.run.run_id} done[/green] -> {handle.final_checkpoint}")


@app.command()
def runs() -> None:
    """List all runs in ./runs/."""
    from rich.table import Table

    from cs2_rl_bot.training.run_dir import RunRegistry

    registry = RunRegistry()
    table = Table(title="runs")
    table.add_column("run id")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("checkpoints", justify="right")
    for run in registry.list_runs():
        try:
            meta = run.read_meta()
        except (OSError, ValueError, KeyError):
            continue
        table.add_row(run.run_id, meta.kind, meta.status, str(len(run.list_checkpoints())))
    console.print(table)


@app.command()
def dashboard(
    run_id: str = typer.Argument(..., help="Run id to attach to."),
    runs_base: str = typer.Option("runs", "--runs-base"),
    refresh_hz: float = typer.Option(4.0, "--refresh-hz"),
) -> None:
    """Live monitoring dashboard with pause / resume / save / stop hotkeys."""
    from cs2_rl_bot.tui.dashboard import run_dashboard

    run_dashboard(run_id, runs_base=runs_base, refresh_hz=refresh_hz)


@app.command()
def menu(
    config: Path | None = typer.Option(None, "--config", "-c"),
    runs_base: str = typer.Option("runs", "--runs-base"),
) -> None:
    """Interactive menu for managing all training runs."""
    from cs2_rl_bot.tui.menu import main as _main

    _main(runs_base=runs_base, config_path=str(config) if config else None)


@app.command(name="parse-demo")
def parse_demo(
    demo: Path = typer.Argument(..., exists=True, dir_okay=False),
    target_steam_id: str | None = typer.Option(None, "--steamid"),
    max_ticks: int = typer.Option(20, "--max-ticks"),
) -> None:
    """Inspect transitions extracted from a .dem file."""
    from cs2_rl_bot.training.demo_parser import DemoFilter, DemoParser

    flt = DemoFilter(target_steam_id=target_steam_id, max_ticks=max_ticks)
    parser = DemoParser(demo, demo_filter=flt)
    for tx in parser.iter_transitions():
        console.print_json(
            data={
                "tick": tx.tick,
                "action": {
                    k: int(v)
                    if hasattr(v, "__int__") and not hasattr(v, "tolist")
                    else v.tolist()
                    if hasattr(v, "tolist")
                    else v
                    for k, v in tx.action.items()
                },
            }
        )


@app.command(name="pretrain-bc")
def pretrain_bc(
    demo_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    epochs: int = typer.Option(5, "--epochs"),
    steam_id: str | None = typer.Option(None, "--steamid"),
    runs_base: str = typer.Option("runs", "--runs-base"),
) -> None:
    """Behaviour-clone a policy on .dem files for warm-starting PPO."""
    from cs2_rl_bot.training.demo_parser import DemoFilter, parse_demos
    from cs2_rl_bot.training.imitation import BCConfig, train_bc

    paths = sorted(demo_dir.glob("*.dem"))
    if not paths:
        console.print(f"[red]no .dem files in {demo_dir}[/red]")
        raise typer.Exit(code=2)
    flt = DemoFilter(target_steam_id=steam_id)
    transitions = parse_demos(paths, demo_filter=flt)
    cfg = BCConfig(epochs=epochs)
    run, metrics = train_bc(transitions, config=cfg, runs_base=Path(runs_base))
    console.print(
        f"[green]BC done -> {run.run_id}[/green] "
        f"epochs={metrics.epoch} train={metrics.train_loss:.4f} "
        f"val={metrics.val_loss:.4f} samples={metrics.samples}"
    )


@app.command(name="train-yolo")
def train_yolo(
    dataset_yaml: Path = typer.Argument(..., exists=True),
    base_model: str = typer.Option("yolov8n.pt", "--base-model"),
    epochs: int = typer.Option(60, "--epochs"),
    batch: int = typer.Option(16, "--batch"),
    runs_base: str = typer.Option("runs", "--runs-base"),
) -> None:
    """Fine-tune YOLOv8 for CS2 enemy detection."""
    from cs2_rl_bot.training.yolo_finetune import YOLOFineTuneConfig, finetune_yolo

    cfg = YOLOFineTuneConfig(base_model=base_model, epochs=epochs, batch_size=batch)
    run = finetune_yolo(dataset_yaml, config=cfg, runs_base=Path(runs_base))
    console.print(f"[green]YOLO done -> {run.run_id}[/green]")


@app.command(name="dump-cfg")
def dump_cfg(out: Path = typer.Argument(Path("cfg/default.yaml"))) -> None:
    """Write the default configuration to YAML for editing."""
    import yaml

    cfg = AppConfig()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    console.print(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
