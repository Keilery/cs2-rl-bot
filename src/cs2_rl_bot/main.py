"""CLI entry point exposed as ``cs2-rl-bot`` after ``pip install -e .``.

Subcommands:

* ``cs2-rl-bot gsi``       — start the GSI listener and dump payloads to stdout.
* ``cs2-rl-bot capture``   — preview the screen capture frame size to stdout.
* ``cs2-rl-bot inference`` — run the agent in inference mode against a live game.
* ``cs2-rl-bot train``     — start training (requires explicit risk acknowledgement).
* ``cs2-rl-bot info``      — print the loaded configuration and exit.
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
        _train(
            cfg,
            total_timesteps=total_timesteps,
            acknowledged_risks=i_understand_the_risks,
        )
    except TrainingRefusedError as exc:
        console.print(f"[red]Training refused:[/red] {exc}")
        raise typer.Exit(code=2) from exc


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
