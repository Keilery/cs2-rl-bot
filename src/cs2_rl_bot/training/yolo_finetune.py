"""YOLOv8 fine-tuning helper for CS2 enemy detection.

The function :func:`finetune_yolo` is a thin wrapper around
``ultralytics.YOLO.train`` that:

1. Optionally pulls a public dataset from Roboflow Universe (requires
   ``ROBOFLOW_API_KEY``). The dataset is cached on disk so subsequent runs
   reuse it.
2. Persists the run as a :class:`RunDir` so the dashboard can attach to it
   and surface progress via ``status.json``.
3. Stores the resulting weights under ``runs/<run_id>/checkpoints/yolo.pt``
   and updates ``cfg/default.yaml``-compatible config snippet for the user
   to copy.

Real fine-tuning requires a GPU and a labelled dataset. We do **not** run
this in CI — the helpers are designed so that a developer with a laptop GPU
can launch them with a single ``cs2-rl-bot train-yolo`` command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cs2_rl_bot.training.run_dir import RunDir, RunRegistry
from cs2_rl_bot.utils.logging import logger


@dataclass(slots=True)
class YOLOFineTuneConfig:
    base_model: str = "yolov8n.pt"  # n / s / m / l
    epochs: int = 60
    image_size: int = 640
    batch_size: int = 16
    device: str = "0"  # "0" -> first CUDA device, "cpu" otherwise
    patience: int = 20
    workers: int = 4


@dataclass(slots=True)
class RoboflowSource:
    workspace: str
    project: str
    version: int
    fmt: str = "yolov8"
    api_key_env: str = "ROBOFLOW_API_KEY"


def download_roboflow_dataset(source: RoboflowSource, *, cache_dir: Path) -> Path:
    """Download a Roboflow Universe dataset to ``cache_dir``."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover — optional extra
        raise RuntimeError(
            "roboflow not installed. Install it via `pip install -e '.[vision]'`."
        ) from exc

    api_key = os.environ.get(source.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"environment variable {source.api_key_env} is not set. "
            "Get an API key at https://roboflow.com/account/api-keys."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source.workspace}__{source.project}__v{source.version}"
    if (target / "data.yaml").exists():
        logger.info("using cached Roboflow dataset at {}", target)
        return target

    logger.info(
        "downloading Roboflow dataset {}/{} v{} -> {}",
        source.workspace,
        source.project,
        source.version,
        target,
    )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source.workspace).project(source.project)
    dataset = project.version(source.version).download(source.fmt, location=str(target))
    return Path(dataset.location)


def finetune_yolo(
    dataset_yaml: Path,
    *,
    config: YOLOFineTuneConfig | None = None,
    runs_base: Path | None = None,
    parent_run_id: str | None = None,
) -> RunDir:
    """Run a YOLOv8 fine-tune. Returns the run directory."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover — optional extra
        raise RuntimeError(
            "ultralytics not installed. Install it via `pip install -e '.[vision]'`."
        ) from exc

    config = config or YOLOFineTuneConfig()
    registry = RunRegistry(base=runs_base or Path("runs"))
    run = registry.create(kind="yolo", parent_run_id=parent_run_id)
    run.update_status("running")

    logger.info(
        "fine-tuning {} on {} (epochs={}, imgsz={}, batch={})",
        config.base_model,
        dataset_yaml,
        config.epochs,
        config.image_size,
        config.batch_size,
    )
    try:
        model = YOLO(config.base_model)
        model.train(
            data=str(dataset_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            patience=config.patience,
            workers=config.workers,
            project=str(run.tensorboard_dir),
            name="yolo",
            exist_ok=True,
        )
        weights_src = _locate_best_weights(run.tensorboard_dir / "yolo")
        if weights_src is not None:
            target = run.checkpoint_dir / "yolo.pt"
            target.write_bytes(weights_src.read_bytes())
            _write_yaml_snippet(run, target)
            logger.info("yolo weights -> {}", target)
        run.update_status("finished")
    except Exception:
        run.update_status("errored")
        raise
    return run


def _locate_best_weights(yolo_run_dir: Path) -> Path | None:
    candidates = [yolo_run_dir / "weights" / "best.pt", yolo_run_dir / "weights" / "last.pt"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _write_yaml_snippet(run: RunDir, weights: Path) -> None:
    snippet: dict[str, Any] = {
        "vision": {
            "enabled": True,
            "model_path": str(weights),
        }
    }
    (run.root / "vision_config.yaml").write_text(
        yaml.safe_dump(snippet, sort_keys=False),
        encoding="utf-8",
    )
