"""Application configuration loaded from YAML + environment overrides.

Configuration is intentionally explicit and serialisable so that runs are
reproducible. `AppConfig.load()` accepts a path to a YAML file (defaulting to
``cfg/default.yaml``) and merges it with environment variables prefixed by
``CS2BOT_``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class GSIConfig(BaseModel):
    """Configuration for the Game State Integration HTTP listener."""

    host: str = "127.0.0.1"
    port: int = Field(3000, ge=1024, le=65535)
    auth_token: str = "CHANGE_ME_DEVIN"
    request_timeout_s: float = 1.0


class CaptureConfig(BaseModel):
    """Screen capture configuration."""

    monitor_index: int = 1  # mss uses 1-based indexing; 0 is "all monitors".
    target_fps: int = Field(30, ge=1, le=240)
    # The model receives a downsampled frame for performance. (H, W) order.
    resize_to: tuple[int, int] = (84, 84)
    grayscale: bool = False


class VisionConfig(BaseModel):
    """Vision pipeline (YOLOv8) configuration."""

    enabled: bool = False
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    device: Literal["cpu", "cuda", "auto"] = "auto"


class ControllerConfig(BaseModel):
    """Mouse / keyboard controller configuration."""

    backend: Literal["pydirectinput", "pynput", "noop"] = "noop"
    mouse_sensitivity: float = Field(2.0, gt=0.0)
    # Maximum mouse delta per step in pixels (clamped before sending to the OS).
    max_mouse_delta: int = Field(200, ge=1)
    # Safety: hard kill switch (drops all input) — toggled by F9 in main loop.
    safety_kill_key: str = "f9"


class AgentConfig(BaseModel):
    """RL agent hyperparameters (PPO)."""

    algo: Literal["ppo", "random", "scripted"] = "ppo"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Path where checkpoints are written between rounds.
    checkpoint_dir: str = "checkpoints/"
    tensorboard_log: str = "tensorboard/"


class RewardConfig(BaseModel):
    """Reward shaping coefficients. Everything is per-step unless noted."""

    kill_bonus: float = 1.0
    death_penalty: float = -1.0
    damage_dealt_coef: float = 0.01
    damage_taken_coef: float = -0.01
    round_win_bonus: float = 2.0
    round_loss_penalty: float = -1.0
    bomb_plant_bonus: float = 0.5
    bomb_defuse_bonus: float = 0.5
    survival_per_step: float = 0.001
    # Penalty applied each step the kill switch is engaged so the agent learns
    # to release input quickly (e.g. when the operator intervenes).
    kill_switch_penalty: float = -0.1


class AppConfig(BaseModel):
    """Top-level application configuration."""

    gsi: GSIConfig = Field(default_factory=GSIConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # When true, the controller is disabled and the agent only observes. This
    # is the default — explicit opt-in is required to actually move the mouse.
    dry_run: bool = True

    @field_validator("dry_run", mode="before")
    @classmethod
    def _coerce_dry_run(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return value

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load configuration from a YAML file, then overlay env vars."""

        path = Path(path) if path else Path("cfg/default.yaml")
        data: dict[str, object] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        # Environment overrides: CS2BOT_<SECTION>__<FIELD>=value
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("CS2BOT_"):
                continue
            path_parts = env_key.removeprefix("CS2BOT_").lower().split("__")
            cursor: dict[str, object] = data
            for part in path_parts[:-1]:
                cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
            cursor[path_parts[-1]] = env_val

        return cls.model_validate(data)
