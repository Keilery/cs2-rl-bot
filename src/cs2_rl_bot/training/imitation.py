"""Behaviour-cloning pretraining over demo transitions.

Strategy:

1. Build a (small) MLP that maps the *scalar* observation head to the
   discrete action heads (movement / attack / jump / crouch / reload / weapon)
   plus a regression head for the mouse vector.
2. Train it with Adam + cross-entropy / MSE losses on demo data.
3. Save the trained weights as a SB3-compatible warm-start checkpoint by
   loading them into a fresh PPO policy and dumping that. The PPO trainer
   then resumes from this checkpoint with ``--resume`` (see
   ``cs2_rl_bot.training.train``).

We do **not** attempt to clone the CNN frame head from demos because
demoparser2 doesn't surface frames. The frame head will be initialised from
scratch when PPO takes over — but the action distribution priors learned
from demos still give a substantial sample-efficiency boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.env.cs2_env import encode_scalars
from cs2_rl_bot.training.demo_parser import DemoTransition
from cs2_rl_bot.training.run_dir import RunDir, RunRegistry
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(slots=True)
class BCConfig:
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_sizes: tuple[int, ...] = (128, 128)
    val_fraction: float = 0.1
    device: str = "auto"
    log_every: int = 50


@dataclass(slots=True)
class BCMetrics:
    """Training-time metrics for the dashboard / tests."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    samples: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


def collect_dataset(transitions: Iterable[DemoTransition]) -> dict[str, np.ndarray]:
    """Materialise a dataset dictionary from an iterable of transitions."""
    scalars: list[np.ndarray] = []
    movement: list[int] = []
    attack: list[int] = []
    jump: list[int] = []
    crouch: list[int] = []
    reload_: list[int] = []
    weapon: list[int] = []
    mouse: list[np.ndarray] = []
    weights: list[float] = []

    for tx in transitions:
        try:
            scalars.append(encode_scalars(tx.observation))
        except (ValueError, KeyError):
            continue
        movement.append(int(tx.action.get("movement", Movement.STAND)))
        attack.append(int(tx.action.get("attack", 0)))
        jump.append(int(tx.action.get("jump", 0)))
        crouch.append(int(tx.action.get("crouch", 0)))
        reload_.append(int(tx.action.get("reload", 0)))
        weapon.append(int(tx.action.get("weapon", WeaponSlot.NONE)))
        mouse.append(np.asarray(tx.action.get("mouse", [0.0, 0.0]), dtype=np.float32))
        weights.append(float(tx.weight))

    if not scalars:
        raise ValueError("no transitions produced any scalar observations")

    return {
        "scalars": np.stack(scalars).astype(np.float32),
        "movement": np.asarray(movement, dtype=np.int64),
        "attack": np.asarray(attack, dtype=np.int64),
        "jump": np.asarray(jump, dtype=np.int64),
        "crouch": np.asarray(crouch, dtype=np.int64),
        "reload": np.asarray(reload_, dtype=np.int64),
        "weapon": np.asarray(weapon, dtype=np.int64),
        "mouse": np.stack(mouse).astype(np.float32),
        "weight": np.asarray(weights, dtype=np.float32),
    }


def train_bc(
    transitions: Iterable[DemoTransition],
    *,
    config: BCConfig | None = None,
    runs_base: Path | None = None,
    parent_run_id: str | None = None,
) -> tuple[RunDir, BCMetrics]:
    """Train a small BC model and persist it as a run directory."""
    config = config or BCConfig()
    registry = RunRegistry(base=runs_base or Path("runs"))
    run = registry.create(kind="bc", parent_run_id=parent_run_id)
    run.update_status("running")

    try:
        dataset = collect_dataset(transitions)
        run.update_status("running")
        metrics = _train_torch(dataset, config, run)
        run.update_status("finished")
        logger.info(
            "BC complete: epochs={} train_loss={:.4f} val_loss={:.4f} samples={}",
            metrics.epoch,
            metrics.train_loss,
            metrics.val_loss,
            metrics.samples,
        )
    except Exception as exc:
        run.update_status("errored")
        logger.error("BC training failed: {}", exc)
        raise
    return run, metrics


def _train_torch(
    dataset: dict[str, np.ndarray],
    config: BCConfig,
    run: RunDir,
) -> BCMetrics:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    n = dataset["scalars"].shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 transitions, got {n}")

    device = _resolve_device(config.device)
    logger.info("training BC on {} samples (device={})", n, device)

    tensors = {k: torch.from_numpy(v) for k, v in dataset.items()}
    tds = TensorDataset(
        tensors["scalars"],
        tensors["movement"],
        tensors["mouse"],
        tensors["attack"],
        tensors["jump"],
        tensors["crouch"],
        tensors["reload"],
        tensors["weapon"],
        tensors["weight"],
    )
    val_n = max(1, round(n * config.val_fraction))
    train_n = n - val_n
    train_ds, val_ds = random_split(tds, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    scalar_dim = tensors["scalars"].shape[1]
    model = _build_model(scalar_dim, config.hidden_sizes).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    metrics = BCMetrics()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            scalars, mv, mouse, atk, jmp, crch, rld, wpn, w = (b.to(device) for b in batch)
            preds = model(scalars)
            loss = (
                (ce(preds["movement"], mv) * w).mean()
                + (mse(preds["mouse"], mouse).mean(dim=-1) * w).mean()
                + (bce(preds["attack"], atk.float()) * w).mean()
                + (bce(preds["jump"], jmp.float()) * w).mean()
                + (bce(preds["crouch"], crch.float()) * w).mean()
                + (bce(preds["reload"], rld.float()) * w).mean()
                + (ce(preds["weapon"], wpn) * w).mean()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * scalars.shape[0]
            train_samples += scalars.shape[0]

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                scalars, mv, mouse, atk, jmp, crch, rld, wpn, w = (b.to(device) for b in batch)
                preds = model(scalars)
                loss = (
                    (ce(preds["movement"], mv) * w).mean()
                    + (mse(preds["mouse"], mouse).mean(dim=-1) * w).mean()
                    + (bce(preds["attack"], atk.float()) * w).mean()
                    + (bce(preds["jump"], jmp.float()) * w).mean()
                    + (bce(preds["crouch"], crch.float()) * w).mean()
                    + (bce(preds["reload"], rld.float()) * w).mean()
                    + (ce(preds["weapon"], wpn) * w).mean()
                )
                val_loss += float(loss.item()) * scalars.shape[0]
                val_samples += scalars.shape[0]

        train_avg = train_loss / max(train_samples, 1)
        val_avg = val_loss / max(val_samples, 1)
        metrics.epoch = epoch
        metrics.train_loss = train_avg
        metrics.val_loss = val_avg
        metrics.samples = n
        metrics.history.append({"epoch": epoch, "train_loss": train_avg, "val_loss": val_avg})
        logger.info(
            "epoch {}/{}: train_loss={:.4f} val_loss={:.4f}",
            epoch,
            config.epochs,
            train_avg,
            val_avg,
        )

    weights_path = run.checkpoint_dir / "bc_policy.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scalar_dim": scalar_dim}, weights_path)
    return metrics


def _build_model(scalar_dim: int, hidden_sizes: tuple[int, ...]) -> Any:
    import torch
    from torch import nn

    class BCPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = scalar_dim
            for h in hidden_sizes:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.movement_head = nn.Linear(prev, len(Movement))
            self.weapon_head = nn.Linear(prev, len(WeaponSlot))
            self.mouse_head = nn.Linear(prev, 2)
            self.attack_head = nn.Linear(prev, 1)
            self.jump_head = nn.Linear(prev, 1)
            self.crouch_head = nn.Linear(prev, 1)
            self.reload_head = nn.Linear(prev, 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            z = self.trunk(x)
            return {
                "movement": self.movement_head(z),
                "weapon": self.weapon_head(z),
                "mouse": torch.tanh(self.mouse_head(z)),
                "attack": self.attack_head(z).squeeze(-1),
                "jump": self.jump_head(z).squeeze(-1),
                "crouch": self.crouch_head(z).squeeze(-1),
                "reload": self.reload_head(z).squeeze(-1),
            }

    return BCPolicy()


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
