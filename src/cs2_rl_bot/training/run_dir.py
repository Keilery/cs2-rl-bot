"""Run directory layout and helpers.

Each training (or BC pretraining, or YOLO finetune) run is materialised as a
self-contained directory under ``runs/<run_id>/`` with the following layout::

    runs/<run_id>/
      ├── meta.json           # run metadata (kind, created_at, parent_run, status)
      ├── config.yaml         # full AppConfig snapshot for reproducibility
      ├── status.json         # live metrics, written by training callbacks
      ├── control.json        # commands from the dashboard (paused, stop, save_now)
      ├── checkpoints/
      │   ├── ppo_step_000010240.zip
      │   └── ppo_step_000020480.zip
      ├── tensorboard/        # SB3 tensorboard event files
      └── logs/
          └── train.log       # mirror of stdout for crash forensics

This module is *intentionally* free of any heavy dependencies so it can be
imported by the dashboard, the CLI menu, and tests without pulling in torch.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RunKind = Literal["ppo", "bc", "yolo"]
RunStatus = Literal["pending", "running", "paused", "stopped", "finished", "errored"]

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


@dataclass(slots=True)
class RunMeta:
    run_id: str
    kind: RunKind
    created_at: float
    status: RunStatus = "pending"
    parent_run_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMeta:
        return cls(
            run_id=str(data["run_id"]),
            kind=data["kind"],
            created_at=float(data["created_at"]),
            status=data.get("status", "pending"),
            parent_run_id=data.get("parent_run_id"),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class RunDir:
    """Filesystem handle for a single run directory."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def control_path(self) -> Path:
        return self.root / "control.json"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root / "tensorboard"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for p in (self.checkpoint_dir, self.tensorboard_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ----- meta.json -------------------------------------------------------

    def read_meta(self) -> RunMeta:
        with self.meta_path.open("r", encoding="utf-8") as f:
            return RunMeta.from_dict(json.load(f))

    def write_meta(self, meta: RunMeta) -> None:
        atomic_write_json(self.meta_path, meta.to_dict())

    def update_status(self, status: RunStatus) -> None:
        meta = self.read_meta()
        meta.status = status
        self.write_meta(meta)

    # ----- checkpoints -----------------------------------------------------

    def list_checkpoints(self) -> list[Path]:
        if not self.checkpoint_dir.exists():
            return []
        return sorted(self.checkpoint_dir.glob("*.zip"))

    def latest_checkpoint(self) -> Path | None:
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None


@dataclass(slots=True)
class RunRegistry:
    """Lazily-evaluated wrapper around the ``runs/`` directory tree."""

    base: Path = field(default_factory=lambda: Path("runs"))

    def ensure(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> list[RunDir]:
        if not self.base.exists():
            return []
        return sorted(
            (RunDir(p) for p in self.base.iterdir() if p.is_dir()),
            key=lambda r: r.root.stat().st_mtime,
            reverse=True,
        )

    def get(self, run_id: str) -> RunDir:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        return RunDir(self.base / run_id)

    def create(
        self,
        kind: RunKind,
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        notes: str = "",
    ) -> RunDir:
        self.ensure()
        rid = run_id or new_run_id(kind)
        run = self.get(rid)
        if run.root.exists():
            raise FileExistsError(f"run already exists: {run.root}")
        run.ensure()
        run.write_meta(
            RunMeta(
                run_id=rid,
                kind=kind,
                created_at=time.time(),
                status="pending",
                parent_run_id=parent_run_id,
                notes=notes,
            )
        )
        return run


def new_run_id(kind: RunKind) -> str:
    """Generate a sortable, human-friendly run id."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suffix = uuid.uuid4().hex[:6]
    return f"{kind}-{ts}-{suffix}"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically by writing to a tempfile and renaming.

    Important on Windows where partial reads can otherwise cause a malformed
    JSON parse in the dashboard.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
