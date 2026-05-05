"""Tests for cs2_rl_bot.training.run_dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cs2_rl_bot.training.run_dir import (
    RunMeta,
    RunRegistry,
    atomic_write_json,
    new_run_id,
)


def test_new_run_id_is_unique() -> None:
    a = new_run_id("ppo")
    b = new_run_id("ppo")
    assert a.startswith("ppo-")
    assert a != b


def test_create_and_list_runs(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    a = registry.create("ppo")
    b = registry.create("bc", parent_run_id=a.run_id)

    runs = registry.list_runs()
    assert {r.run_id for r in runs} == {a.run_id, b.run_id}
    assert (a.root / "meta.json").exists()
    meta = a.read_meta()
    assert meta.kind == "ppo"
    assert meta.status == "pending"

    b_meta = b.read_meta()
    assert b_meta.parent_run_id == a.run_id


def test_update_status_round_trips(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    run.update_status("running")
    assert run.read_meta().status == "running"
    run.update_status("finished")
    assert run.read_meta().status == "finished"


def test_invalid_run_id(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    with pytest.raises(ValueError, match="invalid run_id"):
        registry.get("../etc/passwd")


def test_list_checkpoints_sorted(tmp_path: Path) -> None:
    registry = RunRegistry(base=tmp_path)
    run = registry.create("ppo")
    run.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for step in (10, 20, 5):
        (run.checkpoint_dir / f"ppo_step_{step:09d}.zip").write_bytes(b"fake")
    paths = run.list_checkpoints()
    assert [p.name for p in paths] == [
        "ppo_step_000000005.zip",
        "ppo_step_000000010.zip",
        "ppo_step_000000020.zip",
    ]
    assert run.latest_checkpoint() is not None
    assert run.latest_checkpoint().name == "ppo_step_000000020.zip"  # type: ignore[union-attr]


def test_atomic_write_json(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "status.json"
    atomic_write_json(target, {"ok": True, "value": 1})
    assert json.loads(target.read_text()) == {"ok": True, "value": 1}
    # rewrites cleanly
    atomic_write_json(target, {"ok": False})
    assert json.loads(target.read_text()) == {"ok": False}


def test_run_meta_round_trip() -> None:
    m = RunMeta(run_id="ppo-x-y", kind="ppo", created_at=1.0, status="running")
    rebuilt = RunMeta.from_dict(m.to_dict())
    assert rebuilt == m
