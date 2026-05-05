"""Tests for cs2_rl_bot.training.imitation."""

from __future__ import annotations

import numpy as np
import pytest

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.observation.state import (
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.training.demo_parser import DemoTransition
from cs2_rl_bot.training.imitation import BCConfig, collect_dataset, train_bc


def _tx(*, kills: int = 0, mv: Movement = Movement.FORWARD, mouse=(0.1, -0.2)) -> DemoTransition:
    return DemoTransition(
        observation=Observation(
            player=PlayerState(health=100, team=Team.T, round_kills=kills),
            round=RoundState(phase=RoundPhase.LIVE),
            frame=None,
            raw_gsi={},
        ),
        action={
            "movement": int(mv),
            "mouse": np.array(mouse, dtype=np.float32),
            "attack": 1 if kills else 0,
            "jump": 0,
            "crouch": 0,
            "reload": 0,
            "weapon": int(WeaponSlot.PRIMARY),
        },
        tick=kills,
    )


def test_collect_dataset_shapes() -> None:
    txs = [_tx(kills=k) for k in range(8)]
    ds = collect_dataset(txs)
    assert ds["scalars"].shape == (8, 14)
    assert ds["movement"].shape == (8,)
    assert ds["mouse"].shape == (8, 2)
    assert ds["attack"].dtype == np.int64
    assert (ds["weapon"] == int(WeaponSlot.PRIMARY)).all()


def test_collect_dataset_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no transitions"):
        collect_dataset(iter(()))


def test_train_bc_smoke(tmp_path) -> None:
    transitions = [_tx(kills=k) for k in range(32)]
    config = BCConfig(epochs=2, batch_size=8, hidden_sizes=(16,), val_fraction=0.25)
    run, metrics = train_bc(transitions, config=config, runs_base=tmp_path)

    assert run.read_meta().status == "finished"
    assert metrics.epoch == 2
    assert metrics.samples == 32
    weights_path = run.checkpoint_dir / "bc_policy.pt"
    assert weights_path.exists()
    assert metrics.history and metrics.history[-1]["epoch"] == 2
