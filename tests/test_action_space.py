"""Tests for action space + controller (noop backend)."""

from __future__ import annotations

import numpy as np

from cs2_rl_bot.action.action_space import (
    MOVEMENT_KEYS,
    Movement,
    WeaponSlot,
    build_action_space,
)
from cs2_rl_bot.action.controller import Controller
from cs2_rl_bot.utils.config import ControllerConfig


def test_action_space_sample_has_expected_keys() -> None:
    space = build_action_space()
    sample = space.sample()
    assert set(sample.keys()) == {
        "movement",
        "mouse",
        "attack",
        "jump",
        "crouch",
        "reload",
        "weapon",
    }
    assert sample["mouse"].shape == (2,)
    assert 0 <= int(sample["movement"]) < len(Movement)
    assert 0 <= int(sample["weapon"]) < len(WeaponSlot)


def test_movement_keys_complete() -> None:
    assert set(MOVEMENT_KEYS.keys()) == set(Movement)


def test_controller_dry_run_is_noop_safe() -> None:
    ctrl = Controller(ControllerConfig(), dry_run=True)
    action = {
        "movement": int(Movement.FORWARD),
        "mouse": np.array([0.5, -0.25], dtype=np.float32),
        "attack": 1,
        "jump": 0,
        "crouch": 0,
        "reload": 0,
        "weapon": int(WeaponSlot.PRIMARY),
    }
    ctrl.apply(action)  # must not raise
    ctrl.release_all()


def test_controller_kill_switch_blocks_actions() -> None:
    ctrl = Controller(ControllerConfig(), dry_run=True)
    ctrl.engage_kill_switch()
    assert ctrl.kill_switch_engaged is True

    action = {
        "movement": int(Movement.FORWARD),
        "mouse": np.zeros(2, dtype=np.float32),
        "attack": 1,
        "jump": 0,
        "crouch": 0,
        "reload": 0,
        "weapon": 0,
    }
    ctrl.apply(action)  # blocked, must not raise
    ctrl.clear_kill_switch()
    assert ctrl.kill_switch_engaged is False
