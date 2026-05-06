"""Tests for :class:`FlatActionWrapper`."""

from __future__ import annotations

import time

import numpy as np
import pytest
from gymnasium import spaces

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.action.controller import Controller
from cs2_rl_bot.env.cs2_env import CS2Env
from cs2_rl_bot.env.flat_action import DEFAULT_MOUSE_BINS, FlatActionWrapper
from cs2_rl_bot.observation.state import (
    Frame,
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.utils.config import AppConfig


class _StubProvider:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest(self) -> Observation:
        return Observation(
            player=PlayerState(health=100, team=Team.T),
            round=RoundState(phase=RoundPhase.LIVE, round_number=1),
            frame=Frame(
                image=np.zeros((84, 84, 3), dtype=np.uint8),
                timestamp=time.time(),
            ),
        )


def _make_env() -> FlatActionWrapper:
    config = AppConfig()
    config.capture.target_fps = 240
    config.dry_run = True
    base = CS2Env(
        config,
        observation_provider=_StubProvider(),
        controller=Controller(config.controller, dry_run=True),
    )
    return FlatActionWrapper(base)


def test_flat_action_space_is_multidiscrete() -> None:
    env = _make_env()
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [
        len(Movement),
        DEFAULT_MOUSE_BINS,
        DEFAULT_MOUSE_BINS,
        2,
        2,
        2,
        2,
        len(WeaponSlot),
    ]
    env.close()


def test_flat_action_round_trip_centre_bin_is_zero() -> None:
    env = _make_env()
    centre = (DEFAULT_MOUSE_BINS - 1) // 2
    flat = np.array([Movement.FORWARD, centre, centre, 1, 0, 1, 0, WeaponSlot.PRIMARY])
    decoded = env.action(flat)
    assert decoded["movement"] == int(Movement.FORWARD)
    assert decoded["weapon"] == int(WeaponSlot.PRIMARY)
    assert decoded["attack"] == 1 and decoded["crouch"] == 1
    assert decoded["jump"] == 0 and decoded["reload"] == 0
    assert decoded["mouse"].dtype == np.float32
    np.testing.assert_allclose(decoded["mouse"], np.zeros(2, dtype=np.float32))
    env.close()


def test_flat_action_extreme_bins_map_to_unit_range() -> None:
    env = _make_env()
    flat_max = np.array([0, DEFAULT_MOUSE_BINS - 1, 0, 0, 0, 0, 0, 0])
    flat_min = np.array([0, 0, DEFAULT_MOUSE_BINS - 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(env.action(flat_max)["mouse"], np.array([1.0, -1.0]))
    np.testing.assert_allclose(env.action(flat_min)["mouse"], np.array([-1.0, 1.0]))
    env.close()


def test_flat_action_encode_inverse_of_action() -> None:
    env = _make_env()
    centre = (DEFAULT_MOUSE_BINS - 1) // 2
    flat = np.array([3, centre + 2, centre - 3, 1, 0, 0, 1, 4])
    decoded = env.action(flat)
    re_encoded = env.encode(decoded)
    np.testing.assert_array_equal(re_encoded, flat)
    env.close()


def test_flat_action_step_runs_against_dict_env() -> None:
    env = _make_env()
    obs, _info = env.reset()
    assert "frame" in obs and "scalars" in obs
    sample = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(sample)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert "raw" in info
    env.close()


def test_flat_action_rejects_wrong_shape() -> None:
    env = _make_env()
    with pytest.raises(ValueError):
        env.action(np.zeros(7, dtype=np.int64))
    env.close()


def test_flat_action_rejects_invalid_mouse_bins() -> None:
    config = AppConfig()
    base = CS2Env(
        config,
        observation_provider=_StubProvider(),
        controller=Controller(config.controller, dry_run=True),
    )
    with pytest.raises(ValueError):
        FlatActionWrapper(base, mouse_bins=20)  # even
    with pytest.raises(ValueError):
        FlatActionWrapper(base, mouse_bins=1)  # too small
    base.close()
