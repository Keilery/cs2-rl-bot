"""Tests for cs2_rl_bot.training.demo_parser."""

from __future__ import annotations

import math

import pytest

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.observation.state import RoundPhase, Team
from cs2_rl_bot.training.demo_parser import (
    IN_ATTACK,
    IN_DUCK,
    IN_FORWARD,
    IN_JUMP,
    IN_MOVELEFT,
    IN_MOVERIGHT,
    IN_RELOAD,
    DemoFilter,
    DemoParser,
    _angle_delta,
    _classify_weapon,
    transition_from_row,
)


def test_classify_weapon_buckets() -> None:
    assert _classify_weapon("weapon_knife") is WeaponSlot.KNIFE
    assert _classify_weapon("weapon_c4") is WeaponSlot.BOMB
    assert _classify_weapon("weapon_flashbang") is WeaponSlot.GRENADE_FLASH
    assert _classify_weapon("weapon_smokegrenade") is WeaponSlot.GRENADE_SMOKE
    assert _classify_weapon("weapon_hegrenade") is WeaponSlot.GRENADE_HE
    assert _classify_weapon("weapon_usp_silencer") is WeaponSlot.SECONDARY
    assert _classify_weapon("weapon_ak47") is WeaponSlot.PRIMARY
    assert _classify_weapon("") is WeaponSlot.NONE


def test_angle_delta_wraps() -> None:
    assert _angle_delta(10.0, 350.0) == pytest.approx(20.0)
    assert _angle_delta(-170.0, 170.0) == pytest.approx(20.0)
    assert _angle_delta(5.0, None) == 0.0
    assert _angle_delta(math.nan, 0.0) == 0.0


def test_transition_decodes_buttons_and_movement() -> None:
    row = {
        "tick": 42,
        "health": 80,
        "armor_value": 100,
        "team_num": 2,
        "buttons": IN_ATTACK | IN_FORWARD | IN_MOVELEFT | IN_JUMP,
        "is_alive": True,
        "round_phase": "live",
        "kills_total": 1,
        "deaths_total": 0,
        "damage_total": 47,
        "X": 0.0,
        "Y": 0.0,
        "Z": 0.0,
        "pitch": 5.0,
        "yaw": 90.0,
        "weapon_name": "weapon_ak47",
        "steamid": "76561198000000000",
    }
    tx = transition_from_row(row, prev_yaw=80.0, prev_pitch=4.0)
    assert tx.tick == 42
    assert tx.action["movement"] == int(Movement.FORWARD_LEFT)
    assert tx.action["attack"] == 1
    assert tx.action["jump"] == 1
    assert tx.action["crouch"] == 0
    assert tx.action["reload"] == 0
    assert tx.action["weapon"] == int(WeaponSlot.PRIMARY)
    assert tx.observation.player.health == 80
    assert tx.observation.player.team is Team.T
    assert tx.observation.round.phase is RoundPhase.LIVE
    assert tx.action["mouse"][0] == pytest.approx((90.0 - 80.0) / 30.0)


def test_transition_handles_back_right_crouch_reload() -> None:
    row = {
        "buttons": IN_DUCK | IN_RELOAD | IN_MOVERIGHT,
        "weapon_name": "weapon_c4",
        "round_phase": "freezetime",
        "team_num": 3,
    }
    tx = transition_from_row(row, prev_yaw=None, prev_pitch=None)
    assert tx.action["movement"] == int(Movement.RIGHT)
    assert tx.action["crouch"] == 1
    assert tx.action["reload"] == 1
    assert tx.action["weapon"] == int(WeaponSlot.BOMB)
    assert tx.observation.player.team is Team.CT
    assert tx.observation.round.phase is RoundPhase.FREEZETIME


def test_demo_parser_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        DemoParser(tmp_path / "missing.dem")


def test_demo_filter_defaults() -> None:
    flt = DemoFilter()
    assert "buttons" in flt.fields_required
    assert flt.drop_freezetime is True
    assert flt.drop_dead is True
