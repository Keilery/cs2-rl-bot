"""Tests for GSI payload parsing."""

from __future__ import annotations

from cs2_rl_bot.observation.state import (
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)


def test_player_state_from_gsi_minimal() -> None:
    payload = {
        "player": {
            "team": "T",
            "state": {
                "health": 87,
                "armor": 50,
                "helmet": True,
                "money": 4500,
                "round_kills": 2,
                "round_totaldmg": 134,
                "flashed": 102,
                "defusekit": False,
            },
            "match_stats": {"deaths": 1},
            "weapons": {
                "weapon_0": {"name": "weapon_knife", "state": "holstered"},
                "weapon_1": {"name": "weapon_ak47", "state": "active"},
            },
        }
    }
    p = PlayerState.from_gsi(payload)
    assert p.health == 87
    assert p.armor == 50
    assert p.helmet is True
    assert p.money == 4500
    assert p.team is Team.T
    assert p.round_kills == 2
    assert p.round_deaths == 1
    assert p.round_damage == 134
    assert p.weapon_active == "weapon_ak47"
    assert 0.39 < p.flashed < 0.41
    assert p.has_bomb is False


def test_player_state_handles_empty_payload() -> None:
    p = PlayerState.from_gsi({})
    assert p.health == 0
    assert p.team is Team.UNKNOWN
    assert p.weapon_active == ""


def test_round_state_from_gsi() -> None:
    payload = {
        "round": {"phase": "live", "bomb": "planted"},
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 3},
            "team_t": {"score": 2},
        },
    }
    r = RoundState.from_gsi(payload)
    assert r.phase is RoundPhase.LIVE
    assert r.bomb_planted is True
    assert r.bomb_defused is False
    assert r.map_name == "de_dust2"
    assert r.round_number == 5
    assert r.score_ct == 3
    assert r.score_t == 2


def test_round_state_handles_unknown_phase() -> None:
    r = RoundState.from_gsi({"round": {"phase": "weird"}, "map": {}})
    assert r.phase is RoundPhase.UNKNOWN
