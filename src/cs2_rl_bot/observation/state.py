"""Typed dataclasses describing the game state observed by the agent.

The state is split into three orthogonal sources:

* :class:`PlayerState` — derived from the GSI ``player`` block.
* :class:`RoundState`  — derived from the GSI ``round`` and ``map`` blocks.
* :class:`Frame`       — the current screen capture (numpy array).

These are combined in :class:`Observation`, which is what the Gymnasium
environment converts into a tensor for the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class RoundPhase(StrEnum):
    UNKNOWN = "unknown"
    FREEZETIME = "freezetime"
    LIVE = "live"
    OVER = "over"
    WARMUP = "warmup"


class Team(StrEnum):
    UNKNOWN = "unknown"
    T = "T"
    CT = "CT"


@dataclass(slots=True)
class PlayerState:
    health: int = 100
    armor: int = 0
    helmet: bool = False
    money: int = 0
    team: Team = Team.UNKNOWN
    round_kills: int = 0
    round_deaths: int = 0
    round_damage: int = 0
    weapon_active: str = ""
    flashed: float = 0.0  # 0.0-1.0 fraction of view obscured
    has_defuser: bool = False
    has_bomb: bool = False

    @classmethod
    def from_gsi(cls, payload: dict[str, Any]) -> PlayerState:
        player = payload.get("player") or {}
        state = player.get("state") or {}
        match_stats = player.get("match_stats") or {}
        weapons = player.get("weapons") or {}

        active_weapon = ""
        for w in weapons.values():
            if isinstance(w, dict) and w.get("state") == "active":
                active_weapon = str(w.get("name", ""))
                break

        team_raw = str(player.get("team", "")).upper()
        try:
            team = Team(team_raw) if team_raw in {"T", "CT"} else Team.UNKNOWN
        except ValueError:
            team = Team.UNKNOWN

        return cls(
            health=int(state.get("health", 0)),
            armor=int(state.get("armor", 0)),
            helmet=bool(state.get("helmet", False)),
            money=int(state.get("money", 0)),
            team=team,
            round_kills=int(state.get("round_kills", 0)),
            round_deaths=int(match_stats.get("deaths", 0)),
            round_damage=int(state.get("round_totaldmg", 0)),
            weapon_active=active_weapon,
            flashed=float(state.get("flashed", 0)) / 255.0,
            has_defuser=bool(state.get("defusekit", False)),
            has_bomb="weapon_c4"
            in {w.get("name", "") for w in weapons.values() if isinstance(w, dict)},
        )


@dataclass(slots=True)
class RoundState:
    phase: RoundPhase = RoundPhase.UNKNOWN
    bomb_planted: bool = False
    bomb_defused: bool = False
    win_team: Team = Team.UNKNOWN
    map_name: str = ""
    round_number: int = 0
    score_ct: int = 0
    score_t: int = 0

    @classmethod
    def from_gsi(cls, payload: dict[str, Any]) -> RoundState:
        round_data = payload.get("round") or {}
        map_data = payload.get("map") or {}

        phase_raw = str(round_data.get("phase", "")).lower()
        try:
            phase = RoundPhase(phase_raw) if phase_raw else RoundPhase.UNKNOWN
        except ValueError:
            phase = RoundPhase.UNKNOWN

        win_raw = str(round_data.get("win_team", "")).upper()
        try:
            win_team = Team(win_raw) if win_raw in {"T", "CT"} else Team.UNKNOWN
        except ValueError:
            win_team = Team.UNKNOWN

        bomb = str(round_data.get("bomb", "")).lower()
        team_ct = map_data.get("team_ct") or {}
        team_t = map_data.get("team_t") or {}

        return cls(
            phase=phase,
            bomb_planted=bomb in {"planted", "defused", "exploded"},
            bomb_defused=bomb == "defused",
            win_team=win_team,
            map_name=str(map_data.get("name", "")),
            round_number=int(map_data.get("round", 0)),
            score_ct=int(team_ct.get("score", 0)),
            score_t=int(team_t.get("score", 0)),
        )


@dataclass(slots=True)
class Frame:
    """A single captured frame, optionally with vision detections attached."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    timestamp: float
    detections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape  # type: ignore[return-value]


@dataclass(slots=True)
class Observation:
    """Combined observation passed to the agent."""

    player: PlayerState
    round: RoundState
    frame: Frame | None = None
    raw_gsi: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Whether this observation should end the episode (round)."""
        return self.round.phase in {RoundPhase.OVER}
