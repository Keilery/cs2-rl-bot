"""CS2 demo parser for offline imitation-learning datasets.

Demos in CS2 use the same format as CS:GO. We use the `demoparser2` Python
package (Rust-backed, fast). It is an *optional* dependency declared in
``pyproject.toml`` under the ``demos`` extra — install it with
``pip install -e '.[demos]'``.

The parser produces an iterable of :class:`DemoTransition` objects, each
containing what the agent would have observed (player HP / armor / weapon /
team / round phase) and what it would have done (movement vector + buttons
mask + view angle delta). Frame-level pixel data is *not* recovered from
demos — the BC pretraining therefore uses only the scalar observation head,
and trains a separate "scalar policy" that the PPO model can then warm-start
from. See ``src/cs2_rl_bot/training/imitation.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cs2_rl_bot.action.action_space import Movement, WeaponSlot
from cs2_rl_bot.observation.state import (
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


# Buttons bitmask values from public/in_buttons.h (CS:GO/CS2 share these).
IN_ATTACK = 1 << 0
IN_JUMP = 1 << 1
IN_DUCK = 1 << 2
IN_FORWARD = 1 << 3
IN_BACK = 1 << 4
IN_RELOAD = 1 << 13
IN_MOVELEFT = 1 << 9
IN_MOVERIGHT = 1 << 10


@dataclass(slots=True)
class DemoTransition:
    """One tick of imitation training data extracted from a demo."""

    observation: Observation
    action: dict[str, Any]
    tick: int = 0
    weight: float = 1.0  # used to upweight kill / clutch ticks

    def encode_action(self) -> dict[str, np.ndarray | int]:
        """Convert ``action`` dict into the shape expected by the policy."""
        return {
            "movement": int(self.action.get("movement", Movement.STAND)),
            "mouse": np.asarray(self.action.get("mouse", [0.0, 0.0]), dtype=np.float32),
            "attack": int(self.action.get("attack", 0)),
            "jump": int(self.action.get("jump", 0)),
            "crouch": int(self.action.get("crouch", 0)),
            "reload": int(self.action.get("reload", 0)),
            "weapon": int(self.action.get("weapon", WeaponSlot.NONE)),
        }


@dataclass(slots=True)
class DemoFilter:
    """Criteria for keeping or discarding ticks during parsing."""

    target_steam_id: str | None = None
    drop_freezetime: bool = True
    drop_warmup: bool = True
    drop_dead: bool = True
    drop_zero_movement: bool = False
    max_ticks: int | None = None
    fields_required: tuple[str, ...] = field(
        default_factory=lambda: (
            "tick",
            "health",
            "armor_value",
            "team_num",
            "buttons",
            "FORWARD",
            "LEFT",
            "RIGHT",
            "BACK",
            "is_alive",
            "round_phase",
            "kills_total",
            "deaths_total",
            "damage_total",
            "X",
            "Y",
            "Z",
            "pitch",
            "yaw",
            "weapon_name",
            "steamid",
        )
    )


class DemoParser:
    """High-level wrapper around demoparser2.

    Use as a context manager or call :meth:`iter_transitions` directly. The
    parser is constructed lazily so missing optional deps don't blow up
    import-time.
    """

    def __init__(self, demo_path: str | Path, *, demo_filter: DemoFilter | None = None) -> None:
        self._path = Path(demo_path)
        self._filter = demo_filter or DemoFilter()
        if not self._path.exists():
            raise FileNotFoundError(self._path)

    def iter_transitions(self) -> Iterator[DemoTransition]:
        try:
            import demoparser2 as _dp
        except ImportError as exc:  # pragma: no cover — see optional extra
            raise RuntimeError(
                "demoparser2 is not installed. Install it via `pip install -e '.[demos]'`."
            ) from exc

        logger.info("parsing demo {}", self._path.name)
        parser = _dp.DemoParser(str(self._path))
        df = parser.parse_ticks(list(self._filter.fields_required))
        return self._iter_dataframe(df)

    def _iter_dataframe(self, df: Any) -> Iterator[DemoTransition]:
        # demoparser2 returns a polars / pandas DataFrame depending on version.
        if hasattr(df, "to_pandas"):  # polars
            df = df.to_pandas()

        steam_id = self._filter.target_steam_id
        if steam_id is not None and "steamid" in df:
            df = df[df["steamid"].astype(str) == str(steam_id)]

        if self._filter.drop_freezetime and "round_phase" in df:
            df = df[df["round_phase"] != "freezetime"]
        if self._filter.drop_warmup and "round_phase" in df:
            df = df[df["round_phase"] != "warmup"]
        if self._filter.drop_dead and "is_alive" in df:
            df = df[df["is_alive"]]
        if self._filter.max_ticks is not None:
            df = df.head(self._filter.max_ticks)

        prev_yaw: float | None = None
        prev_pitch: float | None = None
        for row in df.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row.__dict__)
            yield from self._row_to_transition(row_dict, prev_yaw, prev_pitch)
            prev_yaw = float(row_dict.get("yaw", prev_yaw or 0.0))
            prev_pitch = float(row_dict.get("pitch", prev_pitch or 0.0))

    def _row_to_transition(
        self,
        row: dict[str, Any],
        prev_yaw: float | None,
        prev_pitch: float | None,
    ) -> Iterator[DemoTransition]:
        try:
            tx = transition_from_row(row, prev_yaw=prev_yaw, prev_pitch=prev_pitch)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("skip bad demo row: {}", exc)
            return
        if self._filter.drop_zero_movement and tx.action["movement"] == int(Movement.STAND):
            return
        yield tx


def transition_from_row(
    row: dict[str, Any],
    *,
    prev_yaw: float | None,
    prev_pitch: float | None,
) -> DemoTransition:
    """Pure-function row → transition translation, exposed for tests."""

    buttons = int(row.get("buttons", 0) or 0)
    forward = bool(buttons & IN_FORWARD) or bool(row.get("FORWARD", False))
    back = bool(buttons & IN_BACK) or bool(row.get("BACK", False))
    left = bool(buttons & IN_MOVELEFT) or bool(row.get("LEFT", False))
    right = bool(buttons & IN_MOVERIGHT) or bool(row.get("RIGHT", False))

    movement = Movement.STAND
    if forward and left:
        movement = Movement.FORWARD_LEFT
    elif forward and right:
        movement = Movement.FORWARD_RIGHT
    elif back and left:
        movement = Movement.BACKWARD_LEFT
    elif back and right:
        movement = Movement.BACKWARD_RIGHT
    elif forward:
        movement = Movement.FORWARD
    elif back:
        movement = Movement.BACKWARD
    elif left:
        movement = Movement.LEFT
    elif right:
        movement = Movement.RIGHT

    yaw = float(row.get("yaw", 0.0) or 0.0)
    pitch = float(row.get("pitch", 0.0) or 0.0)
    dyaw = _angle_delta(yaw, prev_yaw)
    dpitch = _angle_delta(pitch, prev_pitch)
    # Normalise into [-1, 1] using a 30-degree max-step heuristic.
    mouse = np.array(
        [_clip(dyaw / 30.0, -1.0, 1.0), _clip(dpitch / 30.0, -1.0, 1.0)],
        dtype=np.float32,
    )

    weapon_name = str(row.get("weapon_name", "")).lower()
    weapon = _classify_weapon(weapon_name)

    team_num = int(row.get("team_num", 0) or 0)
    team = Team.T if team_num == 2 else Team.CT if team_num == 3 else Team.UNKNOWN
    phase_raw = str(row.get("round_phase", "live")).lower()
    phase = RoundPhase(phase_raw) if phase_raw in {p.value for p in RoundPhase} else RoundPhase.LIVE

    obs = Observation(
        player=PlayerState(
            health=int(row.get("health", 0) or 0),
            armor=int(row.get("armor_value", 0) or 0),
            money=int(row.get("money", 0) or 0),
            team=team,
            round_kills=int(row.get("kills_total", 0) or 0),
            round_deaths=int(row.get("deaths_total", 0) or 0),
            round_damage=int(row.get("damage_total", 0) or 0),
            weapon_active=weapon_name,
        ),
        round=RoundState(phase=phase),
        frame=None,
        raw_gsi={},
    )

    action = {
        "movement": int(movement),
        "mouse": mouse,
        "attack": int(bool(buttons & IN_ATTACK)),
        "jump": int(bool(buttons & IN_JUMP)),
        "crouch": int(bool(buttons & IN_DUCK)),
        "reload": int(bool(buttons & IN_RELOAD)),
        "weapon": int(weapon),
    }

    return DemoTransition(
        observation=obs,
        action=action,
        tick=int(row.get("tick", 0) or 0),
    )


def parse_demos(
    paths: Iterable[str | Path],
    *,
    demo_filter: DemoFilter | None = None,
) -> Iterator[DemoTransition]:
    for p in paths:
        try:
            yield from DemoParser(p, demo_filter=demo_filter).iter_transitions()
        except FileNotFoundError as exc:
            logger.warning("demo not found, skipping: {}", exc)


def _angle_delta(current: float, previous: float | None) -> float:
    if previous is None:
        return 0.0
    delta = (current - previous + 180.0) % 360.0 - 180.0
    if math.isnan(delta) or math.isinf(delta):
        return 0.0
    return float(delta)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _classify_weapon(name: str) -> WeaponSlot:
    if not name:
        return WeaponSlot.NONE
    if any(s in name for s in ("knife",)):
        return WeaponSlot.KNIFE
    if any(s in name for s in ("c4", "bomb")):
        return WeaponSlot.BOMB
    if "flash" in name:
        return WeaponSlot.GRENADE_FLASH
    if "smoke" in name:
        return WeaponSlot.GRENADE_SMOKE
    if any(s in name for s in ("hegrenade", "_he", "_grenade")):
        return WeaponSlot.GRENADE_HE
    if any(
        s in name
        for s in ("usp", "glock", "deagle", "p250", "tec9", "elite", "p2000", "fiveseven", "cz75")
    ):
        return WeaponSlot.SECONDARY
    return WeaponSlot.PRIMARY
