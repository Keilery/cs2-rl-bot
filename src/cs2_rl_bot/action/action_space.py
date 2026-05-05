"""Discrete + continuous action space definition.

The agent emits a tuple ``(movement, mouse_dx, mouse_dy, attack, jump, crouch,
reload, weapon_slot)`` per step. We use a Gymnasium ``Dict`` so the policy
network sees clearly separated heads instead of one giant MultiDiscrete.

Movement is discrete (8 directions + stand), mouse is continuous (2 floats in
[-1, 1] scaled by ``max_mouse_delta``), and the rest are binary toggles.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from gymnasium import spaces


class Movement(IntEnum):
    STAND = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6
    BACKWARD_LEFT = 7
    BACKWARD_RIGHT = 8


class WeaponSlot(IntEnum):
    NONE = 0
    PRIMARY = 1
    SECONDARY = 2
    KNIFE = 3
    GRENADE_HE = 4
    GRENADE_FLASH = 5
    GRENADE_SMOKE = 6
    BOMB = 7


def build_action_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "movement": spaces.Discrete(len(Movement)),
            "mouse": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "attack": spaces.Discrete(2),
            "jump": spaces.Discrete(2),
            "crouch": spaces.Discrete(2),
            "reload": spaces.Discrete(2),
            "weapon": spaces.Discrete(len(WeaponSlot)),
        }
    )


# Direction vectors used by the controller to translate Movement -> WASD keys.
MOVEMENT_KEYS: dict[Movement, tuple[str, ...]] = {
    Movement.STAND: (),
    Movement.FORWARD: ("w",),
    Movement.BACKWARD: ("s",),
    Movement.LEFT: ("a",),
    Movement.RIGHT: ("d",),
    Movement.FORWARD_LEFT: ("w", "a"),
    Movement.FORWARD_RIGHT: ("w", "d"),
    Movement.BACKWARD_LEFT: ("s", "a"),
    Movement.BACKWARD_RIGHT: ("s", "d"),
}


# Default key bindings — match the in-game defaults. Override via config if the
# player has a custom layout.
WEAPON_SLOT_KEYS: dict[WeaponSlot, str] = {
    WeaponSlot.NONE: "",
    WeaponSlot.PRIMARY: "1",
    WeaponSlot.SECONDARY: "2",
    WeaponSlot.KNIFE: "3",
    WeaponSlot.GRENADE_HE: "6",
    WeaponSlot.GRENADE_FLASH: "7",
    WeaponSlot.GRENADE_SMOKE: "8",
    WeaponSlot.BOMB: "5",
}
