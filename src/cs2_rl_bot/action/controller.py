"""OS-level mouse/keyboard controller with a hard kill-switch.

Three backends are supported:

* ``pydirectinput`` — Windows only, uses SendInput so CS2 sees the events as
  raw input. This is the only backend that actually moves the in-game crosshair.
* ``pynput`` — cross-platform, but CS2 with raw input enabled will *ignore*
  these events. Useful for X11 desktops or when you disable raw input
  (``-no_raw_input``) for offline experimentation.
* ``noop`` — a non-destructive backend that only logs intended actions. This
  is the default and what is used in CI and unit tests.

The kill-switch (default F9) drops the next emitted action and releases all
held keys. This is a manual safety so the operator can immediately regain
control.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from cs2_rl_bot.action.action_space import (
    MOVEMENT_KEYS,
    WEAPON_SLOT_KEYS,
    Movement,
    WeaponSlot,
)
from cs2_rl_bot.utils.config import ControllerConfig
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping


class _Backend(ABC):
    @abstractmethod
    def move_mouse(self, dx: int, dy: int) -> None: ...

    @abstractmethod
    def key_down(self, key: str) -> None: ...

    @abstractmethod
    def key_up(self, key: str) -> None: ...

    @abstractmethod
    def mouse_button(self, button: str, *, down: bool) -> None: ...


class _NoopBackend(_Backend):
    def move_mouse(self, dx: int, dy: int) -> None:
        logger.debug("[noop] move_mouse dx={} dy={}", dx, dy)

    def key_down(self, key: str) -> None:
        logger.debug("[noop] key_down {}", key)

    def key_up(self, key: str) -> None:
        logger.debug("[noop] key_up {}", key)

    def mouse_button(self, button: str, *, down: bool) -> None:
        logger.debug("[noop] mouse_button {}={}", button, "down" if down else "up")


class _PyDirectInputBackend(_Backend):  # pragma: no cover — Windows-only
    def __init__(self) -> None:
        import pydirectinput

        # Disable the failsafe (mouse-to-corner abort) — we have our own kill switch.
        pydirectinput.FAILSAFE = False
        self._pdi = pydirectinput

    def move_mouse(self, dx: int, dy: int) -> None:
        self._pdi.moveRel(dx, dy, relative=True)

    def key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pdi.keyUp(key)

    def mouse_button(self, button: str, *, down: bool) -> None:
        if down:
            self._pdi.mouseDown(button=button)
        else:
            self._pdi.mouseUp(button=button)


class _PynputBackend(_Backend):  # pragma: no cover — exercised manually
    def __init__(self) -> None:
        from pynput.keyboard import Controller as KbCtl
        from pynput.keyboard import KeyCode
        from pynput.mouse import Button
        from pynput.mouse import Controller as MouseCtl

        self._kb = KbCtl()
        self._mouse = MouseCtl()
        self._key_code = KeyCode
        self._button = Button

    def move_mouse(self, dx: int, dy: int) -> None:
        self._mouse.move(dx, dy)

    def key_down(self, key: str) -> None:
        self._kb.press(self._key_code.from_char(key))

    def key_up(self, key: str) -> None:
        self._kb.release(self._key_code.from_char(key))

    def mouse_button(self, button: str, *, down: bool) -> None:
        btn = self._button.left if button == "left" else self._button.right
        if down:
            self._mouse.press(btn)
        else:
            self._mouse.release(btn)


def _make_backend(name: str) -> _Backend:
    if name == "pydirectinput":
        try:
            return _PyDirectInputBackend()
        except (ImportError, OSError) as exc:
            logger.warning("pydirectinput unavailable, falling back to noop: {}", exc)
            return _NoopBackend()
    if name == "pynput":
        try:
            return _PynputBackend()
        except (ImportError, OSError) as exc:
            logger.warning("pynput unavailable, falling back to noop: {}", exc)
            return _NoopBackend()
    return _NoopBackend()


class Controller:
    """High-level controller that translates :class:`Action` dicts into events."""

    def __init__(self, config: ControllerConfig, *, dry_run: bool = True) -> None:
        self._config = config
        self._dry_run = dry_run
        self._backend = _NoopBackend() if dry_run else _make_backend(config.backend)
        self._held_keys: set[str] = set()
        self._mouse_down = False
        self._lock = threading.Lock()
        self._kill_switch = threading.Event()

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch.is_set()

    def engage_kill_switch(self) -> None:
        """Release all input and refuse further actions until cleared."""
        self._kill_switch.set()
        self.release_all()
        logger.warning("kill switch engaged — controller disabled")

    def clear_kill_switch(self) -> None:
        self._kill_switch.clear()
        logger.info("kill switch cleared")

    def release_all(self) -> None:
        with self._lock:
            for key in list(self._held_keys):
                self._backend.key_up(key)
            self._held_keys.clear()
            if self._mouse_down:
                self._backend.mouse_button("left", down=False)
                self._mouse_down = False

    def apply(self, action: Mapping[str, np.ndarray | int | float]) -> None:
        """Translate a sampled action dict into OS-level events."""
        if self._kill_switch.is_set():
            return

        with self._lock:
            self._apply_movement(int(action["movement"]))
            self._apply_mouse(np.asarray(action["mouse"], dtype=np.float32))
            self._apply_attack(bool(int(action["attack"])))
            self._apply_modifier("space", bool(int(action["jump"])))
            self._apply_modifier("ctrl", bool(int(action["crouch"])))
            self._apply_modifier("r", bool(int(action["reload"])))
            self._apply_weapon(int(action["weapon"]))

    def _apply_movement(self, value: int) -> None:
        try:
            move = Movement(value)
        except ValueError:
            move = Movement.STAND
        desired = set(MOVEMENT_KEYS[move])
        movement_keys = {"w", "a", "s", "d"}
        # Release movement keys that should no longer be held.
        for key in self._held_keys & movement_keys - desired:
            self._backend.key_up(key)
            self._held_keys.discard(key)
        # Press movement keys that aren't held yet.
        for key in desired - self._held_keys:
            self._backend.key_down(key)
            self._held_keys.add(key)

    def _apply_mouse(self, vec: np.ndarray) -> None:
        scale = self._config.max_mouse_delta * self._config.mouse_sensitivity
        dx = int(np.clip(vec[0], -1.0, 1.0) * scale)
        dy = int(np.clip(vec[1], -1.0, 1.0) * scale)
        if dx == 0 and dy == 0:
            return
        self._backend.move_mouse(dx, dy)

    def _apply_attack(self, fire: bool) -> None:
        if fire and not self._mouse_down:
            self._backend.mouse_button("left", down=True)
            self._mouse_down = True
        elif not fire and self._mouse_down:
            self._backend.mouse_button("left", down=False)
            self._mouse_down = False

    def _apply_modifier(self, key: str, pressed: bool) -> None:
        if pressed and key not in self._held_keys:
            self._backend.key_down(key)
            self._held_keys.add(key)
        elif not pressed and key in self._held_keys:
            self._backend.key_up(key)
            self._held_keys.discard(key)

    def _apply_weapon(self, value: int) -> None:
        try:
            slot = WeaponSlot(value)
        except ValueError:
            slot = WeaponSlot.NONE
        if slot is WeaponSlot.NONE:
            return
        key = WEAPON_SLOT_KEYS[slot]
        # Tap-style: down + up on the same step.
        self._backend.key_down(key)
        self._backend.key_up(key)
