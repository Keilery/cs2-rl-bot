"""``ActionWrapper`` that flattens our :class:`Dict` action space.

Stable-Baselines3's PPO does not support ``Dict`` action spaces — it only
accepts ``Box``, ``Discrete``, ``MultiDiscrete``, or ``MultiBinary``. We
expose a ``Dict`` to keep the policy heads readable and to match what the
controller / behaviour-cloning code expects, but for SB3 training we wrap
the env so PPO sees a flat ``MultiDiscrete`` instead.

Layout (8 dims):

* dim 0 — movement       (9 categories, see :class:`Movement`)
* dim 1 — mouse_x bin    (``mouse_bins`` categories, default 21 → ±10
                          centred at bin 10)
* dim 2 — mouse_y bin    (same)
* dim 3 — attack         (2)
* dim 4 — jump           (2)
* dim 5 — crouch         (2)
* dim 6 — reload         (2)
* dim 7 — weapon         (8 categories, see :class:`WeaponSlot`)

The mouse axes are quantised because PPO learns categorical heads more
reliably than tiny-magnitude continuous ones in our setting; 21 bins per
axis gives the agent ±1.0 in 0.1 increments which matches the
controller's per-step mouse delta resolution.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cs2_rl_bot.action.action_space import Movement, WeaponSlot

DEFAULT_MOUSE_BINS = 21


class FlatActionWrapper(gym.ActionWrapper):
    """Convert ``MultiDiscrete[9, B, B, 2, 2, 2, 2, 8]`` → original ``Dict``.

    Parameters
    ----------
    env:
        The wrapped environment. Must expose the ``Dict`` action space
        produced by :func:`cs2_rl_bot.action.action_space.build_action_space`.
    mouse_bins:
        Number of discrete bins per mouse axis. Must be odd so that the
        centre bin maps exactly to 0.0. Defaults to 21.
    """

    def __init__(self, env: gym.Env, *, mouse_bins: int = DEFAULT_MOUSE_BINS) -> None:
        super().__init__(env)
        if mouse_bins < 3 or mouse_bins % 2 == 0:
            raise ValueError(f"mouse_bins must be an odd integer >= 3, got {mouse_bins!r}")
        self._mouse_bins = mouse_bins
        self._mouse_centre = (mouse_bins - 1) // 2
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    len(Movement),
                    mouse_bins,
                    mouse_bins,
                    2,
                    2,
                    2,
                    2,
                    len(WeaponSlot),
                ],
                dtype=np.int64,
            )
        )

    def action(self, flat_action: Any) -> dict[str, np.ndarray | int]:
        """Translate the flat action emitted by SB3 back into the env's Dict."""
        a = np.asarray(flat_action, dtype=np.int64).reshape(-1)
        if a.shape != (8,):
            raise ValueError(
                f"FlatActionWrapper expected an 8-dim flat action, got shape {a.shape}"
            )
        movement = int(np.clip(a[0], 0, len(Movement) - 1))
        mouse_x = (a[1] - self._mouse_centre) / self._mouse_centre
        mouse_y = (a[2] - self._mouse_centre) / self._mouse_centre
        weapon = int(np.clip(a[7], 0, len(WeaponSlot) - 1))
        return {
            "movement": movement,
            "mouse": np.array([mouse_x, mouse_y], dtype=np.float32),
            "attack": int(a[3] != 0),
            "jump": int(a[4] != 0),
            "crouch": int(a[5] != 0),
            "reload": int(a[6] != 0),
            "weapon": weapon,
        }

    def encode(self, action: dict[str, Any]) -> np.ndarray:
        """Inverse of :meth:`action` — pack a Dict action into the flat form.

        Useful when feeding behaviour-cloning samples to a SB3 policy that
        was trained against the wrapped env.
        """
        mouse = np.asarray(action["mouse"], dtype=np.float32).reshape(2)
        mouse = np.clip(mouse, -1.0, 1.0)
        mouse_x_bin = round(float(mouse[0]) * self._mouse_centre) + self._mouse_centre
        mouse_y_bin = round(float(mouse[1]) * self._mouse_centre) + self._mouse_centre
        return np.array(
            [
                int(action["movement"]),
                int(np.clip(mouse_x_bin, 0, self._mouse_bins - 1)),
                int(np.clip(mouse_y_bin, 0, self._mouse_bins - 1)),
                int(bool(int(action["attack"]))),
                int(bool(int(action["jump"]))),
                int(bool(int(action["crouch"]))),
                int(bool(int(action["reload"]))),
                int(action["weapon"]),
            ],
            dtype=np.int64,
        )
