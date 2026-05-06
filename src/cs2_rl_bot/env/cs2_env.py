"""Gymnasium-compatible environment that wraps the CS2 client.

The environment is *not* a simulator — it observes the running game via GSI +
screen capture and emits inputs through :class:`Controller`. As a consequence,
``step()`` runs in real wall-clock time and is not vectorisable.

Usage from training/inference loops::

    env = CS2Env(config)
    obs, info = env.reset()
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

For unit tests, pass a fake ``observation_provider`` so the env can step
without a running game.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cs2_rl_bot.action.action_space import build_action_space
from cs2_rl_bot.action.controller import Controller
from cs2_rl_bot.env.rewards import RewardCalculator
from cs2_rl_bot.observation.gsi_server import GSIState, run_server
from cs2_rl_bot.observation.screen_capture import ScreenCapture
from cs2_rl_bot.observation.state import Frame, Observation, PlayerState, RoundState
from cs2_rl_bot.observation.vision import (
    ENEMY_FEATURE_DIM,
    Detection,
    EnemyDetector,
    encode_enemy_features,
)
from cs2_rl_bot.utils.config import AppConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


class ObservationProvider(Protocol):
    def latest(self) -> Observation: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class _LiveObservationProvider:
    """Combines GSI + screen capture into a single observation source."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._gsi_state = GSIState()
        self._capture = ScreenCapture(config.capture)
        self._detector = EnemyDetector(config.vision)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._capture.start()

        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._server_task = loop.create_task(run_server(self._config.gsi, self._gsi_state))
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name="cs2-rl-bot-gsi", daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)

    def stop(self) -> None:
        self._capture.stop()
        if self._loop is not None and self._server_task is not None:
            self._loop.call_soon_threadsafe(self._server_task.cancel)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
            self._loop = None
            self._server_task = None

    def latest(self) -> Observation:
        # GSI: synchronous read of the latest cached payload.
        if self._loop is not None:
            future = asyncio.run_coroutine_threadsafe(self._gsi_state.get(), self._loop)
            payload = future.result(timeout=self._config.gsi.request_timeout_s)
        else:
            payload = {}

        frame = self._capture.latest(timeout=0.05)
        if frame is not None and self._config.vision.enabled:
            # Run YOLO on the full-res frame when available — the 84x84 policy
            # frame is too small for reliable detection.
            detections = self._detector.detect(frame.detection_image)
            frame.detections = [d.as_dict() for d in detections]

        return Observation(
            player=PlayerState.from_gsi(payload),
            round=RoundState.from_gsi(payload),
            frame=frame,
            raw_gsi=payload,
        )


def _build_observation_space(config: AppConfig) -> spaces.Dict:
    h, w = config.capture.resize_to
    channels = 1 if config.capture.grayscale else 3
    enemy_dim = config.vision.max_enemy_slots * ENEMY_FEATURE_DIM
    return spaces.Dict(
        {
            "frame": spaces.Box(low=0, high=255, shape=(h, w, channels), dtype=np.uint8),
            "scalars": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(_SCALAR_DIM,),
                dtype=np.float32,
            ),
            "enemies": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(enemy_dim,),
                dtype=np.float32,
            ),
        }
    )


# health / armor / money / round_kills / round_damage / flashed / has_bomb /
# has_defuser / phase_one_hot(4) / team_one_hot(2)
_SCALAR_DIM = 14


def encode_scalars(obs: Observation) -> np.ndarray:
    p = obs.player
    r = obs.round
    phase_index = {"freezetime": 0, "live": 1, "over": 2, "warmup": 3}.get(r.phase.value, 3)
    phase_oh = np.zeros(4, dtype=np.float32)
    phase_oh[phase_index] = 1.0
    team_oh = np.zeros(2, dtype=np.float32)
    if p.team.value == "T":
        team_oh[0] = 1.0
    elif p.team.value == "CT":
        team_oh[1] = 1.0
    arr = np.array(
        [
            p.health / 100.0,
            p.armor / 100.0,
            min(p.money / 16000.0, 1.0),
            min(p.round_kills / 5.0, 1.0),
            min(p.round_damage / 500.0, 1.0),
            p.flashed,
            float(p.has_bomb),
            float(p.has_defuser),
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr, phase_oh, team_oh])


class CS2Env(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Gymnasium environment wrapping a live CS2 client (or a stub provider)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        config: AppConfig,
        *,
        observation_provider: ObservationProvider | None = None,
        controller: Controller | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._provider = observation_provider or _LiveObservationProvider(config)
        self._controller = controller or Controller(config.controller, dry_run=config.dry_run)
        self._reward = RewardCalculator(config.rewards)
        self._step_index = 0
        self._step_dt = 1.0 / max(config.capture.target_fps, 1)
        self._started = False

        self.action_space = build_action_space()
        self.observation_space = _build_observation_space(config)

    # ----- Gym API ---------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if not self._started:
            self._provider.start()
            self._started = True
        self._reward.reset()
        self._controller.release_all()
        self._step_index = 0
        obs = self._provider.latest()
        return self._encode(obs), {"raw": obs}

    def step(
        self,
        action: Mapping[str, np.ndarray | int | float],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        loop_start = time.perf_counter()

        self._controller.apply(action)
        observation = self._provider.latest()
        breakdown = self._reward.compute(
            observation, kill_switch_engaged=self._controller.kill_switch_engaged
        )

        self._step_index += 1
        terminated = observation.is_terminal()
        truncated = False

        elapsed = time.perf_counter() - loop_start
        if elapsed < self._step_dt:
            time.sleep(self._step_dt - elapsed)

        info = {
            "raw": observation,
            "reward_breakdown": breakdown,
            "step_index": self._step_index,
            "kill_switch": self._controller.kill_switch_engaged,
        }
        return self._encode(observation), float(breakdown.total), terminated, truncated, info

    def close(self) -> None:
        self._controller.release_all()
        self._provider.stop()
        self._started = False

    # ----- helpers ---------------------------------------------------------

    def _encode(self, obs: Observation) -> dict[str, np.ndarray]:
        if obs.frame is not None:
            frame_array = obs.frame.image
        else:
            h, w = self._config.capture.resize_to
            channels = 1 if self._config.capture.grayscale else 3
            frame_array = np.zeros((h, w, channels), dtype=np.uint8)
        if frame_array.ndim == 2:
            frame_array = frame_array[..., None]
        return {
            "frame": frame_array,
            "scalars": encode_scalars(obs),
            "enemies": self._encode_enemies(obs),
        }

    def _encode_enemies(self, obs: Observation) -> np.ndarray:
        max_slots = self._config.vision.max_enemy_slots
        if obs.frame is None:
            return np.zeros(max_slots * ENEMY_FEATURE_DIM, dtype=np.float32)
        # Detections may be raw dicts (from a stub provider) or Detection
        # instances (from the live detector). Normalise to Detection.
        detections: list[Detection] = []
        for d in obs.frame.detections:
            if isinstance(d, Detection):
                detections.append(d)
            elif isinstance(d, dict):
                bbox = tuple(int(v) for v in d.get("bbox", (0, 0, 0, 0)))
                if len(bbox) != 4:
                    continue
                detections.append(
                    Detection(
                        cls=str(d.get("cls", "")),
                        confidence=float(d.get("confidence", 0.0)),
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        team=d.get("team"),
                        is_head=bool(d.get("is_head", False)),
                    )
                )
        # The detection image is what the bboxes are referenced against —
        # full-res when available, otherwise the policy-sized frame.
        det_image = obs.frame.detection_image
        h = det_image.shape[0] if det_image.ndim >= 2 else 0
        w = det_image.shape[1] if det_image.ndim >= 2 else 0
        return encode_enemy_features(
            detections,
            image_shape=(h, w),
            our_team=obs.player.team,
            max_slots=max_slots,
        )


def _ensure_unused() -> None:
    """Compile-time check that ``Frame`` is referenced from the type checker."""
    _ = Frame
