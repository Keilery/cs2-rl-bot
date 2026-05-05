"""Cross-platform screen capture using `mss`.

The capture loop runs on a background thread because `mss` is blocking and the
agent loop is synchronous. Frames are placed into a single-slot queue (latest
wins) to keep latency minimal — RL with stale frames is worse than RL with
dropped frames.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

import numpy as np

from cs2_rl_bot.observation.state import Frame
from cs2_rl_bot.utils.config import CaptureConfig
from cs2_rl_bot.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


def _resize(image: np.ndarray, size: tuple[int, int], grayscale: bool) -> np.ndarray:
    """Lazy-import OpenCV to keep import-time light when capture is unused."""
    import cv2

    h, w = size
    out = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    if grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = out[..., None]  # keep (H, W, 1)
    return out


class ScreenCapture:
    """Threaded screen capture producing the latest frame on demand."""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._queue: Queue[Frame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="cs2-rl-bot-capture",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Screen capture started (monitor={}, fps={}, size={}x{})",
            self._config.monitor_index,
            self._config.target_fps,
            self._config.resize_to[1],
            self._config.resize_to[0],
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self, timeout: float | None = 0.1) -> Frame | None:
        """Return the latest frame, blocking up to ``timeout`` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stream(self) -> Iterator[Frame]:
        """Yield frames as they arrive; stops when ``stop()`` is called."""
        while not self._stop.is_set():
            frame = self.latest(timeout=0.5)
            if frame is not None:
                yield frame

    def _run(self) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover — logged for the operator
            logger.error("mss not installed; screen capture disabled: {}", exc)
            return

        target_dt = 1.0 / max(self._config.target_fps, 1)
        with mss.mss() as sct:
            try:
                monitor = sct.monitors[self._config.monitor_index]
            except IndexError:
                logger.warning(
                    "monitor_index={} out of range; falling back to monitor 1",
                    self._config.monitor_index,
                )
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

            while not self._stop.is_set():
                start = time.perf_counter()
                raw = np.asarray(sct.grab(monitor), dtype=np.uint8)  # BGRA
                # Drop alpha and convert BGR -> RGB in one slice.
                rgb = raw[:, :, [2, 1, 0]]
                resized = _resize(rgb, self._config.resize_to, self._config.grayscale)

                frame = Frame(image=resized, timestamp=time.time())
                # Single-slot queue: drop the previous unread frame if any.
                if self._queue.full():
                    with _suppress_empty():
                        self._queue.get_nowait()
                self._queue.put(frame)

                elapsed = time.perf_counter() - start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)


def _suppress_empty():  # pragma: no cover — trivial helper
    import contextlib

    return contextlib.suppress(Empty)
