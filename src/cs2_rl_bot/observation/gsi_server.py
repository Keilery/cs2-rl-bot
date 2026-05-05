"""HTTP server that receives Game State Integration (GSI) payloads from CS2.

CS2 supports the same GSI mechanism as CS:GO. When ``cfg/gamestate_integration_*.cfg``
is installed in the game's ``cfg`` directory, the client POSTs JSON updates to
the configured URL on every state change. This module exposes the latest
payload as an asyncio-friendly value plus an event that fires on every update.

Run standalone for debugging::

    python -m cs2_rl_bot.observation.gsi_server
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from cs2_rl_bot.utils.config import GSIConfig
from cs2_rl_bot.utils.logging import logger


class GSIState:
    """Thread-safe container for the most recent GSI payload."""

    def __init__(self) -> None:
        self._payload: dict[str, Any] = {}
        self._update_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    async def set(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._payload = payload
            self._update_count += 1
            self._update_event.set()
            self._update_event.clear()

    async def get(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._payload)

    async def wait_for_update(self, timeout: float | None = None) -> dict[str, Any]:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._update_event.wait(), timeout=timeout)
        return await self.get()


def build_app(config: GSIConfig, state: GSIState) -> FastAPI:
    """Construct the FastAPI app bound to the given state container."""

    app = FastAPI(title="CS2 GSI listener", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "updates": state.update_count}

    @app.post("/")
    async def gsi_endpoint(request: Request) -> dict[str, str]:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

        # GSI auth token sent in the payload under "auth.token".
        token = (payload.get("auth") or {}).get("token")
        if token != config.auth_token:
            logger.warning("GSI request with bad auth token (got={!r})", token)
            raise HTTPException(status_code=401, detail="bad auth token")

        await state.set(payload)
        logger.debug("GSI update #{count}", count=state.update_count)
        return {"status": "ok"}

    return app


async def run_server(config: GSIConfig, state: GSIState) -> None:
    """Start uvicorn and serve until cancelled."""
    app = build_app(config, state)
    uv_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)
    logger.info("GSI listener starting on http://{}:{}", config.host, config.port)
    with contextlib.suppress(asyncio.CancelledError):
        await server.serve()


def main() -> None:  # pragma: no cover — manual debugging entry point
    from cs2_rl_bot.utils.config import AppConfig
    from cs2_rl_bot.utils.logging import configure_logging

    cfg = AppConfig.load()
    configure_logging(cfg.log_level)
    asyncio.run(run_server(cfg.gsi, GSIState()))


if __name__ == "__main__":  # pragma: no cover
    main()
