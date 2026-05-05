"""Tests for the FastAPI GSI listener."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cs2_rl_bot.observation.gsi_server import GSIState, build_app
from cs2_rl_bot.utils.config import GSIConfig


@pytest.fixture
def client_and_state() -> tuple[TestClient, GSIState]:
    cfg = GSIConfig(auth_token="test-token")
    state = GSIState()
    app = build_app(cfg, state)
    return TestClient(app), state


def test_health(client_and_state: tuple[TestClient, GSIState]) -> None:
    client, _ = client_and_state
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "updates": 0}


def test_post_with_valid_token_updates_state(
    client_and_state: tuple[TestClient, GSIState],
) -> None:
    client, state = client_and_state
    payload = {
        "auth": {"token": "test-token"},
        "player": {"state": {"health": 73}},
    }
    resp = client.post("/", json=payload)
    assert resp.status_code == 200
    assert state.update_count == 1


def test_post_with_bad_token_rejected(
    client_and_state: tuple[TestClient, GSIState],
) -> None:
    client, state = client_and_state
    resp = client.post("/", json={"auth": {"token": "wrong"}})
    assert resp.status_code == 401
    assert state.update_count == 0


def test_post_with_invalid_json(client_and_state: tuple[TestClient, GSIState]) -> None:
    client, _ = client_and_state
    resp = client.post("/", content=b"not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400
