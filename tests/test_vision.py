"""Unit tests for vision class classifier + enemy feature encoder."""

from __future__ import annotations

import numpy as np

from cs2_rl_bot.observation.state import Team
from cs2_rl_bot.observation.vision import (
    ENEMY_FEATURE_DIM,
    Detection,
    classify_class_name,
    encode_enemy_features,
)


def test_classify_known_short_codes() -> None:
    assert classify_class_name("c") == ("CT", False)
    assert classify_class_name("ch") == ("CT", True)
    assert classify_class_name("t") == ("T", False)
    assert classify_class_name("th") == ("T", True)


def test_classify_underscore_and_verbose_names() -> None:
    assert classify_class_name("ct_head") == ("CT", True)
    assert classify_class_name("CTbody") == ("CT", False)
    assert classify_class_name("Thead") == ("T", True)
    assert classify_class_name("HEAD_CT") == ("CT", True)


def test_classify_unknown_returns_none() -> None:
    assert classify_class_name("dog") == (None, False)
    assert classify_class_name("") == (None, False)


def test_classify_strips_whitespace_and_case() -> None:
    assert classify_class_name(" CT ") == ("CT", False)
    assert classify_class_name("  T  ") == ("T", False)


def test_encode_empty_returns_zero_vector() -> None:
    out = encode_enemy_features([], image_shape=(640, 640), our_team=Team.CT, max_slots=4)
    assert out.shape == (4 * ENEMY_FEATURE_DIM,)
    assert np.allclose(out, 0.0)


def test_encode_marks_slot_as_valid_with_correct_geometry() -> None:
    # Image is 100x200 (HxW). Detection bbox at top-left quadrant.
    det = Detection(
        cls="t",
        confidence=0.9,
        bbox=(10, 10, 30, 40),
        team="T",
        is_head=False,
    )
    out = encode_enemy_features([det], image_shape=(100, 200), our_team=Team.CT, max_slots=2)
    assert out.shape == (2 * ENEMY_FEATURE_DIM,)
    # Slot 0
    assert out[0] == 1.0  # valid
    centre_x, centre_y = 0.5 * (10 + 30), 0.5 * (10 + 40)
    expected_dx = (centre_x - 100) / 100  # screen centre is (100, 50)
    expected_dy = (centre_y - 50) / 50
    assert np.isclose(out[1], expected_dx, atol=1e-5)
    assert np.isclose(out[2], expected_dy, atol=1e-5)
    # Width 20px / 200 = 0.1, height 30px / 100 = 0.3.
    assert np.isclose(out[3], 0.1, atol=1e-5)
    assert np.isclose(out[4], 0.3, atol=1e-5)
    assert out[5] == 0.0  # not head
    assert out[6] == 1.0  # CT vs T -> enemy
    assert np.isclose(out[7], 0.9, atol=1e-5)
    # Slot 1 padded with zeros.
    assert np.allclose(out[ENEMY_FEATURE_DIM:], 0.0)


def test_encode_sorts_by_distance_to_centre() -> None:
    # 200x200 image, centre is at (100, 100).
    far_det = Detection(cls="t", confidence=0.5, bbox=(0, 0, 20, 20), team="T")
    near_det = Detection(cls="t", confidence=0.5, bbox=(95, 95, 105, 105), team="T")
    out = encode_enemy_features(
        [far_det, near_det],
        image_shape=(200, 200),
        our_team=Team.CT,
        max_slots=2,
    )
    # Nearest detection should be in slot 0 (smaller |dx|, |dy|).
    assert abs(out[1]) < abs(out[ENEMY_FEATURE_DIM + 1])


def test_encode_drops_unknown_classes() -> None:
    unknown = Detection(cls="dog", confidence=0.99, bbox=(0, 0, 10, 10), team=None)
    enemy = Detection(cls="t", confidence=0.5, bbox=(50, 50, 60, 60), team="T")
    out = encode_enemy_features(
        [unknown, enemy],
        image_shape=(100, 100),
        our_team=Team.CT,
        max_slots=2,
    )
    # Slot 0 should be the enemy (unknown was filtered out).
    assert out[0] == 1.0
    assert out[7] == pytest_approx(0.5)
    # Slot 1 zero (no other detections).
    assert np.allclose(out[ENEMY_FEATURE_DIM:], 0.0)


def test_encode_is_enemy_zero_for_teammate() -> None:
    # Our team CT, detection also CT -> not an enemy.
    teammate = Detection(cls="c", confidence=0.8, bbox=(40, 40, 60, 60), team="CT")
    out = encode_enemy_features(
        [teammate],
        image_shape=(100, 100),
        our_team=Team.CT,
        max_slots=1,
    )
    assert out[0] == 1.0  # still surfaced (don't shoot teammates)
    assert out[6] == 0.0  # is_enemy=False


def test_encode_caps_at_max_slots() -> None:
    dets = [
        Detection(cls="t", confidence=0.5, bbox=(i, i, i + 5, i + 5), team="T")
        for i in range(10, 100, 5)
    ]
    out = encode_enemy_features(dets, image_shape=(200, 200), our_team=Team.CT, max_slots=3)
    assert out.shape == (3 * ENEMY_FEATURE_DIM,)
    # All three slots filled.
    for i in range(3):
        assert out[i * ENEMY_FEATURE_DIM] == 1.0


def test_encode_unknown_team_means_no_enemy_signal() -> None:
    det = Detection(cls="t", confidence=0.7, bbox=(0, 0, 10, 10), team="T")
    out = encode_enemy_features([det], image_shape=(100, 100), our_team=Team.UNKNOWN, max_slots=1)
    # Detection still surfaced (valid=1) but is_enemy is 0 because we don't
    # know our own team yet — the policy gets dx/dy/conf without team filter.
    assert out[0] == 1.0
    assert out[6] == 0.0


def test_encode_handles_zero_image_shape() -> None:
    # Defensive: callers shouldn't pass (0,0) but if they do, we shouldn't crash.
    det = Detection(cls="t", confidence=0.5, bbox=(0, 0, 10, 10), team="T")
    out = encode_enemy_features([det], image_shape=(0, 0), our_team=Team.CT, max_slots=2)
    assert out.shape == (2 * ENEMY_FEATURE_DIM,)
    assert np.allclose(out, 0.0)


# pytest's approx is more readable than a direct float compare here.
def pytest_approx(value: float, tol: float = 1e-5) -> _Approx:
    return _Approx(value, tol)


class _Approx:
    def __init__(self, value: float, tol: float) -> None:
        self.value = value
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float, np.floating)):
            return NotImplemented
        return abs(float(other) - self.value) < self.tol

    def __repr__(self) -> str:
        return f"~{self.value}"
