"""Tests for the reward calculator."""

from __future__ import annotations

import pytest

from cs2_rl_bot.env.rewards import RewardCalculator
from cs2_rl_bot.observation.state import (
    Observation,
    PlayerState,
    RoundPhase,
    RoundState,
    Team,
)
from cs2_rl_bot.utils.config import RewardConfig


def make_obs(
    *,
    health: int = 100,
    kills: int = 0,
    deaths: int = 0,
    damage: int = 0,
    phase: RoundPhase = RoundPhase.LIVE,
    win_team: Team = Team.UNKNOWN,
    round_number: int = 1,
    bomb_planted: bool = False,
    bomb_defused: bool = False,
    team: Team = Team.T,
) -> Observation:
    return Observation(
        player=PlayerState(
            health=health,
            round_kills=kills,
            round_deaths=deaths,
            round_damage=damage,
            team=team,
        ),
        round=RoundState(
            phase=phase,
            win_team=win_team,
            round_number=round_number,
            bomb_planted=bomb_planted,
            bomb_defused=bomb_defused,
        ),
    )


@pytest.fixture
def calc() -> RewardCalculator:
    return RewardCalculator(RewardConfig())


def test_first_observation_has_only_survival(calc: RewardCalculator) -> None:
    breakdown = calc.compute(make_obs(health=100))
    assert breakdown.survival > 0
    assert breakdown.kills == 0
    assert breakdown.deaths == 0


def test_kill_increases_reward(calc: RewardCalculator) -> None:
    calc.compute(make_obs(kills=0))
    breakdown = calc.compute(make_obs(kills=1))
    assert breakdown.kills == pytest.approx(1.0)


def test_damage_taken_penalised(calc: RewardCalculator) -> None:
    calc.compute(make_obs(health=100))
    breakdown = calc.compute(make_obs(health=70))
    assert breakdown.damage_taken == pytest.approx(-0.30, rel=1e-3)


def test_round_win_applied_once(calc: RewardCalculator) -> None:
    calc.compute(make_obs(phase=RoundPhase.LIVE))
    first = calc.compute(make_obs(phase=RoundPhase.OVER, win_team=Team.T, round_number=1))
    second = calc.compute(make_obs(phase=RoundPhase.OVER, win_team=Team.T, round_number=1))
    assert first.round_outcome == pytest.approx(2.0)
    assert second.round_outcome == 0.0


def test_round_loss_for_other_team(calc: RewardCalculator) -> None:
    calc.compute(make_obs(team=Team.T, phase=RoundPhase.LIVE))
    breakdown = calc.compute(
        make_obs(team=Team.T, phase=RoundPhase.OVER, win_team=Team.CT, round_number=1)
    )
    assert breakdown.round_outcome == pytest.approx(-1.0)


def test_kill_switch_penalty(calc: RewardCalculator) -> None:
    breakdown = calc.compute(make_obs(), kill_switch_engaged=True)
    assert breakdown.kill_switch == pytest.approx(-0.1)


def test_bomb_plant_bonus_for_t(calc: RewardCalculator) -> None:
    calc.compute(make_obs(team=Team.T, bomb_planted=False))
    breakdown = calc.compute(make_obs(team=Team.T, bomb_planted=True))
    assert breakdown.bomb == pytest.approx(0.5)


def test_total_sums_components(calc: RewardCalculator) -> None:
    calc.compute(make_obs(kills=0, health=100))
    breakdown = calc.compute(make_obs(kills=1, health=80))
    expected = breakdown.kills + breakdown.damage_taken + breakdown.survival
    assert breakdown.total == pytest.approx(expected)
