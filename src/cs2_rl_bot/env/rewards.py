"""Reward function shaped from GSI deltas + survival heuristics.

The reward is computed as the difference between the previous and current
:class:`Observation`. Per-step shaping is intentionally small so the agent
learns to optimise round-level outcomes (kills, wins, bomb plants/defuses)
rather than chase noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from cs2_rl_bot.observation.state import Observation, RoundPhase, Team
from cs2_rl_bot.utils.config import RewardConfig


@dataclass(slots=True)
class RewardBreakdown:
    """Inspectable per-step reward components, useful for tensorboard."""

    kills: float = 0.0
    deaths: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    round_outcome: float = 0.0
    bomb: float = 0.0
    survival: float = 0.0
    kill_switch: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.kills
            + self.deaths
            + self.damage_dealt
            + self.damage_taken
            + self.round_outcome
            + self.bomb
            + self.survival
            + self.kill_switch
        )


class RewardCalculator:
    """Stateless-ish reward function — keeps the previous observation."""

    def __init__(self, config: RewardConfig) -> None:
        self._cfg = config
        self._previous: Observation | None = None
        self._round_resolved: int = -1

    def reset(self) -> None:
        self._previous = None
        self._round_resolved = -1

    def compute(
        self,
        observation: Observation,
        *,
        kill_switch_engaged: bool = False,
    ) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        prev = self._previous

        if prev is not None:
            d_kills = max(observation.player.round_kills - prev.player.round_kills, 0)
            d_dmg_dealt = max(observation.player.round_damage - prev.player.round_damage, 0)
            d_hp_taken = max(prev.player.health - observation.player.health, 0)
            d_deaths = max(observation.player.round_deaths - prev.player.round_deaths, 0)

            breakdown.kills = self._cfg.kill_bonus * d_kills
            breakdown.deaths = self._cfg.death_penalty * d_deaths
            breakdown.damage_dealt = self._cfg.damage_dealt_coef * d_dmg_dealt
            breakdown.damage_taken = self._cfg.damage_taken_coef * d_hp_taken

            # Bomb events (one-shot per round)
            if (
                self._cfg.bomb_plant_bonus
                and observation.round.bomb_planted
                and not prev.round.bomb_planted
            ):
                breakdown.bomb += self._cfg.bomb_plant_bonus * (
                    1.0 if observation.player.team is Team.T else -1.0
                )
            if (
                self._cfg.bomb_defuse_bonus
                and observation.round.bomb_defused
                and not prev.round.bomb_defused
            ):
                breakdown.bomb += self._cfg.bomb_defuse_bonus * (
                    1.0 if observation.player.team is Team.CT else -1.0
                )

        # Round outcome — apply once when the round transitions to "over".
        round_no = observation.round.round_number
        if (
            observation.round.phase is RoundPhase.OVER
            and observation.round.win_team is not Team.UNKNOWN
            and round_no != self._round_resolved
        ):
            self._round_resolved = round_no
            won = observation.round.win_team is observation.player.team
            breakdown.round_outcome = (
                self._cfg.round_win_bonus if won else self._cfg.round_loss_penalty
            )

        # Survival: small per-step bonus while the player is alive.
        if observation.player.health > 0 and observation.round.phase is RoundPhase.LIVE:
            breakdown.survival = self._cfg.survival_per_step

        if kill_switch_engaged:
            breakdown.kill_switch = self._cfg.kill_switch_penalty

        self._previous = observation
        return breakdown
