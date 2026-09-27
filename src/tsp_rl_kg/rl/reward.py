"""Stationary rewards computed from transitions, without owning episode state."""

from dataclasses import dataclass

from tsp_rl_kg.config import RewardComponent, RewardConfig
from tsp_rl_kg.rl.episode import efficiency


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class RewardTransition:
    energy_delta: float
    episode_energy: float
    reference_energy: float
    new_outpost: bool
    success: bool
    potential_before: float
    potential_after: float


class RewardCalculator:
    def __init__(self, config: RewardConfig, *, gamma=0.99, disabled_components=()):
        self.config = config
        self.gamma = gamma
        self.disabled_components = set(disabled_components)

    def calculate(self, transition: RewardTransition) -> float:
        reward = self.config.penalty_per_step * transition.energy_delta
        reward += self.config.new_outpost_reward * transition.new_outpost
        if transition.success:
            reward += self.config.completion_reward * min(
                efficiency(transition.reference_energy, transition.episode_energy), 2.0
            )
        if RewardComponent.PROXIMITY not in self.disabled_components:
            reward += self.config.shaping_weight * (
                self.gamma * transition.potential_after - transition.potential_before
            )
        return float(reward)
