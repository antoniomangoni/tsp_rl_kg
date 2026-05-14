from __future__ import annotations

from collections import deque

from loguru import logger

from tsp_rl_kg.config import RewardComponent, RewardConfig


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class RewardCalculator:
    """Composable reward system extracted from CustomEnv._calculate_reward().

    All reward weights are driven by ``RewardConfig``; the calculator is
    stateful (tracks visited outposts, recent path, route history) and
    must be ``reset()``-ed at the start of each episode.
    """

    def __init__(
        self,
        config: RewardConfig,
        outpost_coords: list[tuple[int, int]],
        max_episode_steps: int,
        disabled_components: list[RewardComponent] | None = None,
    ) -> None:
        self.config = config
        self.outpost_coords = outpost_coords
        self.max_episode_steps = max_episode_steps
        self.disabled_components: set[RewardComponent] = set(disabled_components or [])

        # Episode-scoped state
        self.outposts_visited: set[tuple[int, int]] = set()
        self.recent_path: deque[tuple[int, int]] = deque(maxlen=len(outpost_coords))
        self.previous_min_distance: float = float("inf")

        # Route-improvement tracking (persists across episodes on same game)
        self.best_route_energy: float = float("inf")
        self.previous_best_route_energy: float = float("inf")
        self.best_efficiency: float = 0.0
        self.current_efficiency: float = 0.0
        self.improvement: float = 0.0
        self.gap: float = 0.0
        self.num_not_improvement_routes: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode-scoped state. Call at the start of each episode."""
        self.outposts_visited.clear()
        self.recent_path = deque(maxlen=len(self.outpost_coords))
        self.previous_min_distance = float("inf")
        self.num_not_improvement_routes = 0

    def reset_game(self, outpost_coords: list[tuple[int, int]]) -> None:
        """Full reset when switching to a new game world."""
        self.outpost_coords = outpost_coords
        self.best_route_energy = float("inf")
        self.previous_best_route_energy = float("inf")
        self.best_efficiency = 0.0
        self.current_efficiency = 0.0
        self.improvement = 0.0
        self.gap = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # Individual reward components
    # ------------------------------------------------------------------

    def step_penalty(self, terrain_energy: float) -> float:
        return self.config.penalty_per_step * terrain_energy

    def time_penalty(self, episode_step: int) -> float:
        return self.config.time_penalty_factor * episode_step

    def outpost_discovery_reward(self) -> float:
        outposts_visited = len(self.outposts_visited)
        return self.config.new_outpost_reward * (
            1 + self.config.outpost_reward_increase_factor * (outposts_visited - 1)
        )

    def completion_reward(self, episode_step: int) -> float:
        return self.config.completion_reward * (
            1 + self.config.completion_time_bonus_factor / episode_step
        )

    def route_improvement_reward(
        self,
        agent_route_energy: float,
        algorithmic_best_energy: float,
    ) -> tuple[float, bool]:
        """Compute reward for route improvement.

        Returns ``(reward, early_stop)`` - *early_stop* is ``True`` when the
        agent has exhausted its non-improvement budget.
        """
        reward = 0.0
        early_stop = False

        self.current_efficiency = self.calculate_route_efficiency(
            agent_route_energy, algorithmic_best_energy
        )
        logger.info(
            f"Route Efficiency: {self.current_efficiency:.2f} "
            f"- {self.interpret_efficiency(self.current_efficiency)}"
        )

        if self.current_efficiency > self.best_efficiency:
            self.improvement = self.calculate_relative_improvement(
                agent_route_energy, algorithmic_best_energy
            )
            self.gap = self.calculate_efficiency_gap(agent_route_energy, algorithmic_best_energy)
            logger.info(
                f"New best route found. Improvement: {self.improvement:.2f}, Gap: {self.gap:.2f}"
            )
            improvement_reward = self.config.completion_reward * self.improvement
            reward += improvement_reward

            self.best_route_energy = agent_route_energy
            self.best_efficiency = self.current_efficiency
            self.num_not_improvement_routes = 0

            if agent_route_energy < algorithmic_best_energy:
                reward += self.config.better_route_than_algo_reward
        else:
            self.num_not_improvement_routes += 1
            if self.num_not_improvement_routes >= self.config.max_not_improvement_routes:
                early_stop = True

        self.previous_best_route_energy = min(self.previous_best_route_energy, agent_route_energy)
        return reward, early_stop

    def proximity_reward(self, agent_pos: tuple[int, int]) -> float:
        unvisited_outposts = set(self.outpost_coords) - self.outposts_visited
        if not unvisited_outposts:
            return 0.0

        current_min_distance = min(
            manhattan_distance(agent_pos, outpost) for outpost in unvisited_outposts
        )
        reward = 0.0

        if current_min_distance < self.previous_min_distance:
            reward = (
                self.config.closer_to_outpost_reward
                * (self.previous_min_distance - current_min_distance)
                / self.previous_min_distance
            )
            logger.info(f"Agent moved closer to an outpost. Reward: {reward}")
        elif current_min_distance > self.previous_min_distance:
            reward = (
                self.config.farther_from_outpost_penalty
                * (current_min_distance - self.previous_min_distance)
                / self.previous_min_distance
            )
            logger.info(f"Agent moved away from outposts. Penalty: {reward}")

        self.previous_min_distance = current_min_distance
        return reward

    def circular_behavior_penalty(self, agent_pos: tuple[int, int]) -> float:
        if agent_pos in self.recent_path:
            logger.info(f"Agent repeated a path. Penalty: {self.config.circular_behavior_penalty}")
            return self.config.circular_behavior_penalty
        return 0.0

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def calculate(
        self,
        agent_pos: tuple[int, int],
        terrain_energy: float,
        episode_step: int,
        agent_energy_spent: float,
        algorithmic_best_energy: float,
        reset_energy_callback: callable,
    ) -> tuple[float, bool, bool]:
        """Compute the total reward for the current step.

        Returns ``(normalised_reward, early_stop, all_visited)``.
        """
        logger.info("Calculating reward...")
        early_stop = False
        all_visited = False

        # Base penalties
        reward = self.step_penalty(terrain_energy)
        reward += self.time_penalty(episode_step)

        # Outpost discovery
        if agent_pos in self.outpost_coords and agent_pos not in self.outposts_visited:
            self.outposts_visited.add(agent_pos)
            reward += self.outpost_discovery_reward()
            logger.info(
                f"Outposts visited: {len(self.outposts_visited)}/{len(self.outpost_coords)}"
            )
            self.recent_path.clear()

            # All outposts visited → completion
            if len(self.outposts_visited) == len(self.outpost_coords):
                all_visited = True
                logger.debug(
                    f"Agent reached all outposts. Outposts visited: {self.outposts_visited}"
                )
                self.outposts_visited.clear()

                reward += self.completion_reward(episode_step)
                logger.debug(
                    f"Step: {episode_step}. All outposts visited. Completion reward: {reward}"
                )

                reset_energy_callback()

                logger.debug(
                    f"Agent route energy: {agent_energy_spent}, "
                    f"algorithmic best energy: {algorithmic_best_energy}"
                )

                improvement_reward, early_stop = self.route_improvement_reward(
                    agent_energy_spent, algorithmic_best_energy
                )
                if RewardComponent.ROUTE_IMPROVEMENT not in self.disabled_components:
                    reward += improvement_reward

                logger.info(
                    f"Route Completed - Efficiency: {self.current_efficiency:.2f}, "
                    f"Improvement: {self.improvement:.2f}, Gap: {self.gap:.2f}"
                )
            else:
                # Proximity shaping (only when not all outposts complete)
                if RewardComponent.PROXIMITY not in self.disabled_components:
                    reward += self.proximity_reward(agent_pos)

        # Circular behaviour
        if RewardComponent.CIRCULAR_PENALTY not in self.disabled_components:
            reward += self.circular_behavior_penalty(agent_pos)

        # Update path memory
        self.recent_path.append(agent_pos)

        return self._normalize_reward(reward), early_stop, all_visited

    # ------------------------------------------------------------------
    # Normalisation & efficiency helpers
    # ------------------------------------------------------------------

    def _normalize_reward(self, reward: float) -> float:
        """Scale reward to [-1, 1] via min-max normalization."""
        min_reward = (
            self.config.penalty_per_step * self.max_episode_steps
            + self.config.time_penalty_factor * self.max_episode_steps
            + self.config.circular_behavior_penalty
        )
        max_reward = (
            self.config.completion_reward
            + self.config.new_outpost_reward * len(self.outpost_coords)
            + self.config.route_improvement_reward
            + self.config.better_route_than_algo_reward
        )
        if max_reward <= min_reward:
            return 0.0
        normalized = 2.0 * (reward - min_reward) / (max_reward - min_reward) - 1.0
        return max(-1.0, min(normalized, 1.0))

    def calculate_route_efficiency(
        self, agent_route_energy: float, algorithmic_best_energy: float
    ) -> float:
        if algorithmic_best_energy == 0:
            raise ValueError("Algorithmic best energy cannot be zero")
        return (algorithmic_best_energy / agent_route_energy) * self.config.normalisation_scale

    def calculate_relative_improvement(
        self, current_route_energy: float, algorithmic_best_energy: float
    ) -> float:
        current_efficiency = self.calculate_route_efficiency(
            current_route_energy, algorithmic_best_energy
        )
        previous_efficiency = self.calculate_route_efficiency(
            self.previous_best_route_energy, algorithmic_best_energy
        )
        if previous_efficiency == 0:
            return float("inf") if current_efficiency > 0 else 0
        return (current_efficiency - previous_efficiency) / previous_efficiency

    def calculate_efficiency_gap(
        self, agent_route_energy: float, algorithmic_best_energy: float
    ) -> float:
        return max(
            0,
            (agent_route_energy - algorithmic_best_energy) / algorithmic_best_energy,
        )

    def interpret_efficiency(self, efficiency: float) -> str:
        if efficiency == self.config.normalisation_scale:
            return "Matching algorithmic best"
        return f"{self.config.normalisation_scale - efficiency:.2f}% away from algorithmic best"
