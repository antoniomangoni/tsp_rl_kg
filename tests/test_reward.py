"""Tests for RewardCalculator in tsp_rl_kg.rl.reward."""

from __future__ import annotations

import pytest

from tsp_rl_kg.rl.reward import RewardCalculator, manhattan_distance

# ---------------------------------------------------------------------------
# manhattan_distance utility
# ---------------------------------------------------------------------------


class TestManhattanDistance:
    def test_same_point(self):
        assert manhattan_distance((0, 0), (0, 0)) == 0

    def test_horizontal(self):
        assert manhattan_distance((0, 0), (3, 0)) == 3

    def test_vertical(self):
        assert manhattan_distance((0, 0), (0, 5)) == 5

    def test_diagonal(self):
        assert manhattan_distance((1, 2), (4, 6)) == 7

    def test_negative_coordinates(self):
        assert manhattan_distance((-1, -2), (1, 2)) == 6


# ---------------------------------------------------------------------------
# step_penalty
# ---------------------------------------------------------------------------


class TestStepPenalty:
    def test_returns_negative(self, reward_calculator: RewardCalculator):
        result = reward_calculator.step_penalty(terrain_energy=2.0)
        assert result < 0

    def test_scales_with_terrain_energy(self, reward_calculator: RewardCalculator):
        low = reward_calculator.step_penalty(terrain_energy=1.0)
        high = reward_calculator.step_penalty(terrain_energy=5.0)
        assert high < low  # Higher energy → larger penalty (more negative)


# ---------------------------------------------------------------------------
# time_penalty
# ---------------------------------------------------------------------------


class TestTimePenalty:
    def test_returns_negative(self, reward_calculator: RewardCalculator):
        result = reward_calculator.time_penalty(episode_step=100)
        assert result < 0

    def test_increases_over_episode(self, reward_calculator: RewardCalculator):
        early = reward_calculator.time_penalty(episode_step=10)
        late = reward_calculator.time_penalty(episode_step=500)
        assert late < early  # Later step → larger penalty


# ---------------------------------------------------------------------------
# outpost_discovery_reward
# ---------------------------------------------------------------------------


class TestOutpostDiscoveryReward:
    def test_first_outpost_positive(self, reward_calculator: RewardCalculator):
        reward_calculator.outposts_visited.add((1, 1))
        result = reward_calculator.outpost_discovery_reward()
        assert result > 0

    def test_reward_increases_with_more_outposts(self, reward_calculator: RewardCalculator):
        reward_calculator.outposts_visited.add((1, 1))
        first = reward_calculator.outpost_discovery_reward()
        reward_calculator.outposts_visited.add((3, 3))
        second = reward_calculator.outpost_discovery_reward()
        assert second > first


# ---------------------------------------------------------------------------
# completion_reward
# ---------------------------------------------------------------------------


class TestCompletionReward:
    def test_positive(self, reward_calculator: RewardCalculator):
        result = reward_calculator.completion_reward(episode_step=100)
        assert result > 0

    def test_earlier_completion_higher(self, reward_calculator: RewardCalculator):
        early = reward_calculator.completion_reward(episode_step=50)
        late = reward_calculator.completion_reward(episode_step=500)
        assert early > late  # Faster completion → bigger bonus


# ---------------------------------------------------------------------------
# route_improvement_reward
# ---------------------------------------------------------------------------


class TestRouteImprovementReward:
    def test_improved_route_positive(self, reward_calculator: RewardCalculator):
        # First route: set a baseline
        reward_calculator.best_efficiency = 0.0
        reward_calculator.previous_best_route_energy = float("inf")
        reward, early_stop = reward_calculator.route_improvement_reward(
            agent_route_energy=100.0,
            algorithmic_best_energy=80.0,
        )
        assert reward > 0
        assert early_stop is False

    def test_no_improvement_increments_counter(self, reward_calculator: RewardCalculator):
        # Set a very high best efficiency so no improvement is possible
        reward_calculator.best_efficiency = 200.0
        reward, _ = reward_calculator.route_improvement_reward(
            agent_route_energy=200.0,
            algorithmic_best_energy=100.0,
        )
        assert reward == 0.0
        assert reward_calculator.num_not_improvement_routes == 1

    def test_early_stop_after_max_no_improvement(self, reward_calculator: RewardCalculator):
        reward_calculator.best_efficiency = 200.0
        for _ in range(reward_calculator.config.max_not_improvement_routes):
            _, early_stop = reward_calculator.route_improvement_reward(
                agent_route_energy=200.0,
                algorithmic_best_energy=100.0,
            )
        assert early_stop is True


# ---------------------------------------------------------------------------
# proximity_reward
# ---------------------------------------------------------------------------


class TestProximityReward:
    def test_closer_positive(self, reward_calculator: RewardCalculator):
        reward_calculator.previous_min_distance = 10.0
        result = reward_calculator.proximity_reward(agent_pos=(1, 0))
        assert result > 0

    def test_farther_negative(self, reward_calculator: RewardCalculator):
        reward_calculator.previous_min_distance = 1.0
        # Move far away from all outposts
        result = reward_calculator.proximity_reward(agent_pos=(0, 0))
        assert result < 0

    def test_all_visited_returns_zero(self, reward_calculator: RewardCalculator):
        # Mark all outposts as visited
        for coord in reward_calculator.outpost_coords:
            reward_calculator.outposts_visited.add(coord)
        result = reward_calculator.proximity_reward(agent_pos=(0, 0))
        assert result == 0.0


# ---------------------------------------------------------------------------
# circular_behavior_penalty
# ---------------------------------------------------------------------------


class TestCircularBehaviorPenalty:
    def test_repeated_position_penalty(self, reward_calculator: RewardCalculator):
        reward_calculator.recent_path.append((2, 2))
        result = reward_calculator.circular_behavior_penalty(agent_pos=(2, 2))
        assert result < 0

    def test_new_position_no_penalty(self, reward_calculator: RewardCalculator):
        result = reward_calculator.circular_behavior_penalty(agent_pos=(2, 2))
        assert result == 0.0


# ---------------------------------------------------------------------------
# calculate orchestrator
# ---------------------------------------------------------------------------


class TestCalculateOrchestrator:
    def test_returns_tuple(self, reward_calculator: RewardCalculator):
        result = reward_calculator.calculate(
            agent_pos=(0, 0),
            terrain_energy=2.0,
            episode_step=10,
            agent_energy_spent=50.0,
            algorithmic_best_energy=40.0,
            reset_energy_callback=lambda: None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        reward, early_stop = result
        assert isinstance(reward, float)
        assert isinstance(early_stop, bool)

    def test_reward_is_normalised(self, reward_calculator: RewardCalculator):
        reward, _ = reward_calculator.calculate(
            agent_pos=(0, 0),
            terrain_energy=2.0,
            episode_step=10,
            agent_energy_spent=50.0,
            algorithmic_best_energy=40.0,
            reset_energy_callback=lambda: None,
        )
        assert 0 <= reward <= reward_calculator.config.normalisation_scale


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_episode_state(self, reward_calculator: RewardCalculator):
        reward_calculator.outposts_visited.add((1, 1))
        reward_calculator.recent_path.append((2, 2))
        reward_calculator.previous_min_distance = 5.0

        reward_calculator.reset()

        assert len(reward_calculator.outposts_visited) == 0
        assert len(reward_calculator.recent_path) == 0
        assert reward_calculator.previous_min_distance == float("inf")

    def test_reset_game_clears_all(self, reward_calculator: RewardCalculator):
        reward_calculator.best_route_energy = 100.0
        reward_calculator.best_efficiency = 50.0
        reward_calculator.outposts_visited.add((1, 1))

        reward_calculator.reset_game(outpost_coords=[(0, 0), (2, 2)])

        assert reward_calculator.best_route_energy == float("inf")
        assert reward_calculator.best_efficiency == 0.0
        assert len(reward_calculator.outposts_visited) == 0
        assert reward_calculator.outpost_coords == [(0, 0), (2, 2)]


# ---------------------------------------------------------------------------
# Normalisation & efficiency helpers
# ---------------------------------------------------------------------------


class TestEfficiencyHelpers:
    def test_calculate_route_efficiency(self, reward_calculator: RewardCalculator):
        eff = reward_calculator.calculate_route_efficiency(
            agent_route_energy=100.0,
            algorithmic_best_energy=80.0,
        )
        assert eff == pytest.approx(80.0)  # (80/100) * 100

    def test_calculate_route_efficiency_zero_algo(self, reward_calculator: RewardCalculator):
        with pytest.raises(ValueError, match="cannot be zero"):
            reward_calculator.calculate_route_efficiency(
                agent_route_energy=100.0,
                algorithmic_best_energy=0.0,
            )

    def test_interpret_efficiency_perfect(self, reward_calculator: RewardCalculator):
        result = reward_calculator.interpret_efficiency(100.0)
        assert "Matching" in result

    def test_interpret_efficiency_imperfect(self, reward_calculator: RewardCalculator):
        result = reward_calculator.interpret_efficiency(80.0)
        assert "away" in result
