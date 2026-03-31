"""Regression tests for CustomEnv construction."""

from __future__ import annotations

from tsp_rl_kg.config import GameManagerConfig, ModelArgs, SimulationManagerConfig
from tsp_rl_kg.rl.custom_env import CustomEnv


def test_custom_env_initialises_episode_limits_before_first_game_manager():
    env = CustomEnv(
        GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        SimulationManagerConfig(
            number_of_environments=12,
            number_of_curricula=3,
            min_episodes_per_curriculum=1,
        ),
        ModelArgs(num_actions=11),
    )

    assert env.max_episode_steps == env._episode_config.max_episode_steps
    env.close()
