"""Regression tests for CustomEnv construction."""

from __future__ import annotations

from tsp_rl_kg.config import EpisodeConfig, GameManagerConfig, ModelArgs, SimulationManagerConfig
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager


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


def test_environment_manager_passes_episode_config_to_custom_env():
    manager = EnvironmentManager(
        GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        SimulationManagerConfig(
            number_of_environments=12,
            number_of_curricula=3,
            min_episodes_per_curriculum=1,
        ),
        ModelArgs(num_actions=11),
        feature_encoder=None,
        episode_config=EpisodeConfig(max_episode_steps=7, max_steps_without_progress=3),
    )

    env = manager.make_env()

    assert env.max_episode_steps == 7
    assert env.max_steps_without_progress == 3
    env.close()
