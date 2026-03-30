"""Shared pytest fixtures for tsp_rl_kg tests.

All fixtures use headless mode to avoid pygame initialisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsp_rl_kg.config import (
    EpisodeConfig,
    GameManagerConfig,
    RewardConfig,
)
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.knowledge.graph_idx_manager import Graph_Manager
from tsp_rl_kg.rl.reward import RewardCalculator

# ---------------------------------------------------------------------------
# Heightmaps
# ---------------------------------------------------------------------------


@pytest.fixture
def small_heightmap() -> np.ndarray:
    """Deterministic 5x5 heightmap exercising multiple terrain types.

    Terrain codes: 0=DeepWater, 1=Water, 2=Plains, 3=Hills, 4=Mountains, 5=Snow
    """
    return np.array(
        [
            [2, 2, 3, 4, 5],
            [2, 2, 3, 3, 4],
            [1, 2, 2, 3, 3],
            [0, 1, 2, 2, 3],
            [0, 0, 1, 2, 2],
        ],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def headless_game_config() -> GameManagerConfig:
    return GameManagerConfig(num_tiles=5, screen_size=250, vision_range=1, headless=True)


@pytest.fixture
def sample_reward_config() -> RewardConfig:
    return RewardConfig()


@pytest.fixture
def sample_episode_config() -> EpisodeConfig:
    return EpisodeConfig()


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def headless_environment(small_heightmap: np.ndarray) -> Environment:
    """Small headless Environment with 1 outpost."""
    return Environment(
        heightmap=small_heightmap,
        tile_size=50,
        number_of_outposts=1,
        headless=True,
    )


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_manager() -> Graph_Manager:
    gm = Graph_Manager()
    gm.set_max_nodes(100)
    gm.set_max_edges(200)
    return gm


# ---------------------------------------------------------------------------
# Reward fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reward_calculator(sample_reward_config: RewardConfig) -> RewardCalculator:
    outpost_coords = [(1, 1), (3, 3), (4, 0)]
    return RewardCalculator(
        config=sample_reward_config,
        outpost_coords=outpost_coords,
        max_episode_steps=1000,
    )
