"""Regression tests for CustomEnv construction."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tsp_rl_kg.config import (
    EpisodeConfig,
    FeatureEncodingConfig,
    GameManagerConfig,
    ModelArgs,
    SimulationManagerConfig,
)
from tsp_rl_kg.graph.feature_encoder import build_feature_encoder, embedding_metadata_path
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager

_SCHEMA_TEXT = """
[meta]
version = 1

[terrain]
"0" = "deep water terrain"
"1" = "shallow water terrain"
"2" = "open plains terrain"
"3" = "rolling hills terrain"
"4" = "rugged mountain terrain"
"5" = "snow-covered highland terrain"

[entity]
"0" = "empty tile"
"1" = "fish resource"
"2" = "forest tree resource"
"3" = "mossy rock obstacle"
"4" = "snowy rock obstacle"
"5" = "remote outpost destination"
"6" = "wooden path marker"
"7" = "reserved player entity slot"

[player]
descriptor = "travelling player agent"
""".strip()


def _write_embedding_assets(schema_path, embedding_path, embed_dim=6):
    schema_path.write_text(_SCHEMA_TEXT, encoding="utf-8")
    embeddings = np.arange(15 * embed_dim, dtype=np.float32).reshape(15, embed_dim)
    np.save(embedding_path, embeddings)
    embedding_metadata_path(embedding_path).write_text(
        json.dumps(
            {
                "descriptor_count": 15,
                "embed_dim": embed_dim,
                "schema_hash": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "model_name": "fixture-model",
            }
        ),
        encoding="utf-8",
    )


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


def test_custom_env_observation_width_tracks_feature_encoder_node_dim(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    embedding_path = tmp_path / "semantic_embeddings.npy"
    _write_embedding_assets(schema_path, embedding_path, embed_dim=6)

    feature_encoder = build_feature_encoder(
        FeatureEncodingConfig(
            strategy="embedding_lookup",
            schema_path=str(schema_path),
            embedding_path=str(embedding_path),
        ),
        grid_size=5,
    )

    env = CustomEnv(
        GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        SimulationManagerConfig(
            number_of_environments=12,
            number_of_curricula=3,
            min_episodes_per_curriculum=1,
        ),
        ModelArgs(num_actions=11),
        feature_encoder=feature_encoder,
    )

    assert env.observation_space.spaces["node_features"].shape[1] == feature_encoder.node_dim
    env.close()
