"""Tests for config dataclasses in tsp_rl_kg.config."""

from __future__ import annotations

import pytest

from tsp_rl_kg.config import (
    AgentConfig,
    AgentModelConfig,
    AlgorithmConfig,
    AlgorithmName,
    CurriculumConfig,
    EpisodeConfig,
    EvaluationConfig,
    FeatureEncodingConfig,
    GameManagerConfig,
    ModelConfig,
    ReplayConfig,
    RewardConfig,
    RLBackend,
    SequenceConfig,
    SimulationManagerConfig,
    TrainingConfig,
    WorldModelConfig,
)

# ---------------------------------------------------------------------------
# GameManagerConfig
# ---------------------------------------------------------------------------


class TestGameManagerConfig:
    def test_defaults(self):
        cfg = GameManagerConfig()
        assert cfg.num_tiles == 32
        assert cfg.screen_size == 800
        assert cfg.vision_range == 2
        assert cfg.headless is False

    def test_custom_values(self):
        cfg = GameManagerConfig(num_tiles=5, screen_size=250, vision_range=1, headless=True)
        assert cfg.num_tiles == 5
        assert cfg.screen_size == 250
        assert cfg.vision_range == 1
        assert cfg.headless is True

    def test_invalid_num_tiles(self):
        with pytest.raises(ValueError, match="num_tiles must be >= 1"):
            GameManagerConfig(num_tiles=0)

    def test_screen_size_less_than_num_tiles(self):
        with pytest.raises(ValueError, match="screen_size .* must be >= num_tiles"):
            GameManagerConfig(num_tiles=10, screen_size=5)

    def test_negative_vision_range(self):
        with pytest.raises(ValueError, match="vision_range must be >= 0"):
            GameManagerConfig(vision_range=-1)


# ---------------------------------------------------------------------------
# SimulationManagerConfig
# ---------------------------------------------------------------------------


class TestSimulationManagerConfig:
    def test_defaults(self):
        cfg = SimulationManagerConfig()
        assert cfg.number_of_environments == 500
        assert cfg.number_of_curricula == 10
        assert cfg.min_episodes_per_curriculum == 1

    def test_invalid_number_of_environments(self):
        with pytest.raises(ValueError, match="number_of_environments must be >= 1"):
            SimulationManagerConfig(number_of_environments=0)

    def test_invalid_number_of_curricula(self):
        with pytest.raises(ValueError, match="number_of_curricula must be >= 1"):
            SimulationManagerConfig(number_of_curricula=0)


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------


class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.new_outpost_reward == 30.0
        assert cfg.completion_reward == 100.0
        assert cfg.penalty_per_step == -0.5
        assert cfg.normalisation_scale == 100.0
        assert cfg.max_not_improvement_routes == 5


# ---------------------------------------------------------------------------
# EpisodeConfig
# ---------------------------------------------------------------------------


class TestEpisodeConfig:
    def test_defaults(self):
        cfg = EpisodeConfig()
        assert cfg.max_episode_steps == 2048 * 8
        assert cfg.max_steps_without_progress == 2048 * 4
        assert cfg.max_game_worlds_trained_in == 100

    def test_invalid_max_episode_steps(self):
        with pytest.raises(ValueError, match="max_episode_steps must be >= 1"):
            EpisodeConfig(max_episode_steps=0)

    def test_invalid_max_steps_without_progress(self):
        with pytest.raises(ValueError, match="max_steps_without_progress must be >= 1"):
            EpisodeConfig(max_steps_without_progress=0)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.resource_max == 5
        assert cfg.action_energy_cost == 3
        assert cfg.scout_vision_multiplier == 2

    def test_invalid_resource_max(self):
        with pytest.raises(ValueError, match="resource_max must be >= 1"):
            AgentConfig(resource_max=0)

    def test_invalid_action_energy_cost(self):
        with pytest.raises(ValueError, match="action_energy_cost must be >= 0"):
            AgentConfig(action_energy_cost=-1)

    def test_invalid_scout_vision_multiplier(self):
        with pytest.raises(ValueError, match="scout_vision_multiplier must be >= 1"):
            AgentConfig(scout_vision_multiplier=0)


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    def test_defaults(self):
        cfg = CurriculumConfig()
        assert cfg.min_episodes_per_curriculum == 4
        assert cfg.performance_threshold == 0.85

    def test_invalid_performance_threshold(self):
        with pytest.raises(ValueError, match="performance_threshold must be in"):
            CurriculumConfig(performance_threshold=1.5)


# ---------------------------------------------------------------------------
# AgentModelConfig serialisation
# ---------------------------------------------------------------------------


class TestAgentModelConfig:
    def test_to_vision_params(self):
        cfg = AgentModelConfig()
        params = cfg.to_vision_params()
        assert "num_conv_layers" in params
        assert "conv_channels" in params
        assert "fc_dims" in params
        assert params["num_conv_layers"] == 4

    def test_to_graph_params(self):
        cfg = AgentModelConfig()
        params = cfg.to_graph_params()
        assert "num_gat_layers" in params
        assert "gat_heads" in params
        assert "fc_dims" in params
        assert params["num_gat_layers"] == 3

    def test_defaults(self):
        cfg = AgentModelConfig()
        assert cfg.features_dim == 192
        assert cfg.dropout == 0.25
        assert cfg.gat_hidden_dim == 48


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_to_dict(self):
        cfg = ModelConfig()
        d = cfg.to_dict()
        assert d["n_steps"] == 4096
        assert d["batch_size"] == 512
        assert isinstance(d["learning_rate"], float)
        assert isinstance(d["gamma"], float)


class TestAlgorithmConfig:
    def test_defaults(self):
        cfg = AlgorithmConfig()
        assert cfg.backend == RLBackend.SB3
        assert cfg.algorithm == AlgorithmName.PPO
        assert cfg.policy_name == "MultiInputPolicy"
        assert cfg.hyperparameters["n_steps"] == 4096

    def test_non_ppo_defaults_follow_selected_algorithm(self):
        cfg = AlgorithmConfig(algorithm=AlgorithmName.DQN)
        assert cfg.algorithm == AlgorithmName.DQN
        assert cfg.hyperparameters["buffer_size"] == 100_000
        assert "n_steps" not in cfg.hyperparameters

    def test_partial_hyperparameters_merge_algorithm_defaults(self):
        cfg = AlgorithmConfig(
            algorithm=AlgorithmName.DQN,
            hyperparameters={"buffer_size": 5_000},
        )
        assert cfg.hyperparameters["buffer_size"] == 5_000
        assert cfg.hyperparameters["learning_rate"] == pytest.approx(1e-4)

    def test_from_legacy_model_config(self):
        cfg = AlgorithmConfig.from_legacy_model_config(ModelConfig(n_steps=2048, gamma=0.99))
        assert cfg.algorithm == AlgorithmName.PPO
        assert cfg.hyperparameters["n_steps"] == 2048
        assert cfg.hyperparameters["gamma"] == pytest.approx(0.99)


class TestExtendedTrainingConfigs:
    def test_evaluation_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.eval_freq == 10_000
        assert cfg.n_eval_episodes == 10

    def test_replay_defaults(self):
        cfg = ReplayConfig()
        assert cfg.buffer_size == 100_000
        assert cfg.train_freq == 1

    def test_sequence_defaults(self):
        cfg = SequenceConfig()
        assert cfg.sequence_length == 16
        assert cfg.batch_size == 32

    def test_world_model_defaults(self):
        cfg = WorldModelConfig()
        assert cfg.enabled is False
        assert cfg.latent_dim == 128

    def test_feature_encoding_defaults(self):
        cfg = FeatureEncodingConfig()
        assert cfg.strategy == "one_hot"
        assert cfg.schema_path is None
        assert cfg.embedding_path is None

    def test_feature_encoding_requires_paths_for_embedding_lookup(self):
        with pytest.raises(ValueError, match="schema_path is required"):
            FeatureEncodingConfig(strategy="embedding_lookup", embedding_path="embeddings.npy")

        with pytest.raises(ValueError, match="embedding_path is required"):
            FeatureEncodingConfig(
                strategy="embedding_lookup",
                schema_path="configs/semantic_schema.toml",
            )


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_from_dict_basic(self):
        raw = {
            "game_manager_args": {"num_tiles": 10, "screen_size": 500, "vision_range": 2},
            "simulation_manager_args": {"number_of_environments": 50},
            "model_args": {"num_actions": 11},
            "model_config": {"n_steps": 2048},
            "total_timesteps": 50_000,
        }
        cfg = TrainingConfig.from_dict(raw)
        assert cfg.game_manager.num_tiles == 10
        assert cfg.simulation_manager.number_of_environments == 50
        assert cfg.model_args.num_actions == 11
        assert cfg.model_config.n_steps == 2048
        assert cfg.algorithm.algorithm == AlgorithmName.PPO
        assert cfg.algorithm.hyperparameters["n_steps"] == 2048
        assert cfg.total_timesteps == 50_000

    def test_from_dict_defaults(self):
        cfg = TrainingConfig.from_dict({})
        assert cfg.game_manager.num_tiles == 32
        assert cfg.algorithm.algorithm == AlgorithmName.PPO
        assert cfg.total_timesteps == 100_000

    def test_direct_constructor_syncs_algorithm_from_legacy_model_config(self):
        cfg = TrainingConfig(model_config=ModelConfig(n_steps=1024, learning_rate=1e-3))
        assert cfg.algorithm.algorithm == AlgorithmName.PPO
        assert cfg.algorithm.hyperparameters["n_steps"] == 1024
        assert cfg.algorithm.hyperparameters["learning_rate"] == pytest.approx(1e-3)

    def test_from_dict_explicit_algorithm(self):
        raw = {
            "algorithm": {
                "backend": "sb3",
                "algorithm": "DQN",
                "policy_name": "MultiInputPolicy",
                "hyperparameters": {"learning_rate": 1e-3, "buffer_size": 5000},
            }
        }
        cfg = TrainingConfig.from_dict(raw)
        assert cfg.algorithm.backend == RLBackend.SB3
        assert cfg.algorithm.algorithm == AlgorithmName.DQN
        assert cfg.algorithm.hyperparameters["buffer_size"] == 5000

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "game_manager" in d
        assert "simulation_manager" in d
        assert "algorithm" in d
        assert "evaluation" in d
        assert "replay" in d
        assert "sequence" in d
        assert "world_model" in d
        assert "feature_encoding" in d
        assert d["game_manager"]["num_tiles"] == 32
        assert d["feature_encoding"]["strategy"] == "one_hot"

    def test_from_dict_feature_encoding(self):
        cfg = TrainingConfig.from_dict(
            {
                "feature_encoding": {
                    "strategy": "embedding_lookup",
                    "schema_path": "configs/semantic_schema.toml",
                    "embedding_path": "configs/embeddings/example.npy",
                }
            }
        )

        assert cfg.feature_encoding.strategy == "embedding_lookup"
        assert cfg.feature_encoding.schema_path == "configs/semantic_schema.toml"
        assert cfg.feature_encoding.embedding_path == "configs/embeddings/example.npy"

    def test_from_dict_invalid_nested_values(self):
        raw = {"game_manager_args": {"num_tiles": 0}}
        with pytest.raises(ValueError):
            TrainingConfig.from_dict(raw)
