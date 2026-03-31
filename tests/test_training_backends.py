"""Tests for the SB3 backend adapter and generic trainer-facing model setup."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from tsp_rl_kg.config import AgentModelConfig, AlgorithmConfig, AlgorithmName, EvaluationConfig
from tsp_rl_kg.rl.training.backends.sb3 import SB3TrainingBackend
from tsp_rl_kg.rl.training.metrics import TrainingMetrics
from tsp_rl_kg.rl.training.model_trainer import ModelTrainer


class DummyDictEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "vision": gym.spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
                "node_features": gym.spaces.Box(
                    low=-1.0,
                    high=1e4,
                    shape=(8, 4),
                    dtype=np.float32,
                ),
                "edge_attr": gym.spaces.Box(
                    low=-1.0,
                    high=1e4,
                    shape=(12, 2),
                    dtype=np.float32,
                ),
                "edge_index": gym.spaces.Box(
                    low=0,
                    high=7,
                    shape=(2, 12),
                    dtype=np.int64,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(3)
        self._step_count = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "vision": np.zeros((3, 32, 32), dtype=np.float32),
            "node_features": np.zeros((8, 4), dtype=np.float32),
            "edge_attr": np.zeros((12, 2), dtype=np.float32),
            "edge_index": np.zeros((2, 12), dtype=np.int64),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self._observation(), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= 2
        return self._observation(), float(action), terminated, False, {}


def test_sb3_backend_builds_ppo_and_wraps_envs():
    backend = SB3TrainingBackend(
        env=DummyDictEnv(),
        eval_env=DummyDictEnv(),
        device="cpu",
        algorithm_config=AlgorithmConfig(
            algorithm=AlgorithmName.PPO,
            hyperparameters={
                "n_steps": 8,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "gamma": 0.99,
            },
        ),
        agent_model_config=AgentModelConfig(features_dim=32),
        evaluation_config=EvaluationConfig(eval_freq=2, n_eval_episodes=1),
        metrics=TrainingMetrics(3),
    )

    backend.build()

    assert backend.name == "sb3_ppo"
    assert backend.model.__class__.__name__ == "PPO"
    assert isinstance(backend.env, Monitor)
    assert isinstance(backend.eval_env, Monitor)


def test_sb3_backend_builds_dqn_from_algorithm_config():
    backend = SB3TrainingBackend(
        env=DummyDictEnv(),
        eval_env=DummyDictEnv(),
        device="cpu",
        algorithm_config=AlgorithmConfig(
            algorithm=AlgorithmName.DQN,
            hyperparameters={
                "learning_rate": 1e-3,
                "buffer_size": 32,
                "learning_starts": 0,
                "batch_size": 4,
                "train_freq": 1,
                "gamma": 0.99,
            },
        ),
        agent_model_config=AgentModelConfig(features_dim=32),
        evaluation_config=EvaluationConfig(eval_freq=2, n_eval_episodes=1),
        metrics=TrainingMetrics(3),
    )

    backend.build()

    assert backend.name == "sb3_dqn"
    assert backend.model.__class__.__name__ == "DQN"


def test_model_trainer_uses_backend_artifact_name():
    trainer = ModelTrainer(
        env=DummyDictEnv(),
        eval_env=DummyDictEnv(),
        device="cpu",
        evaluation_config=EvaluationConfig(eval_freq=2, n_eval_episodes=1),
    )
    trainer.create_model(
        AlgorithmConfig(
            algorithm=AlgorithmName.DQN,
            hyperparameters={
                "learning_rate": 1e-3,
                "buffer_size": 32,
                "learning_starts": 0,
                "batch_size": 4,
                "train_freq": 1,
                "gamma": 0.99,
            },
        ),
        AgentModelConfig(features_dim=32),
    )

    assert trainer.get_model_artifact_name("demo") == "sb3_dqn_custom_env_demo.zip"
