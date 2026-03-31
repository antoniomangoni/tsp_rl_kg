"""Tests for the backend-neutral episode evaluator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from tsp_rl_kg.rl.training.evaluation import EpisodeEvaluator


class DummyBackend:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, observation, deterministic: bool = True):
        self.predict_calls += 1
        return 1, None


class DummyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count >= 3
        return np.zeros((1,), dtype=np.float32), 1.0, terminated, False, {}


def test_episode_evaluator_uses_backend_predict_and_aggregates_rewards():
    evaluator = EpisodeEvaluator()
    backend = DummyBackend()
    env = DummyEvalEnv()

    result = evaluator.evaluate(backend, env, n_episodes=2)

    assert result["mean_reward"] == 3.0
    assert result["std_reward"] == 0.0
    assert result["mean_episode_length"] == 3.0
    assert backend.predict_calls == 6
