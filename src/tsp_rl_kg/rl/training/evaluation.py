from __future__ import annotations

import mlflow
import numpy as np
from loguru import logger

from tsp_rl_kg.rl.training.backends.base import MetricsDict, TrainingBackend


class EpisodeEvaluator:
    """Backend-neutral evaluator using backend.predict against a Gymnasium env."""

    def evaluate(
        self,
        backend: TrainingBackend,
        env,
        n_episodes: int,
    ) -> MetricsDict:
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            while not done:
                action, _ = backend.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(episode_length)

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
        mean_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        std_episode_length = float(np.std(episode_lengths)) if episode_lengths else 0.0

        logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "evaluation.mean_reward": mean_reward,
                    "evaluation.std_reward": std_reward,
                    "evaluation.mean_episode_length": mean_episode_length,
                    "evaluation.std_episode_length": std_episode_length,
                }
            )

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_episode_length,
            "std_episode_length": std_episode_length,
        }
