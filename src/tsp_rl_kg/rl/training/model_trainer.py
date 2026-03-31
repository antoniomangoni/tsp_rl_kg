import traceback

import mlflow
import numpy as np
from loguru import logger

from tsp_rl_kg.config import (
    AgentModelConfig,
    AlgorithmConfig,
    EvaluationConfig,
    ModelConfig,
    RLBackend,
)
from tsp_rl_kg.rl.training.backends import SB3TrainingBackend, TrainingBackend
from tsp_rl_kg.rl.training.metrics import TrainingMetrics


class ModelTrainer:
    def __init__(self, env, eval_env, device, evaluation_config: EvaluationConfig | None = None):
        self.env = env
        self.eval_env = eval_env
        self.device = device
        self.backend: TrainingBackend | None = None
        self.algorithm_config = AlgorithmConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.metrics = TrainingMetrics(env.action_space.n)

    def create_model(
        self,
        algorithm_config: AlgorithmConfig | ModelConfig | dict,
        agent_model_config: AgentModelConfig | None = None,
    ):
        self.algorithm_config = self._normalise_algorithm_config(algorithm_config)
        if agent_model_config is None:
            agent_model_config = AgentModelConfig()

        self.backend = self._create_backend(self.algorithm_config, agent_model_config)
        logger.info(f"Creating {self.backend.name} backend")
        self.backend.build()
        logger.info(f"{self.backend.name} backend created successfully")

    def _create_backend(
        self,
        algorithm_config: AlgorithmConfig,
        agent_model_config: AgentModelConfig,
    ) -> TrainingBackend:
        if algorithm_config.backend == RLBackend.SB3:
            return SB3TrainingBackend(
                env=self.env,
                eval_env=self.eval_env,
                device=self.device,
                algorithm_config=algorithm_config,
                agent_model_config=agent_model_config,
                evaluation_config=self.evaluation_config,
                metrics=self.metrics,
            )
        raise NotImplementedError(
            f"Unsupported backend '{algorithm_config.backend.value}'. "
            "Only the SB3 backend is wired so far."
        )

    def _normalise_algorithm_config(
        self,
        algorithm_config: AlgorithmConfig | ModelConfig | dict,
    ) -> AlgorithmConfig:
        if isinstance(algorithm_config, AlgorithmConfig):
            return algorithm_config
        if isinstance(algorithm_config, ModelConfig):
            return AlgorithmConfig.from_legacy_model_config(algorithm_config)
        if {"backend", "algorithm", "policy_name", "hyperparameters"}.intersection(
            algorithm_config.keys()
        ):
            return AlgorithmConfig(**algorithm_config)
        return AlgorithmConfig.from_legacy_model_config(algorithm_config)

    def _require_backend(self) -> TrainingBackend:
        if self.backend is None:
            raise RuntimeError("Training backend has not been created")
        return self.backend

    def train(self, total_timesteps, output_dir: str, timeout=3600):
        logger.info("Starting model training")
        backend = self._require_backend()

        try:
            backend.train(total_timesteps=total_timesteps, output_dir=output_dir)
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            logger.error(traceback.format_exc())

        if getattr(backend, "should_stop", False):
            logger.info("Training ended early due to early stop condition")
        else:
            logger.info("Model training completed")

    def log_training_stats(self):
        metrics = self._require_backend().collect_metrics()
        mean_reward = float(metrics.get("mean_reward", 0.0))
        mean_episode_length = float(metrics.get("mean_episode_length", 0.0))
        logger.info(f"Recent mean reward: {mean_reward:.2f}")
        logger.info(f"Recent mean episode length: {mean_episode_length:.2f}")
        logger.info(f"Episode info buffer size: {int(metrics.get('episode_info_buffer_size', 0))}")
        if "train/policy_loss" in metrics:
            logger.info(f"Recent policy loss: {float(metrics['train/policy_loss']):.5f}")
        if "train/value_loss" in metrics:
            logger.info(f"Recent value loss: {float(metrics['train/value_loss']):.5f}")
        if "train/loss" in metrics:
            logger.info(f"Recent training loss: {float(metrics['train/loss']):.5f}")
        if "rollout/exploration_rate" in metrics:
            logger.info(
                f"Recent exploration rate: {float(metrics['rollout/exploration_rate']):.5f}"
            )

    def calculate_mean_reward(self):
        metrics = self._require_backend().collect_metrics()
        return float(metrics.get("mean_reward", 0.0))

    def calculate_mean_episode_length(self):
        metrics = self._require_backend().collect_metrics()
        return float(metrics.get("mean_episode_length", 0.0))

    def get_model_artifact_name(self, experiment_name: str) -> str:
        return f"{self._require_backend().name}_custom_env_{experiment_name}.zip"

    def save_model(self, path):
        logger.info(f"Saving model to {path}")
        self._require_backend().save(path)
        logger.info("Model saved successfully")
        return path

    def evaluate_model(self, eval_env, n_eval_episodes=10):
        logger.info("Starting final model evaluation")
        backend = self._require_backend()

        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = backend.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "evaluation.mean_reward": float(mean_reward),
                    "evaluation.std_reward": float(std_reward),
                }
            )
        return mean_reward, std_reward
