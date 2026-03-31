import traceback

import mlflow
import numpy as np
from loguru import logger
from stable_baselines3 import PPO

from tsp_rl_kg.config import AgentModelConfig, ModelConfig
from tsp_rl_kg.rl.agent_model import AgentModel
from tsp_rl_kg.rl.training.callbacks import CurriculumCallback
from tsp_rl_kg.rl.training.metrics import TrainingMetrics


class ModelTrainer:
    def __init__(self, env, eval_env, device):
        self.env = env
        self.eval_env = eval_env
        self.device = device
        self.rl_model = None
        self.metrics = TrainingMetrics(env.action_space.n)

    def create_model(
        self, model_config: ModelConfig | dict, agent_model_config: AgentModelConfig | None = None
    ):
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if agent_model_config is None:
            agent_model_config = AgentModelConfig()
        logger.info("Creating PPO model")
        self.rl_model = PPO(
            "MultiInputPolicy",
            self.env,
            policy_kwargs={
                "features_extractor_class": AgentModel,
                "features_extractor_kwargs": {
                    "features_dim": agent_model_config.features_dim,
                    "model_config": agent_model_config,
                },
            },
            **model_config.to_dict(),
            device=self.device,
            verbose=1,
        )
        logger.info("PPO model created successfully")

    def train(self, total_timesteps, eval_callback, timeout=3600):
        logger.info("Starting model training")

        curriculum_callback = CurriculumCallback(self.eval_env, self.metrics)

        try:
            self.rl_model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, curriculum_callback],
                reset_num_timesteps=False,
                tb_log_name="PPO",
                progress_bar=True,
            )
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            logger.error(traceback.format_exc())

        if curriculum_callback.should_stop:
            logger.info("Training ended early due to early stop condition")
        else:
            logger.info("Model training completed")

    def log_training_stats(self):
        mean_reward = self.calculate_mean_reward()
        mean_episode_length = self.calculate_mean_episode_length()
        logger.info(f"Recent mean reward: {mean_reward:.2f}")
        logger.info(f"Recent mean episode length: {mean_episode_length:.2f}")
        logger.info(f"Episode info buffer size: {len(self.rl_model.ep_info_buffer)}")
        if len(self.rl_model.ep_info_buffer) > 0:
            logger.info(f"Sample episode info: {self.rl_model.ep_info_buffer[-1]}")
        logger.info(
            f"Recent policy loss: {self.rl_model.logger.name_to_value['train/policy_loss']:.5f}"
        )
        logger.info(
            f"Recent value loss: {self.rl_model.logger.name_to_value['train/value_loss']:.5f}"
        )

    def calculate_mean_reward(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["r"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def calculate_mean_episode_length(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["l"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def save_model(self, path):
        logger.info(f"Saving model to {path}")
        self.rl_model.save(path)
        logger.info("Model saved successfully")
        return path

    def evaluate_model(self, eval_env, n_eval_episodes=10):
        logger.info("Starting final model evaluation")

        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
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
