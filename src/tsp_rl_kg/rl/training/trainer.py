import cProfile
import json
import os
import pstats
import random
from enum import Enum

import mlflow
import numpy as np
import torch
from loguru import logger

from tsp_rl_kg.config import TrainingConfig
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.rl.training.model_trainer import ModelTrainer


class Trainer:
    def __init__(
        self,
        current_kg_completeness,
        *,
        results_dir: str,
        feature_encoder=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.current_kg_completeness = current_kg_completeness
        self.results_dir = results_dir
        self.feature_encoder = feature_encoder

    def setup(self, config: TrainingConfig | dict, seed: int | None = None):
        if isinstance(config, dict):
            config = TrainingConfig.from_dict(config)
        self.config = config

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Propagate ablation flags to AgentModelConfig
        agent_model_config = config.agent_model
        agent_model_config.disable_vision = config.ablation.disable_vision
        agent_model_config.disable_graph = config.ablation.disable_graph

        self.env_manager: EnvironmentManager = EnvironmentManager(
            config.game_manager,
            config.simulation_manager,
            config.model_args,
            self.feature_encoder,
            episode_config=config.episode,
            ablation_config=config.ablation,
        )

        logger.info("Creating environment")
        self.env: CustomEnv = self.env_manager.make_env()
        self.env.unwrapped.simulation_manager.min_episodes_per_curriculum = (
            config.curriculum.min_episodes_per_curriculum
        )
        self.env.unwrapped.simulation_manager.performance_threshold = (
            config.curriculum.performance_threshold
        )
        logger.info("Environment created successfully")

        logger.info("Creating evaluation environment")
        train_game_managers = self.env.unwrapped.simulation_manager.game_managers
        self.eval_env: CustomEnv = self.env_manager.make_eval_env(train_game_managers)
        self.eval_env.unwrapped.simulation_manager.min_episodes_per_curriculum = (
            config.curriculum.min_episodes_per_curriculum
        )
        self.eval_env.unwrapped.simulation_manager.performance_threshold = (
            config.curriculum.performance_threshold
        )
        logger.info("Evaluation environment created successfully")

        self.model_trainer = ModelTrainer(
            self.env,
            self.eval_env,
            self.device,
            evaluation_config=config.evaluation,
        )
        self.model_trainer.create_model(config.algorithm, config.agent_model)

    def _flatten_mlflow_params(
        self,
        prefix: str,
        value,
        output: dict[str, str | int | float | bool],
    ):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._flatten_mlflow_params(nested_prefix, nested_value, output)
            return

        if isinstance(value, Enum):
            output[prefix] = value.value
        elif isinstance(value, list):
            output[prefix] = json.dumps(
                [item.value if isinstance(item, Enum) else item for item in value],
                sort_keys=True,
            )
        elif isinstance(value, (str, int, float, bool)):
            output[prefix] = value
        elif value is not None:
            output[prefix] = str(value)

    def _log_run_context(self, experiment_name: str) -> None:
        if not mlflow.active_run():
            return

        params: dict[str, str | int | float | bool] = {
            "training.device": str(self.device),
            "training.experiment_name": experiment_name,
        }
        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        self._flatten_mlflow_params("config", config_dict, params)
        mlflow.log_params(params)

    def run(self, experiment_name):
        # Create a subdirectory for this experiment within the results directory
        experiment_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        self._log_run_context(experiment_name)

        profiler = cProfile.Profile()
        profiler.enable()

        self.model_trainer.train(
            total_timesteps=self.config.total_timesteps,
            output_dir=experiment_dir,
            timeout=3600,
        )

        # Save metrics
        metrics_file = os.path.join(experiment_dir, f"{experiment_name}_metrics.csv")
        metrics_file = self.model_trainer.metrics.save_to_csv(metrics_file)

        profiler.disable()
        stats_file = os.path.join(experiment_dir, "profile_stats.txt")
        with open(stats_file, "w", encoding="utf-8") as stats_stream:
            pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative").print_stats()
        logger.info(f"Profiling stats saved to {stats_file}")

        model_path = os.path.join(
            experiment_dir,
            self.model_trainer.get_model_artifact_name(experiment_name),
        )
        model_path = self.model_trainer.save_model(model_path)
        mean_reward, std_reward = self.model_trainer.evaluate_model(
            self.eval_env,
            n_eval_episodes=self.config.evaluation.n_eval_episodes,
        )

        if mlflow.active_run():
            mlflow.log_artifacts(experiment_dir, artifact_path="training_outputs")

        logger.info("Closing environments")
        self.env_manager.set_kg_completeness(self.env, self.current_kg_completeness)
        self.env.close()
        self.env_manager.set_kg_completeness(self.eval_env, self.current_kg_completeness)
        self.eval_env.close()
        logger.info("Environments closed successfully")

        logger.info("Training and evaluation completed.")

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            "metrics_file": metrics_file,
            "model_path": model_path,
            "stats_file": stats_file,
        }
