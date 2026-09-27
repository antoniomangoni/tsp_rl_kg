import cProfile
import json
import os
import pstats
import random
from enum import Enum
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import torch
from loguru import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from tsp_rl_kg.config import TrainingConfig
from tsp_rl_kg.graph.feature_encoder import build_feature_encoder
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.rl.training.model_trainer import ModelTrainer


def make_worker(manager, worker):
    return Monitor(
        manager.make_env(worker=worker), info_keywords=("is_success", "laps", "efficiency")
    )


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
        self._feature_encoder_override = feature_encoder
        self.feature_encoder = feature_encoder
        self._closed = False

    def setup(self, config: TrainingConfig | dict, seed: int | None = None):
        try:
            self._setup(config, seed)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        for name in ("training_env", "env", "eval_env", "test_env"):
            env = getattr(self, name, None)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    logger.exception(f"Failed to close {name}")

    def _setup(self, config: TrainingConfig | dict, seed: int | None = None):
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
        agent_model_config.use_path_queries = config.ablation.use_path_queries

        self.feature_encoder = self._feature_encoder_override
        if self.feature_encoder is None:
            self.feature_encoder = build_feature_encoder(
                config.feature_encoding,
                grid_size=config.game_manager.num_tiles,
            )

        self.env_manager: EnvironmentManager = EnvironmentManager(
            config.game_manager,
            config.simulation_manager,
            config.model_args,
            self.feature_encoder,
            episode_config=config.episode,
            ablation_config=config.ablation,
            kg_completeness=self.current_kg_completeness,
            seed=seed or 0,
            reward_config=config.reward,
            curriculum_config=config.curriculum,
            evaluation_config=config.evaluation,
            gamma=float(config.algorithm.hyperparameters["gamma"]),
        )

        logger.info("Creating environment")
        self.env: CustomEnv = self.env_manager.make_env()
        self.eval_env = self.env_manager.make_eval_env(self.env.simulation_manager.game_managers)
        self.training_env = (
            self.env
            if config.n_envs == 1
            else SubprocVecEnv(
                [partial(make_worker, self.env_manager, i) for i in range(config.n_envs)],
                start_method="spawn",
            )
        )

        self.model_trainer = ModelTrainer(
            self.training_env,
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
        try:
            return self._run(experiment_name)
        finally:
            self.close()

    def _run(self, experiment_name):
        # Create a subdirectory for this experiment within the results directory
        experiment_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        self._log_run_context(experiment_name)
        Path(experiment_dir, "manifests.json").write_text(
            json.dumps(self.env_manager.manifest(), indent=2)
        )
        Path(experiment_dir, "resolved_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2)
        )

        profiler = cProfile.Profile()
        with profiler:
            self.model_trainer.train(
                total_timesteps=self.config.total_timesteps,
                output_dir=experiment_dir,
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
        self.test_env = self.env_manager.make_test_env()
        evaluations = {}
        for deterministic in (True, False):
            label = "deterministic" if deterministic else "stochastic"
            evaluations[label] = self.model_trainer.evaluator.evaluate(
                self.model_trainer.backend,
                self.test_env,
                self.config.evaluation.final_eval_episodes,
                deterministic=deterministic,
            )
        Path(experiment_dir, "final_evaluation.json").write_text(json.dumps(evaluations, indent=2))
        Path(experiment_dir, "manifests.json").write_text(
            json.dumps(self.env_manager.manifest(), indent=2)
        )
        mean_reward = evaluations["deterministic"]["mean_reward"]
        std_reward = evaluations["deterministic"]["std_reward"]

        for label, env in (("train", self.env), ("eval", self.eval_env)):
            env.simulation_manager.save_data(
                self.current_kg_completeness,
                static_file_path=os.path.join(experiment_dir, label, "static_data.csv"),
                game_data_file_path=os.path.join(experiment_dir, label, "game_data.csv"),
            )

        if mlflow.active_run():
            mlflow.log_artifacts(experiment_dir, artifact_path="training_outputs")
        logger.info("Training and evaluation completed.")

        return {
            "evaluation": evaluations,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            "metrics_file": metrics_file,
            "model_path": model_path,
            "stats_file": stats_file,
        }
