from __future__ import annotations

import importlib.util
import traceback
from typing import Any

import numpy as np
from loguru import logger
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from tsp_rl_kg.config import AgentModelConfig, AlgorithmConfig, AlgorithmName, EvaluationConfig
from tsp_rl_kg.rl.agent_model import AgentModel
from tsp_rl_kg.rl.training.backends.base import MetricsDict
from tsp_rl_kg.rl.training.callbacks import CurriculumCallback
from tsp_rl_kg.rl.training.curriculum import CurriculumService
from tsp_rl_kg.rl.training.metrics import TrainingMetrics

SB3_ALGORITHMS = {
    AlgorithmName.PPO: PPO,
    AlgorithmName.DQN: DQN,
}


class SB3TrainingBackend:
    def __init__(
        self,
        env,
        eval_env,
        device,
        algorithm_config: AlgorithmConfig,
        agent_model_config: AgentModelConfig,
        evaluation_config: EvaluationConfig,
        metrics: TrainingMetrics,
    ):
        self.raw_env = env
        self.raw_eval_env = eval_env
        self.env = self._ensure_monitored_env(env)
        self.eval_env = self._ensure_monitored_env(eval_env)
        self.device = device
        self.algorithm_config = algorithm_config
        self.agent_model_config = agent_model_config
        self.evaluation_config = evaluation_config
        self.metrics = metrics
        self.model = None
        self.should_stop = False
        self.name = (
            f"{self.algorithm_config.backend.value}_"
            f"{self.algorithm_config.algorithm.value.lower()}"
        )

    def _progress_bar_enabled(self) -> bool:
        return (
            importlib.util.find_spec("tqdm") is not None
            and importlib.util.find_spec("rich") is not None
        )

    def _ensure_monitored_env(self, env):
        if isinstance(env, Monitor):
            return env
        return Monitor(env)

    def build(self) -> None:
        algorithm_class = SB3_ALGORITHMS.get(self.algorithm_config.algorithm)
        if algorithm_class is None:
            raise NotImplementedError(
                f"Unsupported SB3 algorithm '{self.algorithm_config.algorithm.value}'."
            )

        self.model = algorithm_class(
            self.algorithm_config.policy_name,
            self.env,
            policy_kwargs={
                "features_extractor_class": AgentModel,
                "features_extractor_kwargs": {
                    "features_dim": self.agent_model_config.features_dim,
                    "model_config": self.agent_model_config,
                },
            },
            **self.algorithm_config.hyperparameters,
            device=self.device,
            verbose=self.algorithm_config.verbose,
        )

    def train(self, total_timesteps: int, output_dir: str | None = None) -> None:
        if self.model is None:
            raise RuntimeError("SB3 backend model has not been built")

        curriculum_controller = CurriculumService(metrics_sink=self.metrics)
        curriculum_callback = CurriculumCallback(
            self.eval_env,
            curriculum_controller,
            num_actions=self.metrics.num_actions,
            step_interval=self._callback_interval(),
        )
        callbacks: list[Any] = [curriculum_callback]
        if output_dir is not None:
            callbacks.insert(
                0,
                EvalCallback(
                    self.eval_env,
                    best_model_save_path=output_dir,
                    log_path=output_dir,
                    eval_freq=self.evaluation_config.eval_freq,
                    deterministic=self.evaluation_config.deterministic,
                    render=self.evaluation_config.render,
                ),
            )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=(
                    self.algorithm_config.tensorboard_run_name
                    or self.algorithm_config.algorithm.value
                ),
                progress_bar=self._progress_bar_enabled(),
            )
        except Exception as error:
            logger.error(f"An error occurred during backend training: {error}")
            logger.error(traceback.format_exc())
            raise

        self.should_stop = curriculum_callback.should_stop

    def _callback_interval(self) -> int:
        if self.model is None:
            return 1

        n_steps = getattr(self.model, "n_steps", None)
        if n_steps is not None:
            return max(1, int(n_steps))

        train_freq = getattr(self.model, "train_freq", None)
        if hasattr(train_freq, "frequency"):
            return max(1, int(train_freq.frequency))
        if isinstance(train_freq, tuple) and train_freq:
            return max(1, int(train_freq[0]))
        if isinstance(train_freq, int):
            return max(1, train_freq)
        return 1

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("SB3 backend model has not been built")
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> str:
        if self.model is None:
            raise RuntimeError("SB3 backend model has not been built")
        self.model.save(path)
        return path

    def collect_metrics(self) -> MetricsDict:
        if self.model is None:
            return {}

        metrics: MetricsDict = {}
        ep_info_buffer = list(getattr(self.model, "ep_info_buffer", []))
        metrics["episode_info_buffer_size"] = len(ep_info_buffer)
        if ep_info_buffer:
            metrics["mean_reward"] = float(np.mean([ep_info["r"] for ep_info in ep_info_buffer]))
            metrics["mean_episode_length"] = float(
                np.mean([ep_info["l"] for ep_info in ep_info_buffer])
            )
        else:
            metrics["mean_reward"] = 0.0
            metrics["mean_episode_length"] = 0.0

        logger_values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
        for key in (
            "train/policy_loss",
            "train/value_loss",
            "train/loss",
            "rollout/exploration_rate",
        ):
            if key in logger_values:
                metrics[key] = float(logger_values[key])

        return metrics
