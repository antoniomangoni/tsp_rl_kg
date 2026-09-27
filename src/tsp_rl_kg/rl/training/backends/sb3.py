from __future__ import annotations

import importlib.util
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

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


class SaveBestNormalization(BaseCallback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _on_step(self):
        self.model.get_vec_normalize_env().save(
            str(Path(self.output_dir) / "best_model.vecnormalize.pkl")
        )
        Path(self.output_dir, "best_model.schema.json").write_text(
            json.dumps({"observation_schema_version": 4})
        )
        return True


class FixedWorldEvalCallback(EvalCallback):
    """Restart the held-out schedule for every periodic comparison."""

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.env_method("begin_evaluation")
        result = super()._on_step()
        return result


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
        gamma = float(algorithm_config.hyperparameters["gamma"])
        self.env = self._normalized(env, gamma=gamma)
        self.eval_env = self._normalized(eval_env, gamma=gamma, evaluation=True)
        self.device = device
        self.algorithm_config = algorithm_config
        self.agent_model_config = agent_model_config
        self.evaluation_config = evaluation_config
        self.metrics = metrics
        self.model = None
        self.should_stop = False
        self._best_mean_reward = -np.inf
        self.name = (
            f"{self.algorithm_config.backend.value}_{self.algorithm_config.algorithm.value.lower()}"
        )

    def _progress_bar_enabled(self) -> bool:
        return (
            importlib.util.find_spec("tqdm") is not None
            and importlib.util.find_spec("rich") is not None
        )

    def _ensure_monitored_env(self, env):
        if isinstance(env, Monitor):
            return env
        keywords = ("is_success", "laps", "efficiency") if hasattr(env.unwrapped, "episode") else ()
        return Monitor(env, info_keywords=keywords)

    def _normalized(self, env, gamma, evaluation=False):
        if not isinstance(env, VecEnv):
            monitored = self._ensure_monitored_env(env)
            env = DummyVecEnv([lambda: monitored])
        return VecNormalize(
            env, norm_obs=False, norm_reward=not evaluation, training=not evaluation, gamma=gamma
        )

    def build(self) -> None:
        algorithm_class = SB3_ALGORITHMS.get(self.algorithm_config.algorithm)
        if algorithm_class is None:
            raise NotImplementedError(
                f"Unsupported SB3 algorithm '{self.algorithm_config.algorithm.value}'."
            )

        space = self.env.observation_space
        observation_bytes = sum(
            math.prod(v.shape) * np.dtype(v.dtype).itemsize for v in space.spaces.values()
        )
        hp = self.algorithm_config.hyperparameters
        buffer_bytes = (
            2 * observation_bytes * int(hp.get("buffer_size", 0))
            if self.algorithm_config.algorithm == AlgorithmName.DQN
            else observation_bytes * int(hp["n_steps"]) * self.env.num_envs
        )
        self.resource_estimates = dict(
            observation_bytes=observation_bytes, buffer_bytes=buffer_bytes
        )
        logger.info(f"Preallocation memory estimate: {self.resource_estimates}")
        # Match active feature-extractor capacity by widening real fusion layers.
        from tsp_rl_kg.rl.encoders import HybridEncoder

        if self.agent_model_config.parameter_budget is not None:
            with torch.random.fork_rng():
                config = self.agent_model_config
                config.fusion_dims = [1]
                probe = HybridEncoder(space, config.features_dim, config)
                small = sum(p.numel() for p in probe.parameters())
                slope = probe.fc[0].in_features + 1 + config.features_dim
                if small > config.parameter_budget * 1.1:
                    raise ValueError(
                        "Branch capacity exceeds parameter budget; reduce branch dimensions"
                    )
                config.fusion_dims = [max(1, round((config.parameter_budget - small) / slope) + 1)]
        policy_options = (
            {"ortho_init": False, "net_arch": dict(pi=[128], vf=[128])}
            if self.algorithm_config.algorithm == AlgorithmName.PPO
            else {"net_arch": [128]}
        )
        self.model = algorithm_class(
            self.algorithm_config.policy_name,
            self.env,
            policy_kwargs={
                **policy_options,
                "features_extractor_class": AgentModel,
                "features_extractor_kwargs": {
                    "features_dim": self.agent_model_config.features_dim,
                    "model_config": self.agent_model_config,
                },
            },
            **self.algorithm_config.hyperparameters,
            seed=getattr(self.raw_env, "_seed", None),
            device=self.device,
            verbose=self.algorithm_config.verbose,
        )

    def train(
        self, total_timesteps: int, output_dir: str | None = None, extra_callbacks=None
    ) -> None:
        if self.model is None:
            raise RuntimeError("SB3 backend model has not been built")

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            metadata = {
                **self.resource_estimates,
                "active_parameters": sum(
                    p.numel()
                    for group in self.model.policy.optimizer.param_groups
                    for p in group["params"]
                ),
                "fusion_dims": self.agent_model_config.fusion_dims,
            }
            Path(output_dir, "resources.json").write_text(json.dumps(metadata, indent=2))
        curriculum_controller = CurriculumService(metrics_sink=self.metrics)
        curriculum_callback = CurriculumCallback(
            self.eval_env,
            curriculum_controller,
            num_actions=self.metrics.num_actions,
            step_interval=self._callback_interval(),
        )
        callbacks: list[Any] = [curriculum_callback, *(extra_callbacks or [])]
        if output_dir is not None:
            evaluation_callback = FixedWorldEvalCallback(
                self.eval_env,
                best_model_save_path=output_dir,
                log_path=output_dir,
                eval_freq=max(1, self.evaluation_config.eval_freq // self.env.num_envs),
                callback_on_new_best=SaveBestNormalization(output_dir),
                n_eval_episodes=self.evaluation_config.n_eval_episodes,
                deterministic=self.evaluation_config.deterministic,
                render=self.evaluation_config.render,
            )
            evaluation_callback.best_mean_reward = self._best_mean_reward
            callbacks.insert(0, evaluation_callback)

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
            if output_dir is not None:
                self.metrics.save_to_csv(str(Path(output_dir) / "episodes.csv"))
            logger.error(f"An error occurred during backend training: {error}")
            logger.error(traceback.format_exc())
            raise

        if output_dir is not None:
            self._best_mean_reward = evaluation_callback.best_mean_reward
            self.metrics.save_to_csv(str(Path(output_dir) / "episodes.csv"))
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
        self.env.save(str(Path(path).with_suffix(".vecnormalize.pkl")))
        Path(path).with_suffix(".schema.json").write_text(
            json.dumps({"observation_schema_version": 4})
        )
        return path

    def load(self, path):
        metadata = Path(path).with_suffix(".schema.json")
        if (
            not metadata.exists()
            or json.loads(metadata.read_text()).get("observation_schema_version") != 4
        ):
            raise ValueError("Observation schema v4 required; fresh training is required")
        self.env = VecNormalize.load(
            str(Path(path).with_suffix(".vecnormalize.pkl")), self.env.venv
        )
        self.model = SB3_ALGORITHMS[self.algorithm_config.algorithm].load(
            path, env=self.env, device=self.device
        )

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
