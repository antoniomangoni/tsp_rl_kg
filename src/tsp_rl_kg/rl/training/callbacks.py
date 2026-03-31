from typing import Any

import numpy as np
import torch.nn as nn
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from torch_geometric.nn import GATConv

from tsp_rl_kg.rl.training.backends.base import CurriculumController


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        controller: CurriculumController,
        num_actions: int,
        step_interval=1,
        print_weight_stats_freq=1000,
        verbose=0,
    ):
        super(CurriculumCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.controller = controller
        self.should_stop = False
        self.step_interval = max(1, step_interval)
        self.print_weight_stats_freq = print_weight_stats_freq
        self.action_counts = np.zeros(num_actions, dtype=int)
        self.num_envs = getattr(
            eval_env.unwrapped, "num_envs", 1
        )  # Use unwrapped to access num_envs

    def _on_step(self) -> bool:
        # Update action counts based on the actions taken
        actions = self.locals["actions"]
        if isinstance(actions, np.ndarray):
            self.action_counts += np.bincount(actions, minlength=len(self.action_counts))
        else:
            self.action_counts[actions] += 1

        if self.n_calls % self.step_interval == 0:
            logger.info(f"Step {self.n_calls}")

            training_env: Any = self.training_env
            env: Any = training_env.envs[0] if hasattr(training_env, "envs") else training_env
            unwrapped_env: Any = env.unwrapped

            decision = self.controller.on_step(
                self.n_calls,
                unwrapped_env,
                self.action_counts.tolist(),
            )

            self.action_counts = np.zeros(len(self.action_counts), dtype=int)

            if decision.should_reset_environments:
                self.training_env.reset()
                self.eval_env.reset()

            if decision.should_stop:
                self.should_stop = True
                return False

            if self.n_calls % self.print_weight_stats_freq == 0:
                self.print_weight_statistics()

        return True

    def print_weight_statistics(self):
        logger.info("Weight Statistics:")
        agent_model = self.model.policy.features_extractor

        # Vision Processor
        self.print_module_statistics(agent_model.vision_processor, "Vision Processor")

        # Graph Processor
        self.print_module_statistics(agent_model.graph_processor, "Graph Processor")

        # Combined Fully Connected Layers
        self.print_module_statistics(agent_model.fc, "Combined FC")

        # Dropout layer doesn't have learnable parameters, so we skip it

    def print_module_statistics(self, module, module_name):
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, (nn.Conv2d, nn.Linear, GATConv)):
                self.print_layer_statistics(sub_module, f"{module_name} - {name}")

    def print_layer_statistics(self, layer, layer_name):
        if hasattr(layer, "weight"):
            weights = layer.weight.data
            weight_stats = self.compute_stats(weights)
            logger.info(f"{layer_name} weights - {weight_stats}")

        if hasattr(layer, "bias") and layer.bias is not None:
            bias = layer.bias.data
            bias_stats = self.compute_stats(bias)
            logger.info(f"{layer_name} bias - {bias_stats}")

    def compute_stats(self, tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }
