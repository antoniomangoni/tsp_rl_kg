import mlflow
import numpy as np
import torch.nn as nn
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from torch_geometric.nn import GATConv

from tsp_rl_kg.rl.training.metrics import TrainingMetrics


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        metrics: TrainingMetrics,
        print_weight_stats_freq=1000,
        verbose=0,
    ):
        super(CurriculumCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.metrics = metrics
        self.should_stop = False
        self.print_weight_stats_freq = print_weight_stats_freq
        self.action_counts = np.zeros(metrics.num_actions, dtype=int)
        self.num_envs = getattr(
            eval_env.unwrapped, "num_envs", 1
        )  # Use unwrapped to access num_envs

    def _on_step(self) -> bool:
        # Update action counts based on the actions taken
        actions = self.locals["actions"]
        if isinstance(actions, np.ndarray):
            self.action_counts += np.bincount(actions, minlength=self.metrics.num_actions)
        else:
            self.action_counts[actions] += 1

        if self.n_calls % self.model.n_steps == 0:
            logger.info(f"Step {self.n_calls}")

            env = (
                self.training_env.envs[0]
                if hasattr(self.training_env, "envs")
                else self.training_env
            )
            unwrapped_env = env.unwrapped  # Get the unwrapped environment

            if unwrapped_env.early_stop:
                logger.info("Early stop condition met. Stopping training.")
                self.should_stop = True
                return False

            metrics = unwrapped_env.get_metrics()

            performance = metrics.get("performance", 0)
            game_manager_index = metrics.get("game_manager_index", 0)
            best_route_energy = metrics.get("best_route_energy", 0)
            curriculum_level = metrics.get("curriculum_level", 0)
            target_route_energy = metrics.get("target_route_energy", 0)
            efficiency = metrics.get("best_efficiency", 0)
            improvement = metrics.get("improvement", 0)
            gap = metrics.get("gap", 0)

            self.metrics.add_metric(
                self.n_calls,
                performance,
                game_manager_index,
                best_route_energy,
                curriculum_level,
                target_route_energy,
                efficiency,
                improvement,
                gap,
                self.action_counts,
            )

            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "training.performance": performance,
                        "training.game_manager_index": game_manager_index,
                        "training.best_route_energy": best_route_energy,
                        "training.curriculum_level": curriculum_level,
                        "training.target_route_energy": target_route_energy,
                        "training.best_efficiency": efficiency,
                        "training.improvement": improvement,
                        "training.gap": gap,
                    },
                    step=self.n_calls,
                )

            # Reset action counts
            self.action_counts = np.zeros(self.metrics.num_actions, dtype=int)

            # Check if curriculum should advance
            if unwrapped_env.simulation_manager.should_advance_curriculum():
                new_index = unwrapped_env.simulation_manager.advance_curriculum()
                if new_index < 0:
                    logger.info("All curricula completed. Stopping training.")
                    self.should_stop = True
                    return False

                logger.info(
                    f"Advancing to curriculum level "
                    f"{unwrapped_env.simulation_manager.current_curriculum_index}"
                )

                # Reset both training and eval environments
                self.training_env.reset()
                self.eval_env.reset()

            # Print weight statistics
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
            # print(f"{layer_name} weights - {weight_stats}")

        if hasattr(layer, "bias") and layer.bias is not None:
            bias = layer.bias.data
            bias_stats = self.compute_stats(bias)
            logger.info(f"{layer_name} bias - {bias_stats}")
            # print(f"{layer_name} bias - {bias_stats}")

    def compute_stats(self, tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }
