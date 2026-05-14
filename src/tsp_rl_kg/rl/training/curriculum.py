from __future__ import annotations

from typing import Any, Sequence

import mlflow
from loguru import logger

from tsp_rl_kg.rl.training.backends.base import CurriculumDecision, MetricsSink


class CurriculumService:
    """Backend-neutral curriculum and environment-metric coordinator."""

    def __init__(self, metrics_sink: MetricsSink):
        self.metrics_sink = metrics_sink

    def on_step(
        self,
        step: int,
        env: Any,
        action_counts: Sequence[int],
    ) -> CurriculumDecision:
        if env.early_stop:
            logger.info("Early stop condition met. Stopping training.")
            return CurriculumDecision(continue_training=False, should_stop=True)

        metrics = env.get_metrics()
        record = dict(metrics)
        record["action_counts"] = [int(count) for count in action_counts]
        self.metrics_sink.record(step, record)

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "training.performance": float(metrics.get("performance", 0.0)),
                    "training.game_manager_index": float(metrics.get("game_manager_index", 0)),
                    "training.best_route_energy": float(metrics.get("best_route_energy", 0.0)),
                    "training.curriculum_level": float(metrics.get("curriculum_level", 0)),
                    "training.target_route_energy": float(metrics.get("target_route_energy", 0.0)),
                    "training.best_efficiency": float(metrics.get("best_efficiency", 0.0)),
                    "training.improvement": float(metrics.get("improvement", 0.0)),
                    "training.gap": float(metrics.get("gap", 0.0)),
                },
                step=step,
            )

        if env.simulation_manager.should_advance_curriculum():
            new_index = env.simulation_manager.advance_curriculum()
            if new_index < 0:
                logger.info("All curricula completed. Stopping training.")
                return CurriculumDecision(continue_training=False, should_stop=True)

            logger.info(
                f"Advancing to curriculum level {env.simulation_manager.current_curriculum_index}"
            )
            return CurriculumDecision(
                continue_training=True,
                should_reset_environments=True,
            )

        return CurriculumDecision()
