from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_METRIC_KEYS = (
    "performance",
    "game_manager_index",
    "best_route_energy",
    "curriculum_level",
    "target_route_energy",
    "best_efficiency",
    "improvement",
    "gap",
)


class TrainingMetrics:
    def __init__(self, num_actions):
        self.steps = []
        self.performances = []
        self.game_manager_indices = []
        self.best_route_energies = []
        self.curriculum_levels = []
        self.target_route_energies = []
        self.efficiency = []
        self.improvement = []
        self.gap = []
        self.num_actions = num_actions
        self.action_counts = [[] for _ in range(num_actions)]

    def record(self, step, metrics: dict):
        action_counts = metrics.get("action_counts", [0] * self.num_actions)
        padded_action_counts = [int(count) for count in action_counts[: self.num_actions]]
        if len(padded_action_counts) < self.num_actions:
            padded_action_counts.extend([0] * (self.num_actions - len(padded_action_counts)))

        self.steps.append(step)
        self.performances.append(float(metrics.get("performance", 0.0)))
        self.game_manager_indices.append(int(metrics.get("game_manager_index", 0)))
        self.best_route_energies.append(float(metrics.get("best_route_energy", 0.0)))
        self.curriculum_levels.append(int(metrics.get("curriculum_level", 0)))
        self.target_route_energies.append(float(metrics.get("target_route_energy", 0.0)))
        self.efficiency.append(float(metrics.get("best_efficiency", 0.0)))
        self.improvement.append(float(metrics.get("improvement", 0.0)))
        self.gap.append(float(metrics.get("gap", 0.0)))
        for index, count in enumerate(padded_action_counts):
            self.action_counts[index].append(count)

    def add_metric(
        self,
        step,
        performance,
        game_manager_index,
        best_route_energy,
        curriculum_level,
        target_route_energy,
        efficiency,
        improvement,
        gap,
        action_counts,
    ):
        self.record(
            step,
            {
                "performance": performance,
                "game_manager_index": game_manager_index,
                "best_route_energy": best_route_energy,
                "curriculum_level": curriculum_level,
                "target_route_energy": target_route_energy,
                "best_efficiency": efficiency,
                "improvement": improvement,
                "gap": gap,
                "action_counts": list(action_counts),
            },
        )

    def save_to_csv(self, filename):
        output_path = Path(filename)
        safe_stem = output_path.stem.replace(" ", "_").replace("-", "_")
        safe_stem = "".join(ch for ch in safe_stem if ch.isalnum() or ch == "_")
        output_path = output_path.with_name(f"{safe_stem}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "Step": self.steps,
                "Performance": self.performances,
                "Game Manager Index": self.game_manager_indices,
                "Best Route Energy": self.best_route_energies,
                "Curriculum Level": self.curriculum_levels,
                "Target Route Energy": self.target_route_energies,
                "Efficiency": self.efficiency,
                "Improvement": self.improvement,
                "Gap": self.gap,
            }
        )

        total_actions = np.sum(self.action_counts, axis=0)
        # Add columns for each action
        for i in range(self.num_actions):
            df[f"Action_{i}"] = np.divide(
                self.action_counts[i],
                total_actions,
                out=np.zeros(len(total_actions), dtype=float),
                where=total_actions != 0,
            )

        df.to_csv(output_path, index=False)
        return str(output_path)
