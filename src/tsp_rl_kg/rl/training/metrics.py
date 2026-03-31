from pathlib import Path

import numpy as np
import pandas as pd


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
        self.steps.append(step)
        self.performances.append(performance)
        self.game_manager_indices.append(game_manager_index)
        self.best_route_energies.append(best_route_energy)
        self.curriculum_levels.append(curriculum_level)
        self.target_route_energies.append(target_route_energy)
        self.efficiency.append(efficiency)
        self.improvement.append(improvement)
        self.gap.append(gap)
        for i, count in enumerate(action_counts):
            self.action_counts[i].append(count)

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
            df[f"Action_{i}"] = self.action_counts[i] / total_actions

        df.to_csv(output_path, index=False)
        return str(output_path)
