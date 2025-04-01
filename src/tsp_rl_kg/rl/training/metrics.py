import numpy as np

class TrainingMetrics:
    def __init__(self, num_actions):
        self.steps = []
        self.performances = []
        self.game_manager_indices = []
        self.best_route_energies = []
        self.curriculum_steps = []
        self.target_route_energies = []
        self.efficiency = []
        self.improvement = []
        self.gap = []
        self.num_actions = num_actions
        self.action_counts = [[] for _ in range(num_actions)]

    def add_metric(self, step, performance, game_manager_index,
                   best_route_energy, curriculum_step, target_route_energy,
                   efficiency, improvement, gap, action_counts):
        self.steps.append(step)
        self.performances.append(performance)
        self.game_manager_indices.append(game_manager_index)
        self.best_route_energies.append(best_route_energy)
        self.curriculum_steps.append(curriculum_step)
        self.target_route_energies.append(target_route_energy)
        self.efficiency.append(efficiency)
        self.improvement.append(improvement)
        self.gap.append(gap)
        for i, count in enumerate(action_counts):
            self.action_counts[i].append(count)

    def save_to_csv(self, filename):
        # replace any special characters and spaces in the filename with underscores
        filename = filename.replace(' ', '_')
        filename = filename.replace('-', '_')
        filename = filename.replace('.', '_')
        filename = ''.join(e for e in filename if e.isalnum() or e == '_')
        # if file ends with _csv replace it with .csv
        if filename.endswith('_csv'):
            filename = filename[:-4] + '.csv'

        df = pd.DataFrame({
            'Step': self.steps,
            'Performance': self.performances,
            'Game Manager Index': self.game_manager_indices,
            'Best Route Energy': self.best_route_energies,
            'Curriculum Step': self.curriculum_steps,
            'Target Route Energy': self.target_route_energies,
            'Efficiency': self.efficiency,
            'Improvement': self.improvement,
            'Gap': self.gap
        })

        total_actions = np.sum(self.action_counts, axis=0)
        # Add columns for each action
        for i in range(self.num_actions):
            df[f'Action_{i}'] = self.action_counts[i] / total_actions

        df.to_csv(filename, index=False)