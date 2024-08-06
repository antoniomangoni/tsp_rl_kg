import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import deque
from game_manager import GameManager

class SimulationManager:
    def __init__(self, game_manager_args, number_of_environments=500, number_of_curricula=10, plot=False):
        self.game_managers = []
        self.curriculum_indices = []
        self.create_games(number_of_environments, game_manager_args, plot)
        number_of_curricula = min(max(1, number_of_curricula), number_of_environments // 2)
        self.curriculum_indices, step_size = self.get_curriculum(number_of_curricula)
        self.step_size = round(step_size, 2)
        energy_values = [gm.target_manager.target_route_energy for gm in self.game_managers]

        self.current_curriculum_index = 0
        self.min_episodes_per_curriculum = 10
        self.current_curriculum_episodes = 0
        self.performance_window = deque(maxlen=100)
        self.performance_threshold = 0.7  # 70% of max possible reward for current level
        self.max_performance = max(gm.target_manager.target_route_energy for gm in self.game_managers)

        # print(f"Curriculum step size: ~{self.step_size} energy units")
        if plot:
            self.create_plots(energy_values, self.curriculum_indices)

    def create_games(self, number_of_games, game_manager_args, plot):
        num_tiles = game_manager_args['num_tiles']
        screen_size = game_manager_args['screen_size']
        # kg_completeness = game_manager_args['kg_completeness']
        vision_range = game_manager_args['vision_range']

        for _ in range(number_of_games):
            game_manager = GameManager(num_tiles, screen_size, vision_range, plot)
            if len(game_manager.environment.outpost_locations) >= 3:
                self.insert_game_manager_sorted(game_manager)

    def insert_game_manager_sorted(self, game_manager):
        energy = game_manager.target_manager.target_route_energy
        index = next((i for i, gm in enumerate(self.game_managers) if energy < gm.target_manager.target_route_energy), len(self.game_managers))
        self.game_managers.insert(index, game_manager)

    def get_curriculum(self, number_of_curricula):
        energy_values = [gm.target_manager.target_route_energy for gm in self.game_managers]
        min_energy, max_energy = min(energy_values), max(energy_values)
        step_size = (max_energy - min_energy) / number_of_curricula
        simulation_indices = []
        for step in range(number_of_curricula + 1):
            target_energy = min_energy + step * step_size
            closest_index = np.abs(np.array(energy_values) - target_energy).argmin()
            if closest_index not in simulation_indices:
                simulation_indices.append(closest_index)
        return simulation_indices, step_size
    
    def should_advance_curriculum(self):
        if self.current_curriculum_episodes < self.min_episodes_per_curriculum:
            return False
        if not self.performance_window:
            return False
        avg_performance = sum(self.performance_window) / len(self.performance_window)
        current_max = self.game_managers[self.curriculum_indices[self.current_curriculum_index]].target_manager.target_route_energy
        return avg_performance > self.performance_threshold * current_max

    def add_episode_performance(self, performance):
        self.current_curriculum_episodes += 1
        self.performance_window.append(performance)

    def advance_curriculum(self):
        if self.current_curriculum_index < len(self.curriculum_indices) - 1:
            self.current_curriculum_index += 1
            self.current_curriculum_episodes = 0
            self.performance_window.clear()
            self.performance_threshold *= 0.995  # Decrease threshold for harder levels

    def get_current_game_manager(self):
        return self.game_managers[self.curriculum_indices[self.current_curriculum_index]]

    def plot_curriculum(self, x_values, y_values, indices, simulation_points, xlabel, ylabel, title = "Not named"):
        """
        Generalized function to plot curriculum data with optional step size annotation.

        Parameters:
            x_values (list of int): X-axis values for the plot.
            y_values (list of float): Y-axis values (e.g., energy required).
            indices (list of int): Indices of the selected simulation points.
            simulation_points (list of float): Energy values at the selected indices.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
            title (str): Title of the plot.
            step_size (float, optional): Step size value for annotation, if applicable.
        """
        plt.figure(figsize=(8, 4))
        ax1 = plt.gca()  # Get current axis
        ax1.plot(x_values, y_values, label='Energy for trade route', color='blue')
        ax1.scatter(indices, simulation_points, color='red', zorder=5, label='Selected Simulation Points', s=10)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        
        ax1.legend()

        # Creating a twin of the original y-axis to use for the step_size note
        ax2 = plt.twinx()
        ax2.set_ylabel(f'Curriculum step size: ~{self.step_size} energy units', fontsize=10, color='blue')
        ax2.tick_params(right=False, labelright=False)  # Turn off ticks and tick labels for the secondary axis

        plt.tight_layout()
        plt.savefig(f'Writing/{title}.png')
        plt.show()

    def create_plots(self, energy_values, curriculum_indices):
        """
        Creates two plots using the plot_curriculum function.

        Parameters:
            energy_values (list of float): Energy values for all environments.
            curriculum_indices (list of int): Indices of selected curriculum points.
            x_values (list of int): Number of the curriculum point for the x-axis.
        """
        simulation_points = [energy_values[i] for i in curriculum_indices]  # Energy at selected points

        # First plot with the environment index
        self.plot_curriculum(list(range(len(energy_values))), energy_values, curriculum_indices, simulation_points,
                        xlabel='Environment index', ylabel='Energy required for trade route',
                        title='Complete Energy Plot')

        # Second plot with the curriculum index
        # ascending list of length len(curriculum_indices) for x-values
        x_values = list(range(len(curriculum_indices)))  # Use a simple list of indices for the x-values
        self.plot_curriculum(x_values, simulation_points, x_values, simulation_points,
                             xlabel='Curriculum order index', ylabel='Energy required for trade route',
                             title='Energy Plot Indexed by Curriculum Order')
        
    def save_data(self, kg_completeness, static_file_path='Writing/static_data.csv', game_data_file_path='Writing/game_data.csv'):
        # Write static data to CSV
        with open(static_file_path, mode='w', newline='') as static_file:
            writer = csv.writer(static_file)
            writer.writerow(['Curriculum step size', f"{self.step_size} energy units"])
            writer.writerow(['Number of Environments', len(self.game_managers)])
            writer.writerow(['Number of Curricula', len(self.curriculum_indices)])
            writer.writerow(['KG Completeness', kg_completeness])
            writer.writerow(['Vision Range', self.game_managers[0].vision_range])
            writer.writerow(['Map Pixel Size', self.game_managers[0].num_tiles])
            writer.writerow(['Tile Size', self.game_managers[0].tile_size])

        # Write game data to CSV
        with open(game_data_file_path, mode='w', newline='') as game_data_file:
            writer = csv.writer(game_data_file)
            writer.writerow(['Curriculum Index', 'Target Route Energy', 'Achieved Route Energies'])

            for index in self.curriculum_indices:
                target_route_energy = self.game_managers[index].target_manager.target_route_energy
                achieved_route_energies = self.game_managers[index].route_energy_list
                writer.writerow([index, target_route_energy, ','.join(map(str, achieved_route_energies))])
