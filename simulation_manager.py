import numpy as np
import matplotlib.pyplot as plt
from game_manager import GameManager

class SimulationManager:
    def __init__(self, number_of_environments=500, number_of_curricula=60):
        self.game_managers = []
        self.curriculum_indices = []
        self.create_games(number_of_environments)
        self.curriculum_indices = self.get_curriculum(number_of_curricula)
        self.plot_curriculum()

    def create_games(self, number_of_games):
        for _ in range(number_of_games):
            game_manager = GameManager(map_size=32, tile_size=6)
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
        return simulation_indices

    def plot_curriculum(self, number_of_curricula=60):
        energy_values = [gm.target_manager.target_route_energy for gm in self.game_managers]
        simulation_points = [energy_values[i] for i in self.curriculum_indices]
        plt.plot(energy_values, label='Energy for trade route')
        plt.scatter(self.curriculum_indices, simulation_points, color='red', zorder=5, label='Simulation Points', s=10)
        plt.xlabel('Environment index')
        plt.ylabel('Energy required for trade route')
        plt.title('Energy required for trade route for each environment')
        plt.legend()
        plt.show()
