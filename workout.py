import gymnasium as gym

from simulation_manager import SimulationManager

class WorkingOut:
    def __init__(self):
        self.simulation_manager = SimulationManager(number_of_environments=200, number_of_curricula=50)

    def run(self):
        self.simulation_manager.run()