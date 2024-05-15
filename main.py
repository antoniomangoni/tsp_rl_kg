from game_manager import GameManager
from simulation_manager import SimulationManager

if __name__ == '__main__':
    # game_manager = GameManager(map_pixel_size=32, screen_size=800, kg_completness=1)
    # game_manager.run()
    simulation_manager = SimulationManager(number_of_environments=30, number_of_curricula=10)
    # simulation_manager.game_managers[0].run()
    