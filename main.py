from game_manager import GameManager
from simulation_manager import SimulationManager

if __name__ == '__main__':
    game_manager = GameManager(map_pixel_size=7, screen_size=800, kg_completness=4)
    game_manager.run()
    # simulation_manager = SimulationManager(number_of_environments=200, number_of_curricula=50)
    