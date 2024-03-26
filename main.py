from game_manager import GameManager
from simulation_manager import SimulationManager

if __name__ == '__main__':
    # game_manager = GameManager(map_size=32, tile_size=6)
    # game_manager.run()
    simulation_manager = SimulationManager(number_of_environments=1000, number_of_curricula=100)