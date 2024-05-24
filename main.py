from game_manager import GameManager
from simulation_manager import SimulationManager
# from gym_manager import CustomEnv

if __name__ == '__main__':
    # game_manager = GameManager(map_pixel_size=16, screen_size=800, kg_completness=1)
    # game_manager.run()

    model_args = {
        'num_graph_features': 16,
        'num_actions': 11
    }

    simulation_manager_args = {
        'number_of_environments': 1000,
        'number_of_curricula': 100
    }

    game_manager_args = {
        'map_pixel_size': 16,
        'screen_size': 800,
        'kg_completness': 1,
        'agent_vision_range': 2
    }

    # gym_env = CustomEnv(game_manager_args, simulation_manager_args, model_args)

    simulation_manager = SimulationManager(game_manager_args, number_of_environments=10, number_of_curricula=3)
    # simulation_manager.game_managers[0].run()
    