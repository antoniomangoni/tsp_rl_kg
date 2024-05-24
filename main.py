from game_manager import GameManager
from simulation_manager import SimulationManager
# from gym_manager import CustomEnv

if __name__ == '__main__':

    model_args = {
        'num_graph_features': 16,
        'num_actions': 11
    }

    simulation_manager_args = {
        'number_of_environments': 10,
        'number_of_curricula': 5
    }

    game_manager_args = {
        'map_pixel_size': 16,
        'screen_size': 800,
        'kg_completness': 1,
        'agent_vision_range': 2
    }
    game_manager = GameManager(game_manager_args['map_pixel_size'], game_manager_args['screen_size'],
                               game_manager_args['kg_completness'], game_manager_args['agent_vision_range'])
    game_manager.run()

    # simulation_manager = SimulationManager(game_manager_args, simulation_manager_args['number_of_environments'], simulation_manager_args['number_of_curricula'])
    # simulation_manager.game_managers[0].run()
    
    # gym_env = CustomEnv(game_manager_args, simulation_manager_args, model_args)