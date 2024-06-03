# from gym_manager import CustomEnv

if __name__ == '__main__':

    model_args = {
        'num_actions': 11
    }

    simulation_manager_args = {
        'number_of_environments': 2,
        'number_of_curricula': 5
    }

    game_manager_args = {
        'map_pixel_size': 16,
        'screen_size': 800,
        'kg_completness': 1,
        'vision_range': 2
    }
    
    # gym_env = CustomEnv(game_manager_args, simulation_manager_args, model_args)

    

    # from game_manager import GameManager
    # game_manager = GameManager(game_manager_args['map_pixel_size'], game_manager_args['screen_size'],
    #                            game_manager_args['kg_completness'], game_manager_args['vision_range'])
    # game_manager.run()

    from simulation_manager import SimulationManager
    simulation_manager = SimulationManager(game_manager_args, simulation_manager_args['number_of_environments'], simulation_manager_args['number_of_curricula'])
    simulation_manager.game_managers[0].run()
