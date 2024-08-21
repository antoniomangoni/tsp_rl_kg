import gymnasium as gym
from stable_baselines3 import PPO

from custom_env import CustomEnv
from simulation_manager import SimulationManager

if __name__ == '__main__':

    model_args = {
        'num_actions': 11
    }

    simulation_manager_args = {
        'number_of_environments': 2000,
        'number_of_curricula': 50
    }

    game_manager_args = {
        'num_tiles': 4,
        'screen_size': 800,
        'kg_completeness': 0.1,
        'vision_range': 2
    }

    # env = CustomEnv(game_manager_args, simulation_manager_args, model_args)
    
    # # Instantiate the RL model (e.g., PPO)
    # model = PPO('CnnPolicy', env, verbose=1)

    # gym_env = CustomEnv(game_manager_args, simulation_manager_args, model_args)

    

    from game_manager import GameManager
    game_manager = GameManager(game_manager_args['num_tiles'], game_manager_args['screen_size'],
                               game_manager_args['kg_completeness'], game_manager_args['vision_range'])
                               
    game_manager.run()

    # from simulation_manager import SimulationManager
    # simulation_manager = SimulationManager(game_manager_args,
    #                                        simulation_manager_args['number_of_environments'],
    #                                        simulation_manager_args['number_of_curricula'])
    
    # simulation_manager.game_managers[0].run()
