import gymnasium as gym
from stable_baselines3 import PPO

from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.simulation_manager import SimulationManager
from tsp_rl_kg.game_world.game_manager import GameManager

import numpy as np

if __name__ == '__main__':

    model_args = {
        'num_actions': 11
    }

    simulation_manager_args = {
        'number_of_environments': 2,
        'number_of_curricula': 3
    }

    game_manager_args = {
        'num_tiles': 4,
        'screen_size': 800,
        'kg_completeness': 1,
        'vision_range': 2
    }

    # 0: RL, 1: Simulation, 2: just GameManager
    run_type = 2

    if run_type == 0:
        env = CustomEnv(game_manager_args, simulation_manager_args, model_args, word_embedding_dim=512)
        
        # Instantiate the RL model (e.g., PPO)
        model = PPO('CnnPolicy', env, verbose=1)

    elif run_type == 1:

        game_manager = GameManager(game_manager_args['num_tiles'], game_manager_args['screen_size'],
                                game_manager_args['kg_completeness'], game_manager_args['vision_range'])
                               
        game_manager.run()

    elif run_type == 2:
        simulation_manager = SimulationManager(game_manager_args,
                                            simulation_manager_args['number_of_environments'],
                                            simulation_manager_args['number_of_curricula'])
        
        # simulation_manager.game_managers[0].run()

        game_world_array = np.zeros((simulation_manager_args['number_of_environments'],
                                    2,  # terrain and entity
                                    game_manager_args['num_tiles'],
                                    game_manager_args['num_tiles']), dtype=np.uint8)

        for i, game_manager in enumerate(simulation_manager.game_managers):
            print(game_manager.environment.terrain_index_grid)
            print(game_manager.environment.entity_index_grid)
            
            # Assign each grid to the appropriate slice
            game_world_array[i, 0] = game_manager.environment.terrain_index_grid
            game_world_array[i, 1] = game_manager.environment.entity_index_grid

        print(game_world_array)

        unique_terrain_objects = np.unique(game_world_array[:, 0])

        # Save the game world array to a file
        np.save('game_world.npy', game_world_array)

        terrain_dict = {
            0: 'Deep Water',
            1: 'Water',
            2: 'Grass',
            3: 'Hill',
            4: 'Mountain',
            5: 'Snow'
        }

        entity_dict = {
            0: 'None',
            1: 'Fish',
            2: 'Tree',
            3: 'Mossy Rock',
            4: 'Snowy Rock',
            5: 'Outpost',
            6: 'Wood Path',
            7: 'Player'
        }

        import json
        # Save the dictionaries to JSON files
        with open('terrain_dict.json', 'w') as f:
            json.dump(terrain_dict, f)
        with open('entity_dict.json', 'w') as f:
            json.dump(entity_dict, f)