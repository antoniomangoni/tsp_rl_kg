import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from agent_model import AgentModel
from simulation_manager import SimulationManager

class CustomEnv(gym.Env):
    def __init__(self, vision_range=3, number_of_environments=500, number_of_curricula=10):
        super(CustomEnv, self).__init__()
        self.vision_range = vision_range

        # Initialize SimulationManager
        self.simulation_manager = SimulationManager(number_of_environments, number_of_curricula)
        self.current_game_index = 0
        self.set_current_game_manager()

        self.agent_model = AgentModel(self.vision_range, num_graph_features=16, num_actions=11)

        # Define observation and action spaces
        vision_shape = (3, 2 * vision_range + 1, 2 * vision_range + 1)
        graph_shape = (self.knowledge_graph.graph.x.shape[0], self.knowledge_graph.graph.x.shape[1])
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8),
            'graph': spaces.Box(low=-np.inf, high=np.inf, shape=graph_shape, dtype=np.float32)
        })

        self.action_space = spaces.Discrete(11)



    def set_current_game_manager(self):
        self.current_game = self.simulation_manager.game_managers[self.current_game_index]
        self.environment = self.current_game.environment
        self.agent = self.current_game.agent_controler
        self.knowledge_graph = self.current_game.kg_class.graph
        
    def reset(self):
        self.current_game_index += 1
        self.set_current_game_manager()
        self.game_manager.initialize_components()
        observation = self._get_observation()
        return observation

    def start_training(self):
        # Train agent model
        pass

    def step(self, action):
        # Perform action in the environment
        self.agent.agent_action(action)
        reward = self._calculate_reward()
        done = not self.agent.running
        observation = self._get_observation()
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        self.game_manager.render()

    def _get_observation(self):
        vision = self._get_vision()
        graph_data = self._get_graph_data()
        return {'vision': vision, 'graph': graph_data}

    def _get_vision(self):
        vision_range = self.vision_range
        agent_pos = (self.agent.agent.grid_x, self.agent.agent.grid_y)
        x, y = agent_pos
        vision = self.environment.get_vision(x, y, vision_range)
        return vision

    def _get_graph_data(self):
        graph_data = self.knowledge_graph.graph
        return graph_data

    def _calculate_reward(self):
        # Define your reward function here
        reward = 0
        return reward

# Example usage
# env = CustomEnv(vision_range=3)
# obs = env.reset()
# print(obs)
