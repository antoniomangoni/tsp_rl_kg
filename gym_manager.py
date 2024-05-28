import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pygame

from agent_model import AgentModel
from simulation_manager import SimulationManager

class CustomEnv(gym.Env):
    def __init__(self, game_manager_args, simulation_manager_args, model_args):
        super(CustomEnv, self).__init__()

        self.num_graph_features = model_args['num_graph_features']
        self.num_actions = model_args['num_actions']

        self.map_pixel_size = game_manager_args['map_pixel_size']
        self.screen_size = game_manager_args['screen_size']
        self.kg_completeness = game_manager_args['kg_completeness']
        self.vision_range = game_manager_args['agent_vision_range']
    
        # Initialize SimulationManager
        self.simulation_manager = SimulationManager(
            simulation_manager_args['number_of_environments'], 
            simulation_manager_args['number_of_curricula'],
            game_manager_args
        )
        
        self.current_game_index = 0
        self.set_current_game_manager()

        self.agent_model = AgentModel(self.vision_range, num_graph_features=self.num_graph_features, num_actions=self.num_actions)

        # Define observation and action spaces
        vision_shape = (3, 2 * self.vision_range + 1, 2 * self.vision_range + 1)
        graph_shape = self.get_graph_shape()
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8),
            'graph': spaces.Box(low=-np.inf, high=np.inf, shape=graph_shape, dtype=np.float32)
        })

        self.action_space = spaces.Discrete(self.num_actions)

    def get_graph_shape(self):
        sub_graph = self.knowledge_graph.get_subgraph()
        return sub_graph.x.shape

    def set_current_game_manager(self):
        self.current_game_manager = self.simulation_manager.game_managers[self.current_game_index]
        self.current_game_manager.start_game()

        self.environment = self.current_game_manager.environment
        self.agent_controler = self.current_game_manager.agent_controler
        self.knowledge_graph = self.current_game_manager.kg_class

    def reset(self):
        self.current_game_index = (self.current_game_index + 1) % len(self.simulation_manager.game_managers)
        self.set_current_game_manager()
        observation = self._get_observation()
        return observation

    def step(self, action):
        # Perform action in the environment
        self.agent_controler.agent_action(action)
        reward = self._calculate_reward()
        done = not self.agent_controler.running
        observation = self._get_observation()
        info = {}

        return observation, reward, done, info
    
    def progress_step(self):
        self.current_game_manager.game_step()

    def render(self, mode='human'):
        self.current_game_manager.rerender()

    def _get_observation(self):
        vision = self._get_vision()
        graph_data = self._get_graph_data()
        return {'vision': vision, 'graph': graph_data}

    def _get_vision(self):
        vision_range = self.vision_range
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        x, y = agent_pos

        # Calculate the region to capture
        tile_size = self.current_game_manager.tile_size
        vision_pixel_size = (2 * vision_range + 1) * tile_size
        agent_pixel_x = x * tile_size
        agent_pixel_y = y * tile_size
        vision_rect = pygame.Rect(agent_pixel_x - vision_range * tile_size, agent_pixel_y - vision_range * tile_size, vision_pixel_size, vision_pixel_size)

        # Capture the vision region
        vision_surface = self.current_game_manager.renderer.surface.subsurface(vision_rect).copy()
        vision_array = pygame.surfarray.array3d(vision_surface)

        # Convert from (width, height, channels) to (channels, height, width) for PyTorch
        vision_array = np.transpose(vision_array, (2, 1, 0))
        return vision_array

    def _get_graph_data(self):
        return self.knowledge_graph.get_subgraph()

    def _calculate_reward(self):
        # Define your reward function here
        reward = 0
        return reward
