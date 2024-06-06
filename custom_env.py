import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from agent_model import AgentModel
from simulation_manager import SimulationManager

class CustomEnv(gym.Env):
    def __init__(self, game_manager_args, simulation_manager_args, model_args):
        super(CustomEnv, self).__init__()

        self.new_outpost_reward = 10
        self.completion_reward = 100
        self.route_improvement_reward = 200
        self.current_reward = 0
        self.penalty = -1

        self.max_not_improvement_routes = 5
        self.num_not_improvement_routes = 0
        self.best_route_energy = np.inf
        self.outposts_visited = set()

        self.num_actions = model_args['num_actions']

        self.map_pixel_size = game_manager_args['map_pixel_size']
        self.screen_size = game_manager_args['screen_size']
        self.kg_completeness = game_manager_args['kg_completeness']
        self.vision_range = game_manager_args['agent_vision_range']
    
        # Initialize SimulationManager
        self.simulation_manager = SimulationManager(
            game_manager_args,
            simulation_manager_args['number_of_environments'], 
            simulation_manager_args['number_of_curricula']
        )
        
        self.current_game_index = 0
        self.set_current_game_manager()

        # Define observation and action spaces
        vision_shape = (3, 2 * self.vision_range + 1, 2 * self.vision_range + 1)
        graph_shape = self.get_graph_shape()
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8),
            'graph': spaces.Box(low=-np.inf, high=np.inf, shape=graph_shape, dtype=np.float32)
        })

        self.agent_model = AgentModel(vision_shape, graph_shape, self.num_actions,
                                      self.kg.num_node_features, self.kg.num_edge_features)

        self.action_space = spaces.Discrete(self.num_actions)

    def get_graph_shape(self):
        sub_graph = self.kg.get_subgraph()
        return sub_graph.x.shape

    def set_current_game_manager(self):
        self.current_gm = self.simulation_manager.game_managers[self.current_game_index]
        self.current_gm.start_game()

        self.environment = self.current_gm.environment
        self.agent_controler = self.current_gm.agent_controler
        self.kg = self.current_gm.kg_class
        self.outpost_coords = self.environment.outpost_locations
        self.best_route_energy = 0

    def _calculate_reward(self):
        """
        Calculate the reward for the agent based on visiting new outposts and the total route energy.
        """
        # Get the agent's current position
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        self.current_reward = 0
        if agent_pos in self.outpost_coords and agent_pos not in self.outposts_visited:
            if not bool(self.outposts_visited):
                self.first_outpost_energy_tracker = self.agent_controler.energy_spent

            self.outposts_visited.add(agent_pos)
            self.current_reward = self.new_outpost_reward

            # Check if all outposts have been visited
            if len(self.outposts_visited) == self.environment.number_of_outposts:
                self.current_reward += self.completion_reward
                # Calculate the total route energy
                total_route_energy = self.agent_controler.energy_spent - self.first_outpost_energy_tracker
                self.current_gm.route_energy_list.append(total_route_energy)

                # Check for early stopping condition (route energy improvement)
                if total_route_energy < self.best_route_energy:
                    self.current_reward += self.route_improvement_reward
                    self.best_route_energy = total_route_energy
                    self.num_not_improvement_routes = 0
                else:
                    self.num_not_improvement_routes += 1
                if self.num_not_improvement_routes >= self.max_not_improvement_routes:
                    self.early_stop = True

                # Reset for the next round
                self.outposts_visited.clear()

            return self.current_reward

        return self.penalty

    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        self.current_game_index = (self.current_game_index + 1) % len(self.simulation_manager.game_managers)
        self.set_current_game_manager()
        self.best_route_energy = np.inf
        self.early_stop = False
        self.num_not_improvement_routes = 0
        observation = self._get_observation()
        return observation

    def step(self, action):
        """
        Perform the given action in the environment and return the result.
        """
        # Perform action in the environment
        self.agent_controler.agent_action(action)
        reward = self._calculate_reward()
        done = not self.agent_controler.running or self.early_stop
        observation = self._get_observation()
        info = {}

        return observation, reward, done, info
    
    def progress_step(self):
        self.current_gm.game_step()

    def _get_observation(self):
        vision = self._get_vision()
        graph_data = self._get_graph_data()
        return {'vision': vision, 'graph': graph_data}

    def _get_vision(self):
        vision_range = self.vision_range
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        x, y = agent_pos

        # Calculate the region to capture
        tile_size = self.current_gm.tile_size
        vision_pixel_size = (2 * vision_range + 1) * tile_size
        agent_pixel_x = x * tile_size
        agent_pixel_y = y * tile_size
        vision_rect = pygame.Rect(agent_pixel_x - vision_range * tile_size, agent_pixel_y - vision_range * tile_size, vision_pixel_size, vision_pixel_size)

        # Capture the vision region
        vision_surface = self.current_gm.renderer.surface.subsurface(vision_rect).copy()
        vision_array = pygame.surfarray.array3d(vision_surface)

        # Convert from (width, height, channels) to (channels, height, width) for PyTorch
        vision_array = np.transpose(vision_array, (2, 1, 0))
        return vision_array

    def _get_graph_data(self):
        return self.kg.get_subgraph()
