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
        self.penalty = -0.1
        self.max_not_improvement_routes = 5
        self.num_not_improvement_routes = 0
        self.best_route_energy = np.inf
        self.outposts_visited = set()
        self.num_actions = model_args['num_actions']
        self.map_pixel_size = game_manager_args['map_pixel_size']
        self.screen_size = game_manager_args['screen_size']
        self.kg_completeness = game_manager_args['kg_completeness']
        self.vision_range = game_manager_args['vision_range']
    
        self.simulation_manager = SimulationManager(
            game_manager_args,
            simulation_manager_args['number_of_environments'], 
            simulation_manager_args['number_of_curricula']
        )
        
        self.current_game_index = 0
        self.set_current_game_manager()

        num_nodes = self.kg.graph.num_nodes  # The constant number of nodes
        num_edges = self.kg.graph.num_edges  # The constant number of edges
        num_node_features = self.kg.graph.num_node_features  # Features per node
        num_edge_features = self.kg.graph.num_edge_features  # Features per edge

        vision_shape = (3, 2 * self.vision_range + 1, 2 * self.vision_range + 1)
        vision_space = spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8)

        # Node features: Assuming num_node_features is the number of features each node has
        node_feature_space = spaces.Box(low=0, high=7, shape=(num_node_features,), dtype=np.uint8)  # Ensure it's a tuple

        # Edge attributes: Assuming num_edge_features is the number of features each edge has
        edge_attr_space = spaces.Box(low=0, high=1000, shape=(num_edge_features,), dtype=np.uint8)  # Ensure it's a tuple

        # Graph space setup
        graph_space = spaces.Graph(node_space=node_feature_space, edge_space=edge_attr_space)

        # Combined observation space
        self.observation_space = spaces.Dict({
            'vision': vision_space,
            'graph': graph_space
        })

        print("custom_env.py, __init__")
        print(f"Vision shape: {vision_shape} (channels, height, width)")
        print(f"Graph X shape: (num_nodes, num_node_features) ({num_nodes}, {num_node_features})")
        print(f"Graph Edge Attr shape: (num_edges, num_edge_features) ({num_edges}, {num_edge_features})")

        self.agent_model = AgentModel(self.observation_space, num_node_features, self.num_actions)
        self.action_space = spaces.Discrete(self.num_actions)

    def set_current_game_manager(self):
        self.current_gm = self.simulation_manager.game_managers[self.current_game_index]
        self.current_gm.start_game()
        self.environment = self.current_gm.environment
        self.agent_controler = self.current_gm.agent_controler
        self.kg = self.current_gm.kg_class
        self.outpost_coords = self.environment.outpost_locations
        self.best_route_energy = 0

    def _calculate_reward(self):
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        self.current_reward = 0
        if agent_pos in self.outpost_coords and agent_pos not in self.outposts_visited:
            if not bool(self.outposts_visited):
                self.first_outpost_energy_tracker = self.agent_controler.energy_spent
            self.outposts_visited.add(agent_pos)
            self.current_reward = self.new_outpost_reward
            if len(self.outposts_visited) == self.environment.number_of_outposts:
                self.current_reward += self.completion_reward
                total_route_energy = self.agent_controler.energy_spent - self.first_outpost_energy_tracker
                self.current_gm.route_energy_list.append(total_route_energy)
                if total_route_energy < self.best_route_energy:
                    self.current_reward += self.route_improvement_reward
                    self.best_route_energy = total_route_energy
                    self.num_not_improvement_routes = 0
                else:
                    self.num_not_improvement_routes += 1
                if self.num_not_improvement_routes >= self.max_not_improvement_routes:
                    self.early_stop = True
                self.outposts_visited.clear()
            return self.current_reward
        return self.penalty

    def reset(self, index=None):
        if index is None:
            self.current_game_index = (self.current_game_index + 1) % len(self.simulation_manager.game_managers)
        else:
            self.current_game_index = index
        self.set_current_game_manager()
        self.best_route_energy = np.inf
        self.early_stop = False
        self.num_not_improvement_routes = 0
        observation = self._get_observation()
        return observation

    def step(self, action):
        self.agent_controler.agent_action(action)
        self.current_gm.rerender()
        reward = self._calculate_reward()
        done = not self.agent_controler.running or self.early_stop
        observation = self._get_observation()
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        # Ensure the observation is returned as a dictionary with specific keys
        return {'vision': self._get_vision(),
                 'graph': self.current_gm.kg_class.get_subgraph()
                }

    def _get_vision(self):
        # Calculate the size of the vision area in pixels
        vision_pixel_size = (2 * self.vision_range + 1) * self.current_gm.tile_size

        # Calculate the top-left corner of the vision rectangle in pixel coordinates
        vision_rect_x = (self.agent_controler.agent.grid_x - self.vision_range) * self.current_gm.tile_size
        vision_rect_y = (self.agent_controler.agent.grid_y - self.vision_range) * self.current_gm.tile_size

        # Define the vision rectangle
        vision_rect = pygame.Rect(vision_rect_x, vision_rect_y, vision_pixel_size, vision_pixel_size)

        # Ensure the vision rectangle is within the bounds of the game screen
        vision_rect.clamp_ip(self.current_gm.renderer.surface.get_rect())

        # Get a view of the vision rectangle
        vision_view = self.current_gm.renderer.surface.subsurface(vision_rect)

        # Convert the surface to an array
        vision_array = pygame.surfarray.array3d(vision_view)

        # The array needs to be transposed from (width, height, channels) to (channels, height, width)
        vision_array = np.transpose(vision_array, (2, 0, 1))

        # Prepare the padded array with the expected shape
        expected_shape = (3, vision_pixel_size, vision_pixel_size)  # RGB channels
        padded_array = np.zeros(expected_shape, dtype=np.uint8)

        # Only update the valid area, avoiding shape mismatch
        _, h, w = vision_array.shape  # Extract actual dimensions after transpose
        padded_array[:, :h, :w] = vision_array

        return padded_array

    def evaluate_policy(self, model, n_eval_episodes=10):
        all_episode_rewards = []

        for episode in range(n_eval_episodes):
            obs = self.reset()
            done = False
            episode_rewards = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)
                episode_rewards += reward

            all_episode_rewards.append(episode_rewards)

        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        return mean_reward, std_reward

    def close(self):
        self.current_gm.end_game()
        self.simulation_manager.save_data()