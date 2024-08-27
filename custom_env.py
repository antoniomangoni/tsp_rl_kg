import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from torch_geometric.data import Data
from collections import deque
import logging
from agent import Agent
from agent_model import AgentModel
from simulation_manager import SimulationManager

def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class CustomEnv(gym.Env):
    def __init__(self, game_manager_args, simulation_manager_args, model_args, plot=False):
        super(CustomEnv, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CustomEnv")

        # Base rewards
        self.new_outpost_reward = 30
        self.completion_reward = 100
        self.route_improvement_reward = 200
        self.better_route_than_algo_reward = 200
        
        # Penalties
        self.penalty_per_step = -0.5
        self.farther_from_outpost_penalty = -1.0
        self.circular_behavior_penalty = -2.0
        
        # Positive reinforcements
        self.closer_to_outpost_reward = 0.55 # Decreased from 2.0
        
        # New parameters
        self.time_penalty_factor = -0.01  # For the new time penalty
        self.outpost_reward_increase_factor = 0.5  # For increasing rewards of subsequent outposts
        self.completion_time_bonus_factor = 1.0  # For time bonus in completion reward

        # Route improvement tracking
        self.max_not_improvement_routes = 5
        self.num_not_improvement_routes = 0

        self.best_route_energy = float('inf')
        self.previous_best_route_energy = float('inf')
        self.best_efficiency = 0
        self.current_efficiency = 0
        self.improvement = 0
        self.gap = 0

        self.agent_steps = 0
        self.current_reward = 0
        self.outposts_visited = set()
        self.recent_path = None  # Will be initialized in reset()
        self.game_worlds_trained_in = 0
        self.max_game_worlds_trained_in = min(100, simulation_manager_args['number_of_environments'] // 2)

        self.num_actions = model_args['num_actions']
        self.num_tiles = game_manager_args['num_tiles']
        self.screen_size = game_manager_args['screen_size']
        # self.kg_completeness = game_manager_args['kg_completeness']
        self.vision_range = game_manager_args['vision_range']
    
        self.simulation_manager = SimulationManager(
            game_manager_args,
            simulation_manager_args['number_of_environments'], 
            simulation_manager_args['number_of_curricula'],
            simulation_manager_args['min_episodes_per_curriculum'],
            plot=plot
        )
        
        self.current_game_index = self.simulation_manager.curriculum_indices[0]
        self.set_current_game_manager()

        self.max_nodes = self.kg.graph_manager.max_nodes
        self.max_edges = self.kg.graph_manager.max_edges

        self.vision_pixel_side_size = (2 * self.vision_range + 1) * self.current_gm.tile_size
        vision_shape = (3, self.vision_pixel_side_size, self.vision_pixel_side_size)
        vision_space = spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.float16)

        # Flatten graph data into fixed-size arrays
        node_feature_space = spaces.Box(low=0, high=7, shape=(self.max_nodes, self.kg.graph.num_node_features), dtype=np.uint8)
        edge_attr_space = spaces.Box(low=0, high=1000, shape=(self.max_edges, self.kg.graph.num_edge_features), dtype=np.uint8)
        edge_index_space = spaces.Box(low=0, high=self.max_nodes-1, shape=(2, self.max_edges), dtype=np.int64)

        self.observation_space = spaces.Dict({
            'vision': vision_space,
            'node_features': node_feature_space,
            'edge_attr': edge_attr_space,
            'edge_index': edge_index_space
        })

        self.action_space = spaces.Discrete(self.num_actions)
        self.step_count = 0
        self.max_episode_steps = 5000  # Maximum number of steps per episode

        # New attributes for progress tracking
        self.steps_without_progress = 0
        self.max_steps_without_progress = 500  # Adjust as needed
        self.best_distance_to_unvisited = float('inf')

        self.episode_step = 0
        self.total_reward = 0
        self.logger.info("CustomEnv initialized successfully")

    def set_kg_completeness(self, completeness):
        self.logger.info(f"Setting KG completeness to {completeness} using SimulationManager")
        self.kg_completeness = completeness

    def set_current_game_manager(self):
        self.logger.info(f"Setting current game manager to index {self.current_game_index}")
        print(f'Current game index: {self.current_game_index}')

        self.current_gm = self.simulation_manager.game_managers[self.current_game_index]
        self.current_gm.start_game()
        self.environment = self.current_gm.environment
        self.agent_controler: Agent = self.current_gm.agent_controler
        self.agent_controler.reset_agent()
        self.kg = self.current_gm.kg_class
        self.outpost_coords = self.environment.outpost_locations
        self.logger.info("Current game manager set successfully")

    def reset(self, seed=None, options=None):
        self.logger.info("Resetting environment")
        self.episode_step = 0
        self.total_reward = 0
        self.outposts_visited.clear()
        self.early_stop = False
        self.step_count = 0
        self.steps_without_progress = 0
        self.best_distance_to_unvisited = float('inf')
        self.recent_path = deque(maxlen=len(self.outpost_coords))
        
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        
        # Update the game manager
        self.current_game_index += 1
        if self.current_game_index >= self.simulation_manager.number_of_environments:
            self.early_stop = True
            self.logger.info("All environments completed. Ending simulation.")
            return self._get_observation(), {} 
        self.set_current_game_manager()

        self.num_not_improvement_routes = 0
        self.previous_min_distance = float('inf')
        
        observation = self._get_observation()
        
        assert self.observation_space['vision'].contains(observation['vision']), f"Vision data out of bounds: min={observation['vision'].min()}, max={observation['vision'].max()}"
        
        return observation, {}  # Return observation and an empty info dict

    def _calculate_reward(self):
        self.logger.info("Calculating reward...")
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        
        # Start with the step penalty based on the energy requirement of the current terrain
        current_terrain_energy_requirement = self.current_gm.target_manager.energy_req_grid[agent_pos]
        reward = self.penalty_per_step * current_terrain_energy_requirement
        
        # Implement time penalty
        time_penalty = -0.01 * self.episode_step
        reward += time_penalty
        
        # Check if agent reached a new outpost
        if agent_pos in self.outpost_coords and agent_pos not in self.outposts_visited:
            print(f'Agent reached new outpost at {agent_pos}')
            self.outposts_visited.add(agent_pos)
            outposts_visited = len(self.outposts_visited)
            # Increase reward for reaching outposts, with higher rewards for later outposts
            outpost_reward = self.new_outpost_reward * (1 + 0.5 * (outposts_visited - 1))
            reward += outpost_reward
            self.logger.info(f"Agent reached new outpost. Reward: {outpost_reward}")
            self.recent_path.clear()  # Clear the path memory when reaching a new outpost
            
            # Check if all outposts are visited
            if len(self.outposts_visited) == len(self.outpost_coords):
                completion_reward = self.completion_reward * (1 + 1 / self.episode_step)
                reward += completion_reward
                
                agent_route_energy = self.agent_controler.energy_spent
                algorithmic_best_energy = self.current_gm.target_manager.target_route_energy
                
                self.current_efficiency = self.calculate_route_efficiency(agent_route_energy, algorithmic_best_energy)
                
                if self.current_efficiency > self.best_efficiency:
                    self.improvement = self.calculate_relative_improvement(agent_route_energy, algorithmic_best_energy)
                    self.gap = self.calculate_efficiency_gap(agent_route_energy, algorithmic_best_energy)
                    
                    improvement_reward = self.completion_reward * self.improvement  # Scale improvement reward
                    reward += improvement_reward
                    
                    self.best_route_energy = agent_route_energy
                    self.best_efficiency = self.current_efficiency
                    self.num_not_improvement_routes = 0
                    
                    if agent_route_energy < algorithmic_best_energy:
                        reward += self.better_route_than_algo_reward
                else:
                    self.num_not_improvement_routes += 1
                    if self.num_not_improvement_routes >= self.max_not_improvement_routes:
                        self.early_stop = True
                
                self.previous_best_route_energy = min(self.previous_best_route_energy, agent_route_energy)
                self.agent_controler.reset_energy_spent()
                
                # Log performance metrics
                self.logger.info(f"Route Completed - Efficiency: {self.current_efficiency:.2f}, Improvement: {self.improvement:.2f}, Gap: {self.gap:.2f}")

            else:     
                unvisited_outposts = set(self.outpost_coords) - self.outposts_visited
                current_min_distance = min(manhattan_distance(agent_pos, outpost) for outpost in unvisited_outposts)
                
                if current_min_distance < self.previous_min_distance:
                    closer_reward = self.closer_to_outpost_reward * (self.previous_min_distance - current_min_distance) / self.previous_min_distance
                    reward += closer_reward
                    self.logger.info(f"Agent moved closer to an outpost. Reward: {closer_reward}")
                elif current_min_distance > self.previous_min_distance:
                    farther_penalty = self.farther_from_outpost_penalty * (current_min_distance - self.previous_min_distance) / self.previous_min_distance
                    reward += farther_penalty
                    self.logger.info(f"Agent moved away from outposts. Penalty: {farther_penalty}")
                
                self.previous_min_distance = current_min_distance

        
        # Check for circular behavior
        if agent_pos in self.recent_path:
            reward += self.circular_behavior_penalty
            self.logger.info(f"Agent repeated a path. Penalty: {self.circular_behavior_penalty}")
        
        # Update recent path memory
        self.recent_path.append(agent_pos)
        
        return self._normalize_reward(reward)

    def _normalize_reward(self, reward):
        # Normalize reward to a fixed range (0-100)
        min_reward = self.penalty_per_step * self.max_episode_steps
        max_reward = self.completion_reward + self.new_outpost_reward * len(self.outpost_coords) + self.route_improvement_reward + self.better_route_than_algo_reward
        normalized_reward = 100 * (reward - min_reward) / (max_reward - min_reward)
        return max(0, min(normalized_reward, 100))  # Clamp between 0 and 100

    def get_episode_performance(self):
        return self.total_reward

    def step(self, action):
        self.episode_step += 1

        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = action.item()
        
        prev_position = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        self.agent_controler.agent_action(action)
        new_position = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        
        self.current_gm.rerender()
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        # Determine if the episode was successful (all outposts visited)
        success = len(self.outposts_visited) == len(self.outpost_coords)

        if terminated or truncated:
            self.simulation_manager.add_episode_performance(self.total_reward, success)
            if self.simulation_manager.should_advance_curriculum():
                self.current_game_index = self.simulation_manager.advance_curriculum()
                if self.current_game_index > self.simulation_manager.number_of_environments:
                    self.logger.info("All curricula completed. Ending simulation.")
                    self.early_stop = True
                else:
                    self.set_current_game_manager()
                    self.reset(False)

        observation = self._get_observation()
        info = {
            "episode_step": self.episode_step,
            "prev_position": prev_position,
            "new_position": new_position,
            "energy_spent": self.agent_controler.energy_spent,
            "outposts_visited": len(self.outposts_visited),
            "total_reward": self.total_reward
        }
        
        self.logger.debug(f"Step complete. Reward: {reward}, Total Reward: {self.total_reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        
        if terminated or truncated:
            self.logger.info(f"Episode ended. Total steps: {self.episode_step}, Total reward: {self.total_reward}, Outposts visited: {len(self.outposts_visited)}/{len(self.outpost_coords)}")
        
        return observation, reward, terminated, truncated, info

    def _check_termination(self):
        # Check if all outposts are visited
        if len(self.outposts_visited) == len(self.outpost_coords):
            self.logger.info("All outposts visited. Terminating episode.")
            return True

        # Check for no progress
        current_position = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        unvisited_outposts = set(self.outpost_coords) - self.outposts_visited
        if unvisited_outposts:
            current_min_distance = min(manhattan_distance(current_position, outpost) for outpost in unvisited_outposts)
            
            if current_min_distance < self.best_distance_to_unvisited:
                self.best_distance_to_unvisited = current_min_distance
                self.steps_without_progress = 0
            else:
                self.steps_without_progress += 1

            if self.steps_without_progress >= self.max_steps_without_progress:
                self.logger.info(f"No progress made for {self.max_steps_without_progress} steps. Terminating episode.")
                return True

        return False

    def _get_observation(self):
        self.logger.debug("Getting observation")
        vision = self._get_vision()
        graph: Data = self.current_gm.kg_class.get_subgraph()

        # Ensure correct shapes
        node_features = np.zeros((self.max_nodes, graph.num_node_features), dtype=np.float16)
        node_features[:graph.num_nodes, :] = graph.x.numpy()

        edge_attr = np.zeros((self.max_edges, graph.num_edge_features), dtype=np.float16)
        edge_attr[:graph.num_edges, :] = graph.edge_attr.numpy()

        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_index[:, :graph.num_edges] = graph.edge_index.numpy()

        self.logger.debug("Observation retrieved")
        return {
            'vision': vision.astype(np.float16) / 255.0,  # Normalize to [0, 1]
            'node_features': node_features,
            'edge_attr': edge_attr,
            'edge_index': edge_index
        }

    def get_clamped_surface(self):
        x = (self.agent_controler.agent.grid_x - self.vision_range) * self.current_gm.tile_size
        y = (self.agent_controler.agent.grid_y - self.vision_range) * self.current_gm.tile_size
        width = height = self.vision_pixel_side_size
        surface_rect = pygame.Rect(x, y, width, height)
        surface_rect.clamp_ip(self.current_gm.renderer.surface.get_rect())
        return self.current_gm.renderer.surface.subsurface(surface_rect)

    def _get_vision(self):
        vision_surface = self.get_clamped_surface()
        vision_array = pygame.surfarray.array3d(vision_surface).astype(np.float16)
        vision_array = np.transpose(vision_array, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
        return vision_array
    
    def close(self):
        self.current_gm.end_game()
        self.simulation_manager.save_data(self.kg_completeness)

    def _validate_graph_observation(self, observation):
        valid_node_range = (observation['node_features'].min() >= 0) and (observation['node_features'].max() <= 7)
        valid_edge_attr_range = (observation['edge_attr'].min() >= 0) and (observation['edge_attr'].max() <= 232)
        valid_edge_index_range = (observation['edge_index'].min() >= 0) and (observation['edge_index'].max() <= 128)
        return valid_node_range and valid_edge_attr_range and valid_edge_index_range
    
    def calculate_route_efficiency(self, agent_route_energy, algorithmic_best_energy):
        if algorithmic_best_energy == 0:
            return 0
        return algorithmic_best_energy / agent_route_energy  # No longer capped at 1

    def calculate_relative_improvement(self, current_route_energy, algorithmic_best_energy):
        current_efficiency = self.calculate_route_efficiency(current_route_energy, algorithmic_best_energy)
        previous_efficiency = self.calculate_route_efficiency(self.previous_best_route_energy, algorithmic_best_energy)
        
        if previous_efficiency == 0:
            return float('inf') if current_efficiency > 0 else 0
        
        return (current_efficiency - previous_efficiency) / previous_efficiency

    def calculate_efficiency_gap(self, agent_route_energy, algorithmic_best_energy):
        return (agent_route_energy - algorithmic_best_energy) / algorithmic_best_energy

    def interpret_efficiency(self, efficiency):
        if efficiency > 1:
            return f"Outperforming algorithm by {(efficiency - 1) * 100:.2f}%"
        elif efficiency == 1:
            return "Matching algorithmic best"
        else:
            return f"{(1 - efficiency) * 100:.2f}% away from algorithmic best"

    def get_metrics(self):
        return {
            'performance': self.get_episode_performance(),
            'game_manager_index': self.current_game_index,
            'best_route_energy': self.best_route_energy,
            'curriculum_index': self.simulation_manager.current_curriculum_index,
            'target_route_energy': self.current_gm.target_manager.target_route_energy,
            'best_efficiency': self.best_efficiency,
            'improvement': self.improvement,
            'gap': self.gap
            }
    