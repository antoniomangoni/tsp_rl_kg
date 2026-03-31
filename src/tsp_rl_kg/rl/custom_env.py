import logging

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from torch_geometric.data import Data

from tsp_rl_kg.config import (
    EpisodeConfig,
    GameManagerConfig,
    ModelArgs,
    RewardConfig,
    SimulationManagerConfig,
)
from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.rl.reward import RewardCalculator, manhattan_distance
from tsp_rl_kg.rl.simulation_manager import SimulationManager


class CustomEnv(gym.Env):
    def __init__(
        self,
        game_manager_args: GameManagerConfig | dict,
        simulation_manager_args: SimulationManagerConfig | dict,
        model_args: ModelArgs | dict,
        converter=None,
        plot=False,
        episode_config: EpisodeConfig | None = None,
    ):
        super(CustomEnv, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CustomEnv")

        # Normalize to typed configs
        if isinstance(game_manager_args, dict):
            gm_dict = {
                k: v
                for k, v in game_manager_args.items()
                if k in GameManagerConfig.__dataclass_fields__
            }
            gm_dict.setdefault("headless", True)
            self._gm_config = GameManagerConfig(**gm_dict)
        else:
            self._gm_config = game_manager_args

        if not self._gm_config.headless:
            self.logger.warning(
                "CustomEnv created with headless=False — training will require a display server."
            )

        if isinstance(simulation_manager_args, dict):
            self._sim_config = SimulationManagerConfig(
                **{
                    k: v
                    for k, v in simulation_manager_args.items()
                    if k in SimulationManagerConfig.__dataclass_fields__
                }
            )
        else:
            self._sim_config = simulation_manager_args

        if isinstance(model_args, dict):
            self._model_args = ModelArgs(
                **{k: v for k, v in model_args.items() if k in ModelArgs.__dataclass_fields__}
            )
        else:
            self._model_args = model_args

        # Reward system
        self._reward_config = RewardConfig()
        self._reward_calculator: RewardCalculator | None = None  # created after first game set

        # Episode limits
        if episode_config is None:
            episode_config = EpisodeConfig()
        self._episode_config = episode_config

        self.agent_steps = 0
        self.current_reward = 0
        self.game_worlds_trained_in = 0
        self.max_game_worlds_trained_in = min(
            self._episode_config.max_game_worlds_trained_in,
            self._sim_config.number_of_environments // 2,
        )

        self.num_actions = self._model_args.num_actions
        self.num_tiles = self._gm_config.num_tiles
        self.screen_size = self._gm_config.screen_size
        # self.kg_completeness = game_manager_args['kg_completeness']
        self.kg_completeness = 0.5
        self.vision_range = self._gm_config.vision_range

        self.simulation_manager = SimulationManager(
            self._gm_config,
            sim_config=self._sim_config,
            plot=plot,
            converter=converter,
        )

        self.current_game_index = self.simulation_manager.curriculum_indices[
            0
        ]  # Start with the first curriculum
        self.set_current_game_manager()

        self.max_nodes = self.kg.graph_manager.max_nodes
        self.max_edges = self.kg.graph_manager.max_edges

        self.vision_pixel_side_size = (2 * self.vision_range + 1) * self.current_gm.tile_size
        vision_shape = (3, self.vision_pixel_side_size, self.vision_pixel_side_size)
        vision_space = spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.float16)

        # Flatten graph data into fixed-size arrays
        if converter is None:
            node_feature_space = spaces.Box(
                low=0,
                high=7,
                shape=(self.max_nodes, self.kg.graph.num_node_features),
                dtype=np.uint8,
            )
        else:
            node_feature_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.max_nodes, converter.embedding_dim),
                dtype=np.float64,
            )

        edge_attr_space = spaces.Box(
            low=0,
            high=self.max_edges - 1,
            shape=(self.max_edges, self.kg.graph.num_edge_features),
            dtype=np.uint8,
        )
        edge_index_space = spaces.Box(
            low=0, high=self.max_nodes - 1, shape=(2, self.max_edges), dtype=np.int64
        )

        self.observation_space = spaces.Dict(
            {
                "vision": vision_space,
                "node_features": node_feature_space,
                "edge_attr": edge_attr_space,
                "edge_index": edge_index_space,
            }
        )

        self.action_space = spaces.Discrete(self.num_actions)
        self.step_count = 0
        self.max_episode_steps = self._episode_config.max_episode_steps

        # New attributes for progress tracking
        self.steps_without_progress = 0
        self.max_steps_without_progress = self._episode_config.max_steps_without_progress
        self.best_distance_to_unvisited = float("inf")

        self.episode_step = 0
        self.total_reward = 0
        self.logger.info("CustomEnv initialized successfully")

    def set_kg_completeness(self, completeness):
        self.logger.info(f"Setting KG completeness to {completeness} using SimulationManager")
        self.kg_completeness = completeness

    def set_current_game_manager(self):
        self.logger.info(f"Setting current game manager to index {self.current_game_index}")
        print(f"Current game index: {self.current_game_index}")

        self.current_gm = self.simulation_manager.game_managers[self.current_game_index]
        self.current_gm.start_game(kg_completeness=self.kg_completeness)
        self.environment = self.current_gm.environment
        self.agent_controler: Agent = self.current_gm.agent_controler
        self.agent_controler.reset_agent()
        self.kg = self.current_gm.kg_class
        self.outpost_coords = self.environment.outpost_locations

        # (Re)initialise reward calculator for this game world
        if self._reward_calculator is None:
            self._reward_calculator = RewardCalculator(
                self._reward_config,
                list(self.outpost_coords),
                self.max_episode_steps,
            )
        else:
            self._reward_calculator.reset_game(list(self.outpost_coords))
        self.logger.info("Current game manager set successfully")

    def reset(self, seed=None, options=None):
        self.logger.info("Resetting environment")
        self.episode_step = 0
        self.total_reward = 0
        self.early_stop = False
        self.step_count = 0
        self.steps_without_progress = 0
        self.best_distance_to_unvisited = float("inf")

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

        observation = self._get_observation()

        assert self.observation_space["vision"].contains(observation["vision"]), (
            f"Vision data out of bounds: "
            f"min={observation['vision'].min()}, "
            f"max={observation['vision'].max()}"
        )

        return observation, {}  # Return observation and an empty info dict

    def _calculate_reward(self) -> tuple[float, bool]:
        agent_pos = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        terrain_energy = self.current_gm.target_manager.energy_req_grid[agent_pos]
        agent_energy_spent = self.agent_controler.energy_spent
        algorithmic_best_energy = self.current_gm.target_manager.target_route_energy

        reward, early_stop = self._reward_calculator.calculate(
            agent_pos=agent_pos,
            terrain_energy=terrain_energy,
            episode_step=self.episode_step,
            agent_energy_spent=agent_energy_spent,
            algorithmic_best_energy=algorithmic_best_energy,
            reset_energy_callback=self.agent_controler.reset_energy_spent,
        )
        return reward, early_stop

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
        reward, early_stop = self._calculate_reward()
        if early_stop:
            self.early_stop = True
        self.total_reward += reward

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps

        # Determine if the episode was successful (all outposts visited)
        success = len(self._reward_calculator.outposts_visited) == len(self.outpost_coords)

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
            "outposts_visited": len(self._reward_calculator.outposts_visited),
            "total_reward": self.total_reward,
        }

        self.logger.debug(
            f"Step complete. Reward: {reward}, "
            f"Total Reward: {self.total_reward}, "
            f"Terminated: {terminated}, "
            f"Truncated: {truncated}, Info: {info}"
        )

        if terminated or truncated:
            visited = len(self._reward_calculator.outposts_visited)
            total = len(self.outpost_coords)
            self.logger.info(
                f"Episode ended. Total steps: {self.episode_step}, "
                f"Total reward: {self.total_reward}, "
                f"Outposts visited: {visited}/{total}"
            )

        return observation, reward, terminated, truncated, info

    def _check_termination(self):
        # Check if all outposts are visited
        if self.early_stop:
            self.logger.info("Early stop condition reached. Terminating episode.")
            return True

        # Check for no progress
        current_position = (self.agent_controler.agent.grid_x, self.agent_controler.agent.grid_y)
        unvisited_outposts = set(self.outpost_coords) - self._reward_calculator.outposts_visited
        if unvisited_outposts:
            current_min_distance = min(
                manhattan_distance(current_position, outpost) for outpost in unvisited_outposts
            )

            if current_min_distance < self.best_distance_to_unvisited:
                self.best_distance_to_unvisited = current_min_distance
                self.steps_without_progress = 0
            else:
                self.steps_without_progress += 1

            if self.steps_without_progress >= self.max_steps_without_progress:
                self.logger.info(
                    f"No progress made for "
                    f"{self.max_steps_without_progress} steps. "
                    f"Terminating episode."
                )
                return True

        return False

    def _get_observation(self):
        self.logger.debug("Getting observation")
        vision = self._get_vision()
        graph: Data = self.current_gm.kg_class.get_subgraph()

        # Ensure correct shapes
        node_features = np.zeros((self.max_nodes, graph.num_node_features), dtype=np.float16)
        node_features[: graph.num_nodes, :] = graph.x.numpy()

        edge_attr = np.zeros((self.max_edges, graph.num_edge_features), dtype=np.float16)
        edge_attr[: graph.num_edges, :] = graph.edge_attr.numpy()

        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_index[:, : graph.num_edges] = graph.edge_index.numpy()

        self.logger.debug("Observation retrieved")
        return {
            "vision": vision.astype(np.float16) / 255.0,  # Normalize to [0, 1]
            "node_features": node_features,
            "edge_attr": edge_attr,
            "edge_index": edge_index,
        }

    def get_clamped_surface(self):
        x = (self.agent_controler.agent.grid_x - self.vision_range) * self.current_gm.tile_size
        y = (self.agent_controler.agent.grid_y - self.vision_range) * self.current_gm.tile_size
        width = height = self.vision_pixel_side_size
        surface_rect = pygame.Rect(x, y, width, height)
        surface_rect.clamp_ip(self.current_gm.renderer.surface.get_rect())
        return self.current_gm.renderer.surface.subsurface(surface_rect)

    def _get_vision(self):
        if self.current_gm.headless:
            return self._get_vision_headless()
        vision_surface = self.get_clamped_surface()
        vision_array = pygame.surfarray.array3d(vision_surface).astype(np.float16)
        vision_array = np.transpose(vision_array, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
        return vision_array

    def _get_vision_headless(self):
        """Build vision array from terrain colours without pygame surfaces."""
        env = self.environment
        agent_x = self.agent_controler.agent.grid_x
        agent_y = self.agent_controler.agent.grid_y
        vr = self.vision_range
        ts = self.current_gm.tile_size
        side = self.vision_pixel_side_size
        view_tiles = 2 * vr + 1

        # Clamp viewport origin to stay within map bounds (matches pygame clamp_ip)
        view_x = max(0, min(agent_x - vr, env.width - view_tiles))
        view_y = max(0, min(agent_y - vr, env.height - view_tiles))

        vision = np.zeros((side, side, 3), dtype=np.uint8)

        for dx in range(view_tiles):
            for dy in range(view_tiles):
                gx = view_x + dx
                gy = view_y + dy
                if 0 <= gx < env.width and 0 <= gy < env.height:
                    terrain = env.terrain_object_grid[gx, gy]
                    colour = terrain.colour if terrain.colour else (0, 0, 0)
                    px = dx * ts
                    py = dy * ts
                    vision[px : px + ts, py : py + ts] = colour

        return np.transpose(vision, (2, 0, 1)).astype(np.float16)

    def close(self):
        self.current_gm.end_game()
        self.simulation_manager.save_data(self.kg_completeness)

    def get_metrics(self):
        rc = self._reward_calculator
        return {
            "performance": self.get_episode_performance(),
            "game_manager_index": self.current_game_index,
            "best_route_energy": rc.best_route_energy,
            "curriculum_index": self.simulation_manager.current_curriculum_index,
            "target_route_energy": self.current_gm.target_manager.target_route_energy,
            "best_efficiency": rc.best_efficiency,
            "improvement": rc.improvement,
            "gap": rc.gap,
        }
