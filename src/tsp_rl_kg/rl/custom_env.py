"""Closed-tour navigation with separate task, sensed, and prior-knowledge state."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch

from tsp_rl_kg.config import (
    AblationConfig,
    CurriculumConfig,
    EpisodeConfig,
    GameManagerConfig,
    ModelArgs,
    RewardConfig,
    SimulationManagerConfig,
)
from tsp_rl_kg.knowledge.state import validate_completeness
from tsp_rl_kg.observation.encoder import PaddedPyGObservationEncoder
from tsp_rl_kg.observation.semantic_vision import compact_semantic_vision
from tsp_rl_kg.rl.episode import EpisodeState, efficiency, sensing_memory
from tsp_rl_kg.rl.pathfinding import remembered_grid, shortest_paths
from tsp_rl_kg.rl.reward import RewardCalculator, RewardTransition
from tsp_rl_kg.rl.simulation_manager import SimulationManager


def typed(value, cls):
    return cls(**value) if isinstance(value, dict) else value or cls()


class CustomEnv(gym.Env):
    def __init__(
        self,
        game_manager_args,
        simulation_manager_args,
        model_args,
        feature_encoder=None,
        plot=False,
        episode_config=None,
        game_managers=None,
        ablation_config=None,
        kg_completeness=0.5,
        seed=0,
        evaluation=False,
        reward_config=None,
        curriculum_config=None,
        gamma=0.99,
        world_seed=None,
    ):
        super().__init__()
        super().reset(seed=seed)
        self._gm_config = typed(game_manager_args, GameManagerConfig)
        self._sim_config = typed(simulation_manager_args, SimulationManagerConfig)
        self._model_args = typed(model_args, ModelArgs)
        self._episode_config = typed(episode_config, EpisodeConfig)
        self._ablation_config = typed(ablation_config, AblationConfig)
        self._reward_config = typed(reward_config, RewardConfig)
        self._seed = seed
        self.kg_completeness = validate_completeness(kg_completeness)
        self.evaluation = evaluation
        self.num_actions = self._model_args.num_actions
        self.num_tiles = self._gm_config.num_tiles
        self.screen_size = self._gm_config.screen_size
        self.vision_range = self._gm_config.vision_range
        self.max_episode_steps = self._episode_config.max_episode_steps
        self.max_steps_without_progress = self._episode_config.max_steps_without_progress
        self.training_complete = False
        self._pending_curriculum = False
        self.simulation_manager = SimulationManager(
            self._gm_config,
            sim_config=self._sim_config,
            plot=plot,
            feature_encoder=feature_encoder,
            game_managers=game_managers,
            curriculum_config=curriculum_config or CurriculumConfig(),
            rng=np.random.default_rng(seed if world_seed is None else world_seed),
        )
        # Starting outpost depends on the world/run seed, never on completeness.
        for gm in self.simulation_manager.game_managers:
            template = gm.world_template
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(template.terrain_fingerprint[:8], 16)])
            )
            spawn = template.outposts[int(rng.integers(len(template.outposts)))]
            gm.world_template = replace(template, spawn=spawn)
        self._reward_calculator = RewardCalculator(
            self._reward_config,
            gamma=gamma,
            disabled_components=self._ablation_config.disable_reward_components,
        )
        self.current_game_index = 0
        self.set_current_game_manager()
        self.max_nodes = self.kg.graph_manager.max_nodes
        self.max_edges = self.kg.graph_manager.max_edges
        self.encoder = PaddedPyGObservationEncoder(
            self.max_nodes,
            self.max_edges,
            self.kg.graph.num_node_features + 5,
            self.kg.graph.num_edge_features + 2,
            (15, 2 * self.vision_range + 1, 2 * self.vision_range + 1),
            status_dim=2 * len(self.outpost_coords) + 5,
            num_outposts=len(self.outpost_coords),
        )
        self.observation_space = self.encoder.observation_space()
        self.action_space = gym.spaces.Discrete(self.num_actions, seed=seed)

    def set_kg_completeness(self, completeness):
        self.kg_completeness = validate_completeness(completeness)

    def set_current_game_manager(self):
        self.current_gm = self.simulation_manager.game_managers[self.current_game_index]
        self.current_gm.start_game(
            self.kg_completeness, seed=self._seed, record=False, restore=True
        )
        self.environment = self.current_gm.environment
        self.agent_controler = self.current_gm.agent_controler
        self.agent_controler.reset_agent()
        self.kg = self.current_gm.kg_class
        self.outpost_coords = tuple(self.environment.outpost_locations)
        start = self.current_gm.world_template.spawn
        self.episode = EpisodeState(self.outpost_coords, start)
        self.sensed = sensing_memory(self.environment, self._seed, self.current_gm.world_template)
        self.sensed.observe(self.environment, self.kg.visible_tiles)
        self.episode.best_distance = self.episode.distance(self.sensed, start)
        self.total_reward = 0.0
        self.early_stop = False

    @property
    def episode_step(self):
        return self.episode.steps

    @property
    def steps_without_progress(self):
        return self.episode.no_progress

    @property
    def best_distance_to_unvisited(self):
        return self.episode.best_distance

    def begin_evaluation(self):
        self.current_game_index = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.current_game_index = -1
            self.action_space.seed(seed)
        if self._pending_curriculum and not self.evaluation:
            self.simulation_manager.advance_curriculum()
            self._pending_curriculum = False
        if options and "world_index" in options:
            self.current_game_index = int(options["world_index"])
            if not 0 <= self.current_game_index < self.simulation_manager.number_of_environments:
                raise ValueError("world_index is outside the world pool")
        elif self.evaluation:
            self.current_game_index = (
                self.current_game_index + 1
            ) % self.simulation_manager.number_of_environments
        elif self._ablation_config.disable_curriculum:
            self.current_game_index = int(
                self.np_random.integers(self.simulation_manager.number_of_environments)
            )
        else:
            self.current_game_index = self.simulation_manager.get_next_game_in_curriculum(
                self.current_game_index
            )
        self.set_current_game_manager()
        return self._get_observation(), {}

    def step(self, action):
        if self.episode.done:
            raise RuntimeError("Episode is complete; call reset before step")
        action = int(np.asarray(action).item())
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        agent = self.agent_controler
        previous = (agent.agent.grid_x, agent.agent.grid_y)
        phi_before = self.episode.potential(self.sensed, previous)
        energy_before = agent.energy_spent
        known_before = self.sensed.known_tiles.sum()
        agent.agent_action(action)
        position = (agent.agent.grid_x, agent.agent.grid_y)
        self.sensed.observe(self.environment, self.kg.visible_tiles)
        new_visit = self.episode.update(
            position,
            agent.energy_spent,
            action,
            self.sensed.known_tiles.sum() > known_before,
            self.sensed,
        )
        terminated = self.episode.success
        truncated = not terminated and (
            self.episode.steps >= self.max_episode_steps
            or self.episode.no_progress >= self.max_steps_without_progress
        )
        self.episode.done = terminated or truncated
        reward = self._reward_calculator.calculate(
            RewardTransition(
                agent.energy_spent - energy_before,
                self.episode.energy,
                self.current_gm.target_manager.target_route_energy,
                new_visit,
                terminated,
                phi_before,
                self.episode.potential(self.sensed, position),
            )
        )
        self.total_reward += reward
        info = self.get_metrics()
        info.update(prev_position=previous, new_position=position)
        if self.episode.done:
            reason = (
                "success"
                if terminated
                else (
                    "step_limit" if self.episode.steps >= self.max_episode_steps else "no_progress"
                )
            )
            info["termination_reason"] = reason
            info["episode_record"] = dict(info)
            if not self.evaluation:
                self.simulation_manager.add_episode_performance(info["efficiency"], terminated)
                if not self._ablation_config.disable_curriculum:
                    self._pending_curriculum = self.simulation_manager.should_advance_curriculum()
        self.current_gm.rerender()
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        graph = self.kg.get_subgraph()
        ids = graph.world_node_ids.cpu().numpy()
        coords = self.kg.node_coordinates[ids]
        levels = self.kg.node_levels[ids]
        position = np.array(self.kg.player_pos)
        extra = np.zeros((len(ids), 5), dtype=np.float32)
        extra[:, :2] = (coords - position) / max(1, self.num_tiles - 1)
        extra[:, 2] = levels == self.kg.player_z_level
        for i, (coord, level) in enumerate(zip(coords, levels, strict=False)):
            p = tuple(coord)
            if level == self.kg.entity_z_level and self.kg.state.entities[p] == 5:
                extra[i, 3] = p in self.episode.visited
                extra[i, 4] = p == self.episode.start
        graph.x = torch.cat([graph.x, torch.from_numpy(extra)], dim=1)
        source, dest = graph.edge_index.numpy()
        directions = coords[dest] - coords[source]
        graph.edge_attr = torch.cat(
            [graph.edge_attr, torch.tensor(directions, dtype=torch.float32)], dim=1
        )
        return self.encoder.encode(
            graph,
            self._get_vision(),
            self.episode.status(self.agent_controler, self._episode_config),
            self.path_queries(),
        )

    def path_queries(self):
        queries = np.zeros((len(self.outpost_coords), 6), dtype=np.float32)
        memory = self.kg.state
        costs, passable = remembered_grid(memory)
        distances, _ = shortest_paths(costs, passable, self.kg.player_pos)
        for i, p in enumerate(self.outpost_coords):
            if not memory.known_tiles[p] or memory.entities[p] != 5:
                continue
            reachable = np.isfinite(distances[p])
            queries[i] = [
                1,
                reachable,
                p in self.episode.visited,
                (p[0] - self.kg.player_pos[0]) / max(1, self.num_tiles - 1),
                (p[1] - self.kg.player_pos[1]) / max(1, self.num_tiles - 1),
                distances[p] / (12 * self.num_tiles**2) if reachable else 0,
            ]
        return queries

    def _get_vision(self):
        return compact_semantic_vision(self.environment, self.vision_range, self.kg.visible_tiles)

    def _get_vision_headless(self):
        return self._get_vision()

    def get_episode_performance(self):
        return self.total_reward

    def get_metrics(self):
        energy = self.current_gm.target_manager.target_route_energy
        ratio = efficiency(energy, self.episode.energy) if self.episode.success else 0.0
        return dict(
            performance=self.total_reward,
            total_reward=self.total_reward,
            episode_step=self.episode.steps,
            energy_spent=self.episode.energy,
            success=self.episode.success,
            is_success=self.episode.success,
            laps=int(self.episode.success),
            efficiency=ratio,
            best_efficiency=ratio,
            best_route_energy=self.episode.energy if self.episode.success else 0.0,
            best_route_found=self.episode.success,
            target_route_energy=energy,
            outposts_visited=len(self.episode.visited),
            known_tile_fraction=self.kg.state.known_fraction,
            kg_completeness=self.kg_completeness,
            action_counts=self.episode.action_counts.tolist(),
            game_manager_index=self.current_game_index,
            curriculum_level=self.simulation_manager.current_curriculum_index,
            world_id=self.current_gm.world_template.fingerprint,
        )

    def close(self):
        self.current_gm.end_game()
