"""Executable reference, lifecycle, sensing, and policy-input regression gates."""

import numpy as np
import pytest
import torch
from loguru import logger

from tsp_rl_kg.config import (
    AgentModelConfig,
    EpisodeConfig,
    GameManagerConfig,
    ModelArgs,
    RewardConfig,
    SimulationManagerConfig,
    TrainingConfig,
)
from tsp_rl_kg.game_world.actions import ActionType
from tsp_rl_kg.game_world.game_manager import GameManager
from tsp_rl_kg.game_world.world_template import WorldTemplate
from tsp_rl_kg.graph.feature_encoder import OneHotEncoder
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.encoders import HybridEncoder
from tsp_rl_kg.rl.training.baselines import FullKnowledgeBaseline, RememberedBFSBaseline


@pytest.fixture
def env_factory():
    logger.disable("tsp_rl_kg")
    environments = []

    def create(completeness=1, **kwargs):
        # Winding but connected path. All locations/energies are hand controlled.
        terrain = tuple(tuple(2 for _ in range(6)) for _ in range(6))
        entities = np.zeros((6, 6), dtype=int)
        entities[0, 0] = entities[5, 5] = 5
        entities[2, :5] = 2
        template = WorldTemplate(
            terrain, tuple(map(tuple, entities.tolist())), (0, 0), ((0, 0), (5, 5))
        )
        cfg = GameManagerConfig(num_tiles=6, number_of_outposts=2, headless=True, vision_range=1)
        gm = GameManager(config=cfg, template=template)
        env = CustomEnv(
            cfg,
            SimulationManagerConfig(number_of_environments=1),
            ModelArgs(),
            game_managers=[gm],
            feature_encoder=OneHotEncoder(),
            kg_completeness=completeness,
            seed=42,
            **kwargs,
        )
        environments.append(env)
        env.reset(options={"world_index": 0})
        return env

    yield create
    for env in environments:
        env.close()
    logger.enable("tsp_rl_kg")


def complete(env, baseline=FullKnowledgeBaseline):
    policy = baseline(env)
    observation = env._get_observation()
    records = []
    for _ in range(env.max_episode_steps):
        action, _ = policy.predict(observation)
        observation, _, terminal, truncated, info = env.step(action)
        if "episode_record" in info:
            records.append(info["episode_record"])
        if terminal or truncated:
            return terminal, truncated, records
    raise AssertionError("Episode failed to end")


def test_executable_reference_matches_energy_and_closes_once(env_factory):
    env = env_factory()
    expected = env.current_gm.target_manager.target_route_energy
    terminal, truncated, records = complete(env)
    assert terminal and not truncated and len(records) == 1
    assert records[0]["energy_spent"] == expected
    assert records[0]["success"] and records[0]["laps"] == 1
    assert env.kg.player_pos == env.episode.start
    assert records[0]["efficiency"] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset()
    assert env.episode.visited == {env.episode.start}
    assert env.episode.energy == env.episode.steps == env.episode.no_progress == 0


def test_last_outpost_does_not_terminate_before_return(env_factory):
    env = env_factory()
    policy = FullKnowledgeBaseline(env)
    while not env.episode.returning:
        obs, _, done, timeout, _ = env.step(policy.predict(env._get_observation())[0])
        assert not done and not timeout
    assert env.episode.start != env.kg.player_pos
    assert obs["status"][-2] == 1
    assert env.episode.no_progress == 0


def test_success_wins_over_step_limit(env_factory):
    env = env_factory()
    length = len(env.current_gm.target_manager.route_tiles) - 1
    env.max_episode_steps = length
    assert complete(env)[:2] == (True, False)


def test_no_progress_and_step_limit_are_truncations(env_factory):
    for limits in (
        EpisodeConfig(max_episode_steps=2),
        EpisodeConfig(max_episode_steps=100, max_steps_without_progress=2),
    ):
        env = env_factory(episode_config=limits)
        # Building at an outpost is stationary and senses no new tiles.
        for _ in range(2):
            _, _, terminal, truncated, info = env.step(ActionType.BUILD_PATH)
        assert not terminal and truncated and not info["success"]
        assert env.episode.energy == 6


def test_fixed_trace_task_signals_and_vision_ignore_completeness(env_factory):
    envs = [env_factory(c) for c in (0, 0.25, 1)]
    for action in [4, 5, 0, 1, 2, 3, 7, 8]:
        observations = [e.step(action) for e in envs]
        for obs, reward, done, trunc, _ in observations[1:]:
            first, first_reward, first_done, first_trunc, _ = observations[0]
            np.testing.assert_array_equal(first["vision"], obs["vision"])
            np.testing.assert_array_equal(first["status"], obs["status"])
            assert (reward, done, trunc) == (first_reward, first_done, first_trunc)


def test_stationary_actions_use_actual_energy_without_circular_penalty(env_factory):
    env = env_factory(reward_config=RewardConfig(shaping_weight=0))
    for action in (ActionType.BUILD_PATH, ActionType.SCOUT, ActionType.COLLECT_LEFT):
        previous = env.agent_controler.energy_spent
        _, reward, _, _, _ = env.step(action)
        assert reward == pytest.approx(-0.01 * (env.agent_controler.energy_spent - previous))


def test_compact_vision_display_independence_and_status_inventory(env_factory):
    env = env_factory()
    before = env._get_observation()
    env.environment.tile_size = 200
    env.agent_controler.wood = 3
    after = env._get_observation()
    assert before["vision"].shape == (15, 3, 3)
    np.testing.assert_array_equal(before["vision"], after["vision"])
    assert after["status"][4] == pytest.approx(0.6)
    assert env.observation_space.contains(after)


def test_queries_hide_unknown_outposts_and_mask_disconnected_paths(env_factory):
    unknown = env_factory(0)
    queries = unknown.path_queries()
    assert queries[:, 0].sum() == 1
    assert (queries[queries[:, 0] == 0] == 0).all()
    full = env_factory(1)
    assert full.path_queries()[:, 1].all()
    far = next(p for p in full.outpost_coords if p != full.episode.start)
    full.kg.state.known_tiles[:] = False
    full.kg.state.known_tiles[full.episode.start] = full.kg.state.known_tiles[far] = True
    queries = full.path_queries()
    row = queries[full.outpost_coords.index(far)]
    assert row[0] == 1 and row[1] == row[-1] == 0


def test_direction_reverse_and_player_geometry(env_factory):
    env = env_factory()
    obs = env._get_observation()
    n, e = obs["num_nodes"][0], obs["num_edges"][0]
    nodes, edges, attrs = obs["node_features"][:n], obs["edge_index"][:, :e], obs["edge_attr"][:e]
    assert nodes[:, -3].sum() == 1
    for i, (a, b) in enumerate(edges.T):
        reverse = np.flatnonzero((edges[0] == b) & (edges[1] == a))
        assert len(reverse) == 1
        np.testing.assert_array_equal(attrs[i, -2:], -attrs[reverse[0], -2:])


def test_policy_modes_identical_and_queries_disabled_with_graph(env_factory):
    env = env_factory()
    obs = {k: torch.tensor(v)[None] for k, v in env._get_observation().items()}
    model = HybridEncoder(env.observation_space)
    model.train()
    first = model(obs)
    model.eval()
    torch.testing.assert_close(first, model(obs), rtol=0, atol=0)
    cfg = AgentModelConfig(disable_graph=True, use_path_queries=True)
    vision_only = HybridEncoder(env.observation_space, model_config=cfg)
    expected = vision_only(obs)
    obs["path_queries"].fill_(float("nan"))
    obs["node_features"].fill_(float("nan"))
    torch.testing.assert_close(expected, vision_only(obs))


def test_far_disconnected_outpost_affects_gnn_without_queries(env_factory):
    env = env_factory()
    model = HybridEncoder(env.observation_space, model_config=AgentModelConfig(disable_vision=True))
    obs = env._get_observation()
    n = int(obs["num_nodes"][0])
    # Delete all edges: attention still accesses distant entity geometry.
    obs["num_edges"][:] = 0
    batch = {k: torch.tensor(v)[None].float() for k, v in obs.items()}
    first = model(batch)
    far = np.flatnonzero(obs["node_features"][:n, -4] != 0)[-1]
    batch["node_features"][0, far, -5:-3] *= -1
    assert not torch.allclose(first, model(batch))


def test_remembered_bfs_completes_full_knowledge(env_factory):
    assert complete(env_factory(), RememberedBFSBaseline)[:2] == (True, False)


def test_reward_and_schema_configuration_roundtrip():
    config = TrainingConfig(reward=RewardConfig(shaping_weight=0.5))
    assert TrainingConfig.from_dict(config.to_dict()).reward == config.reward
    with pytest.raises(ValueError, match="fresh training"):
        TrainingConfig.from_dict(dict(config.to_dict(), observation_schema_version=2))


def test_build_collect_blocked_move_and_reward_use_actual_energy(env_factory):
    from tsp_rl_kg.game_world.entities import Tree

    env = env_factory(reward_config=RewardConfig(shaping_weight=0))
    # Exercise actual mechanics on a known non-outpost plains tile.
    env.agent_controler.agent.grid_x = 1
    env.agent_controler.agent.grid_y = 1
    env.kg.move_player_node(1, 1)
    env.agent_controler.wood = 1
    _, reward, _, _, _ = env.step(ActionType.BUILD_PATH)
    assert reward == pytest.approx(-0.03)
    assert env.environment.entity_index_grid[1, 1] == 6
    assert env.environment.terrain_object_grid[1, 1].energy_requirement == 0
    previous = env.agent_controler.energy_spent
    _, reward, _, _, _ = env.step(ActionType.MOVE_RIGHT)  # blocked by the wall
    assert env.kg.player_pos == (1, 1)
    assert env.agent_controler.energy_spent == previous
    assert reward == 0  # actual stationary-tile cost is zero after path building
    assert isinstance(env.environment.terrain_object_grid[2, 1].entity_on_tile, Tree)
    _, reward, _, _, _ = env.step(ActionType.COLLECT_RIGHT)
    assert reward == pytest.approx(-0.03)
    assert env.environment.terrain_object_grid[2, 1].passable
    previous = env.agent_controler.energy_spent
    _, reward, _, _, _ = env.step(ActionType.MOVE_RIGHT)
    assert env.kg.player_pos == (2, 1)
    assert reward == pytest.approx(-0.01 * (env.agent_controler.energy_spent - previous))


def test_sense_exactly_once_per_action(env_factory, monkeypatch):
    env = env_factory()
    calls = []
    original = env.kg.sense

    def record(radius=None):
        calls.append(radius)
        return original(radius)

    monkeypatch.setattr(env.kg, "sense", record)
    for action in (ActionType.MOVE_RIGHT, ActionType.SCOUT, ActionType.BUILD_PATH):
        calls.clear()
        env.step(action)
        assert len(calls) == 1
