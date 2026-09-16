"""Behavioral regressions for the observation schema and initial-prior semantics."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tsp_rl_kg.config import AgentModelConfig
from tsp_rl_kg.game_world.actions import ActionType
from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.game_world.world_template import WorldTemplate
from tsp_rl_kg.graph.feature_encoder import EmbeddingLookupEncoder, OneHotEncoder, RawIntEncoder
from tsp_rl_kg.graph.projection import KHopProjection
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph
from tsp_rl_kg.knowledge.state import KnowledgeState
from tsp_rl_kg.observation.encoder import PaddedPyGObservationEncoder
from tsp_rl_kg.observation.semantic_vision import render_semantic_vision
from tsp_rl_kg.rl.encoders import HybridEncoder


def test_projection_relabels_nonzero_original_ids():
    graph = Data(
        x=torch.arange(5).float()[:, None],
        edge_index=torch.tensor([[3, 4], [4, 3]]),
        edge_attr=torch.tensor([[13.0], [14.0]]),
    )
    projected = KHopProjection(1).project(graph, graph.edge_index, 4)
    assert projected.x[:, 0].tolist() == [3, 4]
    assert projected.edge_index.tolist() == [[0, 1], [1, 0]]
    assert projected.edge_attr.tolist() == [[13], [14]]


def make_encoder(max_nodes=8, max_edges=12):
    observation = PaddedPyGObservationEncoder(max_nodes, max_edges, 2, 1, (3, 4, 4))
    model = HybridEncoder(
        observation.observation_space(),
        features_dim=8,
        model_config=AgentModelConfig(
            vision_num_conv_layers=1,
            vision_conv_channels=[4],
            vision_fc_dims=[8],
            graph_num_gat_layers=1,
            graph_gat_heads=[1],
            graph_fc_dims=[8],
            gat_hidden_dim=4,
            dropout=0,
        ),
    )
    model.eval()
    return observation, model


def sample(encoder, n, zero_edges=False):
    graph = Data(
        x=torch.arange(n * 2).float().reshape(n, 2),
        edge_index=(
            torch.empty((2, 0), dtype=torch.long)
            if zero_edges
            else torch.tensor([[0, n - 1], [n - 1, 0]])
        ),
        edge_attr=torch.empty((0, 1)) if zero_edges else torch.tensor([[0.0], [1.0]]),
    )
    return encoder.encode(graph, np.zeros((3, 4, 4), dtype=np.uint8))


def stack(samples):
    return {key: torch.as_tensor(np.stack([s[key] for s in samples])).float() for key in samples[0]}


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batch_isolation_and_standalone_equivalence(batch_size):
    encoder, model = make_encoder()
    samples = [sample(encoder, i + 1, zero_edges=i == 0) for i in range(batch_size)]
    batch = stack(samples)
    x, edges, membership, attrs = model._prepare_graph_batch(batch)
    assert len(x) == sum(range(1, batch_size + 1))
    assert len(attrs) == edges.shape[1]
    assert torch.equal(membership[edges[0]], membership[edges[1]])
    with torch.no_grad():
        expected = torch.cat([model(stack([s])) for s in samples])
        torch.testing.assert_close(model(batch), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            model(stack(samples[::-1])), expected.flip(0), rtol=1e-5, atol=1e-6
        )


def test_padding_capacity_and_values_do_not_affect_embeddings():
    small, model = make_encoder(4, 4)
    large, other = make_encoder(10, 20)
    other.load_state_dict(model.state_dict())
    a, b = sample(small, 2), sample(large, 2)
    b["node_features"][2:] = np.nan
    b["edge_attr"][2:] = np.nan
    b["edge_index"][:, 2:] = 999
    with torch.no_grad():
        torch.testing.assert_close(model(stack([a])), other(stack([b])))


@pytest.mark.parametrize("count", [-1, 0, 9, 1.5, float("nan")])
def test_invalid_node_counts_fail(count):
    encoder, model = make_encoder()
    obs = stack([sample(encoder, 2)])
    obs["num_nodes"][0] = count
    with pytest.raises(ValueError):
        model(obs)


def test_legacy_observation_requires_retraining():
    encoder, _ = make_encoder()
    space = encoder.observation_space()
    del space.spaces["num_nodes"]
    with pytest.raises(ValueError, match="fresh training"):
        HybridEncoder(space)


def test_nested_priors_and_rng_isolation(headless_environment):
    env = headless_environment
    np.random.seed(12)
    expected = np.random.random(4)
    np.random.seed(12)
    states = [KnowledgeState(env, c, 17) for c in (0, 0.25, 0.5, 1)]
    np.testing.assert_array_equal(expected, np.random.random(4))
    for a, b in zip(states, states[1:], strict=False):
        assert (a.prior_tiles <= b.prior_tiles).all()
    for c, state in zip((0, 0.25, 0.5, 1), states, strict=True):
        assert state.prior_tiles.sum() == int(c * env.width * env.height)
        np.testing.assert_array_equal(state.prior_tiles, KnowledgeState(env, c, 17).prior_tiles)
    assert not np.array_equal(states[2].prior_tiles, KnowledgeState(env, 0.5, 18).prior_tiles)


def test_unseen_changes_stay_remembered_and_prior_does_not_discover(headless_environment):
    env = headless_environment
    env.player.grid_x = env.player.grid_y = 0
    kg = KnowledgeGraph(env, vision_range=1, completion=1)
    assert env.discovered_grid.sum() == 4
    before = kg.get_subgraph().x.clone()
    env.terrain_index_grid[-1, -1] = (env.terrain_index_grid[-1, -1] + 1) % 6
    kg.sense()
    torch.testing.assert_close(kg.get_subgraph().x, before)
    kg.observe_tile(env.width - 1, env.height - 1)
    assert not torch.equal(kg.get_subgraph().x, before)


def test_unknown_tiles_are_absent_and_local_edges_are_valid(headless_environment):
    kg = KnowledgeGraph(headless_environment, 0, completion=0)
    graph = kg.get_subgraph()
    assert graph.num_nodes == 3  # player, current terrain, current entity slot
    assert graph.edge_index.max() < graph.num_nodes


@pytest.mark.parametrize("kind", ["raw_int", "one_hot", "embedding_lookup"])
def test_player_movement_preserves_features(kind, headless_environment, tmp_path):
    if kind == "embedding_lookup":
        from test_custom_env import _write_embedding_assets

        schema, embeddings = tmp_path / "schema.toml", tmp_path / "embeddings.npy"
        _write_embedding_assets(schema, embeddings)
        encoder = EmbeddingLookupEncoder(str(embeddings), str(schema))
    else:
        encoder = RawIntEncoder() if kind == "raw_int" else OneHotEncoder()
    kg = KnowledgeGraph(headless_environment, 1, feature_encoder=encoder)
    headless_environment.player.grid_x = 0
    kg.move_player_node(0, headless_environment.player.grid_y)
    torch.testing.assert_close(kg.graph.x[kg.graph_manager.player_idx], encoder.encode_player())


def test_template_reset_and_path_persistence(headless_environment):
    env = headless_environment
    template = WorldTemplate.capture(env)
    kg = KnowledgeGraph(env, 1)
    agent = Agent(env, 1)
    agent.get_kg(kg)
    x, y = kg.player_pos
    # Ensure a legal adjacent move independently of generated entities.
    nx = x + 1 if x + 1 < env.width else x - 1
    entity = env.terrain_object_grid[nx, y].entity_on_tile
    if entity:
        env.delete_entity(entity)
    env.terrain_object_grid[nx, y].passable = True
    env.place_path(x, y)
    kg.build_path_node(x, y)
    agent.move_agent(nx - x, 0)
    assert env.entity_index_grid[x, y] == 6
    assert kg.state.entities[x, y] == 6
    restored = template.restore(tile_size=env.tile_size, headless=True)
    assert WorldTemplate.capture(restored) == template
    restored.terrain_index_grid[0, 0] = 99
    assert template.terrain[0][0] != 99
    assert restored.discovered_grid.sum() == 0


def test_vision_is_independent_of_completeness_and_display(headless_environment):
    template = WorldTemplate.capture(headless_environment)
    images = []
    for completeness in (0, 0.5, 1):
        env = template.restore(tile_size=2, headless=True)
        kg = KnowledgeGraph(env, 1, completion=completeness)
        image = render_semantic_vision(env, 1, kg.visible_tiles)
        env.headless = False
        np.testing.assert_array_equal(image, render_semantic_vision(env, 1, kg.visible_tiles))
        images.append(image)
    for image in images[1:]:
        np.testing.assert_array_equal(images[0], image)


def test_disabled_graph_is_not_executed():
    encoder, model = make_encoder()
    model.disable_graph = True

    def fail(*args, **kwargs):
        raise AssertionError("disabled branch ran")

    model.graph_processor.forward = fail
    model(stack([sample(encoder, 2)]))


def test_environment_reset_restores_same_world_and_observations(tmp_path, monkeypatch):
    from tsp_rl_kg.config import GameManagerConfig, ModelArgs, SimulationManagerConfig
    from tsp_rl_kg.rl.custom_env import CustomEnv

    monkeypatch.chdir(tmp_path)
    env = CustomEnv(
        GameManagerConfig(num_tiles=5, screen_size=20, headless=True),
        SimulationManagerConfig(number_of_environments=8, number_of_curricula=2),
        ModelArgs(),
        seed=17,
    )
    try:
        before, _ = env.reset(seed=17, options={"world_index": 0})
        env.agent_controler.wood = 5
        env.environment.terrain_index_grid[:] = 0
        env.step(ActionType.SCOUT)
        after, _ = env.reset(seed=17, options={"world_index": 0})
        assert env.agent_controler.wood == 0
        assert env.observation_space.contains(after)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        assert not (tmp_path / "results").exists()  # training cannot create play recordings
    finally:
        env.close()


def test_observations_backpropagate_with_real_environment(headless_environment):
    from stable_baselines3.common.preprocessing import preprocess_obs

    kg = KnowledgeGraph(headless_environment, 1, completion=0.25, feature_encoder=OneHotEncoder())
    vision = render_semantic_vision(headless_environment, 1, kg.visible_tiles)
    obs_encoder = PaddedPyGObservationEncoder(
        kg.num_possible_nodes,
        kg.num_possible_edges,
        kg.graph.num_node_features,
        kg.graph.num_edge_features,
        vision.shape,
    )
    obs = obs_encoder.encode(kg.get_subgraph(), vision)
    space = obs_encoder.observation_space()
    assert space.contains(obs)
    model = HybridEncoder(
        space,
        features_dim=8,
        model_config=AgentModelConfig(
            vision_num_conv_layers=1,
            vision_conv_channels=[4],
            vision_fc_dims=[8],
            graph_num_gat_layers=1,
            graph_gat_heads=[1],
            graph_fc_dims=[8],
            gat_hidden_dim=4,
        ),
    )
    batch = preprocess_obs(stack([obs, obs]), space)
    result = model(batch)
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_vision_entity_and_player_markers(headless_environment):
    env = headless_environment
    env.player.grid_x = env.player.grid_y = 0
    images = []
    for entity in range(7):
        env.entity_index_grid[0, 0] = entity
        images.append(render_semantic_vision(env, 1))
    assert len({image.tobytes() for image in images}) == 7
    assert not images[0][:, : env.tile_size, :].any()  # outside-world top row


@pytest.mark.parametrize("failure", ["edge", "capacity", "nan", "dimension"])
def test_observation_encoder_rejects_invalid_graph(failure):
    encoder, _ = make_encoder()
    graph = Data(
        x=torch.ones((2, 2)), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros((1, 1))
    )
    if failure == "edge":
        graph.edge_index[0, 0] = 2
    elif failure == "capacity":
        graph.x = torch.ones((9, 2))
    elif failure == "nan":
        graph.x[0, 0] = float("nan")
    else:
        graph.edge_attr = torch.zeros((1, 2))
    with pytest.raises(ValueError):
        encoder.encode(graph, np.zeros((3, 4, 4)))
