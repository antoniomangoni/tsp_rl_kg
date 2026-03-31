"""Tests for GraphConstitution protocol and DefaultGridConstitution."""

from __future__ import annotations

import torch

from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.graph.constitution import (
    DefaultGridConstitution,
    GraphConstitution,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    assert isinstance(DefaultGridConstitution(), GraphConstitution)


# ---------------------------------------------------------------------------
# DefaultGridConstitution.build()
# ---------------------------------------------------------------------------


class TestDefaultGridConstitution:
    def test_build_returns_data_and_graph_manager(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        constitution = DefaultGridConstitution()
        graph, gm = constitution.build(env, player_pos, env.discovered_grid)

        assert graph is not None
        assert gm is not None

    def test_correct_node_count(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        graph, _gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        expected = env.width * env.height * 2 + 1
        assert graph.num_nodes == expected

    def test_correct_edge_count(self, headless_environment: Environment):
        env = headless_environment
        w, h = env.width, env.height
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, w)

        graph, _gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        terrain_edges = 2 * (w * (h - 1) + h * (w - 1))
        entity_edges = w * h * 2
        player_edges = 2
        assert graph.num_edges == terrain_edges + entity_edges + player_edges

    def test_all_nodes_initialized(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        graph, _gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        assert torch.all(graph.x[:, 0] >= 0)

    def test_all_edges_initialized(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        graph, _gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        assert torch.all(graph.edge_index >= 0)
        assert torch.all(graph.edge_attr[:, 1] >= 0)

    def test_graph_manager_player_idx_set(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        _graph, gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        assert gm.player_idx is not None
        assert gm.player_idx == 0  # Player node is always first

    def test_graph_manager_lookups_work(self, headless_environment: Environment):
        env = headless_environment
        player_pos = (env.player.grid_x, env.player.grid_y)
        env.init_discovered_area(player_pos, env.width)

        _graph, gm = DefaultGridConstitution().build(env, player_pos, env.discovered_grid)

        # Should be able to look up terrain/entity nodes
        terrain_idx = gm.get_node_idx((0, 0), DefaultGridConstitution.TERRAIN_Z)
        entity_idx = gm.get_node_idx((0, 0), DefaultGridConstitution.ENTITY_Z)
        assert terrain_idx is not None
        assert entity_idx is not None
        assert terrain_idx != entity_idx

    def test_matches_kg_construction(self, headless_environment: Environment):
        """Graph produced by constitution standalone should match KG's graph."""
        from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )

        assert (
            kg.graph.num_nodes == headless_environment.width * headless_environment.height * 2 + 1
        )
        assert kg.graph_manager.player_idx is not None
