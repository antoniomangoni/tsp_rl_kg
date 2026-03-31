"""Tests for Graph_Manager and KnowledgeGraph."""

from __future__ import annotations

import pytest

from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.graph.graph_idx_manager import Graph_Manager
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph

# ===========================================================================
# Graph_Manager
# ===========================================================================


class TestGraphManagerCreateIdx:
    def test_increments_sequentially(self, graph_manager: Graph_Manager):
        idx0 = graph_manager.create_idx((0, 0), z_level=0)
        idx1 = graph_manager.create_idx((1, 0), z_level=0)
        idx2 = graph_manager.create_idx((0, 0), z_level=1)
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_get_node_idx_returns_stored(self, graph_manager: Graph_Manager):
        expected = graph_manager.create_idx((3, 4), z_level=1)
        assert graph_manager.get_node_idx((3, 4), z_level=1) == expected

    def test_get_node_idx_missing_returns_none(self, graph_manager: Graph_Manager):
        assert graph_manager.get_node_idx((99, 99), z_level=0) is None


class TestGraphManagerEdges:
    def test_create_edge_idx_creates_bidirectional(self, graph_manager: Graph_Manager):
        graph_manager.create_idx((0, 0), 0)
        graph_manager.create_idx((1, 0), 0)
        direct, reverse = graph_manager.create_edge_idx(0, 1)
        assert direct == 0
        assert reverse == 1

    def test_retrieve_edge_indices(self, graph_manager: Graph_Manager):
        graph_manager.create_idx((0, 0), 0)
        graph_manager.create_idx((1, 0), 0)
        graph_manager.create_edge_idx(0, 1)
        direct, reverse = graph_manager.retrieve_edge_indices(0, 1)
        assert direct == 0
        assert reverse == 1

    def test_overflow_raises(self):
        gm = Graph_Manager()
        gm.set_max_nodes(10)
        gm.set_max_edges(2)  # Only room for 1 edge pair
        gm.create_idx((0, 0), 0)
        gm.create_idx((1, 0), 0)
        gm.create_idx((2, 0), 0)
        gm.create_edge_idx(0, 1)  # Uses indices 0, 1
        with pytest.raises(RuntimeError, match="Max edges"):
            gm.create_edge_idx(1, 2)  # Would need indices 2, 3 but max is 2

    def test_retrieve_edges_from_node(self, graph_manager: Graph_Manager):
        graph_manager.create_idx((0, 0), 0)
        graph_manager.create_idx((1, 0), 0)
        graph_manager.create_idx((2, 0), 0)
        graph_manager.create_edge_idx(0, 1)
        graph_manager.create_edge_idx(0, 2)
        pairs = graph_manager.retrieve_edge_node_pairs_from_node(0)
        # Node 0 is in edges (0,1), (1,0), (0,2), (2,0)
        assert len(pairs) == 4


# ===========================================================================
# KnowledgeGraph
# ===========================================================================


class TestKnowledgeGraphInit:
    def test_complete_graph_populates_nodes(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        # All node features should be initialised (no -1 placeholders)
        total_nodes = kg.num_possible_nodes
        assert kg.graph.x.shape[0] == total_nodes

    def test_node_count_matches_grid(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        w, h = headless_environment.width, headless_environment.height
        # 2 z-levels (terrain, entity) per tile + 1 player node
        expected = w * h * 2 + 1
        assert kg.num_possible_nodes == expected
        assert kg.graph_manager.node_idx == expected


class TestKnowledgeGraphDiscovery:
    def test_discover_coordinate(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=0.1,  # Small initial discovery
        )
        # Pick a coordinate outside initial discovery
        x, y = headless_environment.width - 1, headless_environment.height - 1
        headless_environment.discovered_grid[x, y] = False  # Force undiscovered
        headless_environment.discover_coordinate(x, y)
        kg.activate_discovered_coordinate(x, y)
        assert headless_environment.discovered_grid[x, y]

    def test_discover_already_discovered_returns_false(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        # All coordinates are already discovered at completion=1.0
        px, py = kg.player_pos
        result = headless_environment.discover_coordinate(px, py)
        assert result is False


class TestKnowledgeGraphPlayerNode:
    def test_move_player_node(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        old_pos = kg.player_pos
        # Move to a different valid position
        new_x = min(old_pos[0] + 1, headless_environment.width - 1)
        new_y = min(old_pos[1] + 1, headless_environment.height - 1)
        # Must move via Environment so the property reflects the change
        headless_environment.player.grid_x = new_x
        headless_environment.player.grid_y = new_y
        kg.move_player_node(new_x, new_y)
        assert kg.player_pos == (new_x, new_y)


class TestKnowledgeGraphEdges:
    def test_terrain_edges_created(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        # Every adjacent terrain pair should have edges
        w, h = headless_environment.width, headless_environment.height
        expected_terrain_edges = 2 * (w * (h - 1) + h * (w - 1))
        # Terrain edges are created in the graph; total edges also include entity + player edges
        entity_edges = w * h * 2
        player_edges = 2
        assert kg.num_possible_edges == expected_terrain_edges + entity_edges + player_edges
