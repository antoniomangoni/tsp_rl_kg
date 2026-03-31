"""Tests for ProjectionPolicy, KHopProjection, and CompletenessProjection."""

from __future__ import annotations

from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.graph.projection import (
    CompletenessProjection,
    FullGraphProjection,
    KHopProjection,
    ProjectionPolicy,
)
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_khop_protocol_conformance():
    assert isinstance(KHopProjection(distance=2), ProjectionPolicy)


def test_completeness_protocol_conformance():
    assert isinstance(CompletenessProjection(0.5, 1, 10), ProjectionPolicy)


# ---------------------------------------------------------------------------
# CompletenessProjection distance property
# ---------------------------------------------------------------------------


class TestCompletenessProjectionDistance:
    def test_distance_property(self):
        proj = CompletenessProjection(completeness=0.5, vision_range=1, grid_width=10)
        assert proj.distance == max(int(0.5 * 10), 1)
        assert proj.distance == 5

    def test_distance_clamped_to_vision_range(self):
        proj = CompletenessProjection(completeness=0.01, vision_range=3, grid_width=10)
        # int(0.01 * 10) = 0, but vision_range=3 is the floor
        assert proj.distance == 3

    def test_completeness_capped_at_one(self):
        proj = CompletenessProjection(completeness=2.0, vision_range=1, grid_width=10)
        assert proj.distance == max(int(1.0 * 10), 1)
        assert proj.distance == 10


# ---------------------------------------------------------------------------
# Integration: projection with real KG
# ---------------------------------------------------------------------------


class TestProjectionIntegration:
    def test_full_projection_returns_all_nodes(self, headless_environment: Environment):
        """With completion=1.0 and sufficient hops, all nodes should be returned.

        After the topology simplification (player connects to one terrain node,
        not all entities), the k-hop distance needs to be large enough to cover
        the entire grid via terrain adjacency.  We use an explicit KHopProjection
        with distance equal to the grid perimeter to guarantee full coverage.
        """
        from tsp_rl_kg.graph.projection import KHopProjection

        w = headless_environment.width
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            projection=KHopProjection(distance=w * 2),
        )
        subgraph = kg.get_subgraph()
        assert subgraph.num_nodes == kg.graph.num_nodes

    def test_partial_projection_returns_subset(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=0.1,
        )
        subgraph = kg.get_subgraph()
        assert subgraph.num_nodes <= kg.graph.num_nodes

    def test_different_completeness_different_subgraphs(self, headless_environment: Environment):
        proj_small = KHopProjection(distance=1)
        proj_full = KHopProjection(distance=headless_environment.width)

        kg_small = KnowledgeGraph(
            environment=headless_environment, vision_range=1, projection=proj_small
        )
        sub_small = kg_small.get_subgraph()

        # Reset discovered_grid for a fresh KG
        headless_environment.discovered_grid[:] = False
        kg_full = KnowledgeGraph(
            environment=headless_environment, vision_range=1, projection=proj_full
        )
        sub_full = kg_full.get_subgraph()

        assert sub_full.num_nodes >= sub_small.num_nodes

    def test_projection_preserves_edge_consistency(self, headless_environment: Environment):
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        subgraph = kg.get_subgraph()
        # Subgraph should have edges and nodes
        assert subgraph.num_nodes > 0
        assert subgraph.num_edges > 0
        # Edge attr should have same count as edges
        assert subgraph.edge_attr.shape[0] == subgraph.num_edges

    def test_explicit_projection_replaces_completion(self, headless_environment: Environment):
        proj = KHopProjection(distance=1)
        kg = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            projection=proj,
        )
        assert kg.projection is proj
        assert kg.distance == 1
        subgraph = kg.get_subgraph()
        assert subgraph.num_nodes > 0


# ---------------------------------------------------------------------------
# FullGraphProjection
# ---------------------------------------------------------------------------


def test_full_graph_projection_protocol_conformance():
    assert isinstance(FullGraphProjection(), ProjectionPolicy)


def test_full_graph_projection_distance_is_none():
    assert FullGraphProjection().distance is None


class TestFullGraphProjectionIntegration:
    def test_returns_all_nodes(self, headless_environment: Environment):
        proj = FullGraphProjection()
        kg = KnowledgeGraph(environment=headless_environment, vision_range=1, projection=proj)
        subgraph = kg.get_subgraph()
        assert subgraph.num_nodes == kg.graph.num_nodes

    def test_returns_all_edges(self, headless_environment: Environment):
        proj = FullGraphProjection()
        kg = KnowledgeGraph(environment=headless_environment, vision_range=1, projection=proj)
        subgraph = kg.get_subgraph()
        assert subgraph.num_edges == kg.graph.num_edges

    def test_preserves_edge_attr(self, headless_environment: Environment):
        proj = FullGraphProjection()
        kg = KnowledgeGraph(environment=headless_environment, vision_range=1, projection=proj)
        subgraph = kg.get_subgraph()
        assert subgraph.edge_attr.shape == kg.graph.edge_attr.shape
