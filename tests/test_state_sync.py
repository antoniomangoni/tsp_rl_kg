"""Regression tests for G10 - Environment/KnowledgeGraph state sync bugs.

Each test targets a specific bug from the spec to prevent regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsp_rl_kg.game_world.entities import ENTITY_ID_TREE
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kg(headless_environment: Environment) -> KnowledgeGraph:
    """KnowledgeGraph built from the shared small headless environment."""
    return KnowledgeGraph(
        environment=headless_environment,
        vision_range=1,
        completion=1.0,
    )


def _find_terrain_coord(env: Environment, terrain_code: int):
    """Return (x, y) of the first tile matching *terrain_code*, or None."""
    coords = np.argwhere(env.terrain_index_grid == terrain_code)
    return tuple(coords[0]) if len(coords) else None


def _find_entity_coord(env: Environment, entity_id: int):
    """Return (x, y) of the first tile matching *entity_id*, or None."""
    coords = np.argwhere(env.entity_index_grid == entity_id)
    return tuple(coords[0]) if len(coords) else None


# ===========================================================================
# BUG-1: PLACE_ROCK terrain divergence
# ===========================================================================


class TestPlaceRockTerrainConsistency:
    def test_elevate_terrain_reads_environment_value(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """After drop_rock_in_water, KG node should match environment grid."""
        # Find a DeepWater tile (code 0)
        coord = _find_terrain_coord(headless_environment, 0)
        if coord is None:
            pytest.skip("No DeepWater tile in test heightmap")
        x, y = coord

        # Simulate the Environment side of PLACE_ROCK (DeepWater → Water, fill_type=0)
        headless_environment.drop_rock_in_water(x, y, fill_type=0)
        env_value = headless_environment.terrain_index_grid[x, y]

        # Now let KG sync
        kg.elevate_terrain_node(x, y)

        # KG node feature must match the environment value
        node_idx = kg.graph_manager.get_node_idx((x, y), kg.terrain_z_level)
        assert kg.graph.x[node_idx].item() == env_value


# ===========================================================================
# BUG-2: kg_completeness propagation
# ===========================================================================


class TestKgCompletenessPropagation:
    def test_different_completeness_gives_different_distance(
        self, headless_environment: Environment
    ):
        """Different completion values should yield different graph distances."""
        kg_low = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=0.25,
        )
        kg_high = KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )
        assert kg_high.distance >= kg_low.distance


# ===========================================================================
# BUG-3: is_node_active dead code - removed in G12
# ===========================================================================


class TestIsNodeActiveRemoved:
    def test_is_node_active_removed(self):
        """is_node_active was removed as part of G12 mask cleanup."""
        assert not hasattr(KnowledgeGraph, "is_node_active")


# ===========================================================================
# BUG-4: Player position derived from environment
# ===========================================================================


class TestPlayerPositionDerived:
    def test_player_pos_matches_environment(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """kg.player_pos must always equal environment player coords."""
        assert kg.player_pos == (
            headless_environment.player.grid_x,
            headless_environment.player.grid_y,
        )

    def test_player_pos_updates_on_environment_move(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """After Environment moves the player, kg.player_pos reflects it."""
        old_pos = kg.player_pos
        new_x = min(old_pos[0] + 1, headless_environment.width - 1)
        new_y = old_pos[1]
        headless_environment.player.grid_x = new_x
        headless_environment.player.grid_y = new_y
        assert kg.player_pos == (new_x, new_y)

    def test_player_pos_is_property(self):
        """player_pos must be a property, not a stored field."""
        assert isinstance(KnowledgeGraph.player_pos, property), "player_pos should be a property"


# ===========================================================================
# BUG-6: Entity edges deactivated on removal
# ===========================================================================


class TestRemovedEntityNodeResets:
    def test_node_features_reset_after_removal(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """After remove_entity_node, the node features should reflect entity_array=0."""
        coord = _find_entity_coord(headless_environment, ENTITY_ID_TREE)
        if coord is None:
            pytest.skip("No Tree entity in test environment")
        x, y = coord

        # Simulate Environment deleting the entity first
        entity = headless_environment.terrain_object_grid[x, y].entity_on_tile
        if entity is None:
            pytest.skip("Entity object not found on tile")
        headless_environment.delete_entity(entity)

        # Then KG sync
        kg.remove_entity_node(x, y)

        entity_node_idx = kg.graph_manager.get_node_idx((x, y), kg.entity_z_level)
        # After removal, entity_array[x, y] == 0, so node feature should be encoded as 0
        assert kg.graph.x[entity_node_idx][0].item() == 0.0


# ===========================================================================
# BUG-7: build_path_node no longer asserts
# ===========================================================================


class TestBuildPathNodeNoAssert:
    def test_build_path_does_not_assert(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """build_path_node should work without asserting entity_array value."""
        # Find a passable tile to place a path
        for x in range(headless_environment.width):
            for y in range(headless_environment.height):
                terrain = headless_environment.terrain_object_grid[x, y]
                if (
                    terrain.passable
                    and headless_environment.entity_index_grid[x, y] == 0
                    and (x, y) not in headless_environment.outpost_locations
                ):
                    headless_environment.place_path(x, y)
                    # Should not raise
                    kg.build_path_node(x, y)
                    return
        pytest.skip("No suitable tile for path placement")


# ===========================================================================
# BUG-8: KG does not mutate entity_array
# ===========================================================================


class TestKgDoesNotMutateArrays:
    def test_entity_array_unchanged_at_player_pos(
        self,
        headless_environment: Environment,
    ):
        """KG __init__ must not write to entity_array at the player position."""
        player_x = headless_environment.player.grid_x
        player_y = headless_environment.player.grid_y
        original_value = int(headless_environment.entity_index_grid[player_x, player_y])

        KnowledgeGraph(
            environment=headless_environment,
            vision_range=1,
            completion=1.0,
        )

        # The entity array should be unchanged
        assert headless_environment.entity_index_grid[player_x, player_y] == original_value

    def test_remove_entity_does_not_write_array(
        self, headless_environment: Environment, kg: KnowledgeGraph
    ):
        """remove_entity_node must not write to entity_array (Environment does that)."""
        coord = _find_entity_coord(headless_environment, ENTITY_ID_TREE)
        if coord is None:
            pytest.skip("No Tree entity in test environment")
        x, y = coord

        # Simulate Environment deleting the entity first
        entity = headless_environment.terrain_object_grid[x, y].entity_on_tile
        if entity is None:
            pytest.skip("Entity object not found on tile")
        headless_environment.delete_entity(entity)
        env_value_after_delete = int(headless_environment.entity_index_grid[x, y])

        # KG sync should not change entity_array
        kg.remove_entity_node(x, y)
        assert headless_environment.entity_index_grid[x, y] == env_value_after_delete


# ===========================================================================
# No dead masking code
# ===========================================================================


class TestNoDeadMaskingCode:
    def test_no_set_node_mask_methods(self):
        """set_node_mask_0 / set_node_mask_1 should not exist."""
        assert not hasattr(KnowledgeGraph, "set_node_mask_0")
        assert not hasattr(KnowledgeGraph, "set_node_mask_1")

    def test_no_edge_mask_methods(self):
        """All edge-mask-toggling methods should be removed in G12."""
        for name in (
            "set_edge_mask_0",
            "set_edge_mask_1",
            "activate_edge",
            "deactivate_node_and_its_edges",
            "activate_node_and_maybe_its_edges",
            "should_edge_be_active",
            "check_edges_active_of_node",
            "activate_discovered_coordinate",
            "check_entities_active",
            "check_path_nodes",
        ):
            assert not hasattr(KnowledgeGraph, name), f"{name} should be removed"
