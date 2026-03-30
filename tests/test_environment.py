"""Tests for headless Environment in tsp_rl_kg.game_world.environment."""

from __future__ import annotations

import numpy as np
import pytest

from tsp_rl_kg.game_world.entities import ENTITY_ID_OUTPOST, ENTITY_ID_PLAYER, ENTITY_ID_WOOD_PATH
from tsp_rl_kg.game_world.environment import Environment, _HeadlessEntityGroup

# ---------------------------------------------------------------------------
# Terrain grid initialisation
# ---------------------------------------------------------------------------


class TestEnvironmentInit:
    def test_terrain_grid_shape(self, headless_environment: Environment):
        h, w = headless_environment.heightmap.shape
        assert headless_environment.terrain_index_grid.shape == (h, w)

    def test_terrain_grid_values(self, headless_environment: Environment):
        unique = set(np.unique(headless_environment.terrain_index_grid))
        # All values should be valid terrain codes (0–5)
        assert unique.issubset({0, 1, 2, 3, 4, 5})

    def test_terrain_object_grid_populated(self, headless_environment: Environment):
        for (x, y), obj in np.ndenumerate(headless_environment.terrain_object_grid):
            assert obj is not None, f"Terrain object missing at ({x}, {y})"

    def test_terrain_types_from_heightmap(self, small_heightmap: np.ndarray):
        env = Environment(
            heightmap=small_heightmap, tile_size=50, number_of_outposts=0, headless=True
        )
        # Corner (0,0) has heightmap value 2 → Plains (elevation 2)
        assert env.terrain_object_grid[0, 0].elevation == 2
        # Corner (4,0) has heightmap value 0 → DeepWater (elevation 0)
        assert env.terrain_object_grid[4, 0].elevation == 0
        # Corner (0,4) has heightmap value 5 → Snow (elevation 5)
        assert env.terrain_object_grid[0, 4].elevation == 5


# ---------------------------------------------------------------------------
# Outposts
# ---------------------------------------------------------------------------


class TestOutposts:
    def test_adds_correct_number(self, small_heightmap: np.ndarray):
        env = Environment(
            heightmap=small_heightmap, tile_size=50, number_of_outposts=2, headless=True
        )
        assert len(env.outpost_locations) == 2

    def test_outpost_locations_within_bounds(self, headless_environment: Environment):
        for x, y in headless_environment.outpost_locations:
            assert headless_environment.within_bounds(x, y)

    def test_outpost_entity_id_in_grid(self, headless_environment: Environment):
        for x, y in headless_environment.outpost_locations:
            assert headless_environment.entity_index_grid[x, y] == ENTITY_ID_OUTPOST


# ---------------------------------------------------------------------------
# Player init
# ---------------------------------------------------------------------------


class TestPlayerInit:
    def test_player_exists(self, headless_environment: Environment):
        assert headless_environment.player is not None
        assert headless_environment.player.id == ENTITY_ID_PLAYER

    def test_player_within_bounds(self, headless_environment: Environment):
        px, py = headless_environment.player.grid_x, headless_environment.player.grid_y
        assert headless_environment.within_bounds(px, py)


# ---------------------------------------------------------------------------
# within_bounds
# ---------------------------------------------------------------------------


class TestWithinBounds:
    def test_valid_origin(self, headless_environment: Environment):
        assert headless_environment.within_bounds(0, 0) is True

    def test_valid_max(self, headless_environment: Environment):
        w, h = headless_environment.width, headless_environment.height
        assert headless_environment.within_bounds(w - 1, h - 1) is True

    def test_negative_x(self, headless_environment: Environment):
        assert headless_environment.within_bounds(-1, 0) is False

    def test_negative_y(self, headless_environment: Environment):
        assert headless_environment.within_bounds(0, -1) is False

    def test_overflow_x(self, headless_environment: Environment):
        assert headless_environment.within_bounds(headless_environment.width, 0) is False

    def test_overflow_y(self, headless_environment: Environment):
        assert headless_environment.within_bounds(0, headless_environment.height) is False


# ---------------------------------------------------------------------------
# move_entity
# ---------------------------------------------------------------------------


class TestMoveEntity:
    def test_move_player(self, headless_environment: Environment):
        player = headless_environment.player
        old_x, old_y = player.grid_x, player.grid_y
        # Find a passable neighbour
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = old_x + dx, old_y + dy
            if headless_environment.is_move_valid(nx, ny):
                new_x, new_y = headless_environment.move_entity(player, dx, dy)
                assert (new_x, new_y) == (nx, ny)
                assert player.grid_x == nx
                assert player.grid_y == ny
                return
        pytest.skip("No passable neighbour found for player")

    def test_move_entity_blocked(self, headless_environment: Environment):
        player = headless_environment.player
        old_x, old_y = player.grid_x, player.grid_y
        # Try to move out of bounds
        new_x, new_y = headless_environment.move_entity(player, -old_x - 1, 0)
        # Should stay at original position
        assert (new_x, new_y) == (old_x, old_y)


# ---------------------------------------------------------------------------
# place_path
# ---------------------------------------------------------------------------


class TestPlacePath:
    def test_place_path(self, headless_environment: Environment):
        # Find an empty passable tile
        for x in range(headless_environment.width):
            for y in range(headless_environment.height):
                terrain = headless_environment.terrain_object_grid[x, y]
                if terrain.passable and headless_environment.entity_index_grid[x, y] == 0:
                    headless_environment.place_path(x, y)
                    assert headless_environment.entity_index_grid[x, y] == ENTITY_ID_WOOD_PATH
                    assert terrain.entity_on_tile is not None
                    return
        pytest.skip("No empty passable tile found")


# ---------------------------------------------------------------------------
# Headless mode specifics
# ---------------------------------------------------------------------------


class TestHeadlessMode:
    def test_entity_group_is_headless(self, headless_environment: Environment):
        assert isinstance(headless_environment.entity_group, _HeadlessEntityGroup)

    def test_headless_flag_propagated(self, headless_environment: Environment):
        from tsp_rl_kg.game_world.entities import Entity
        from tsp_rl_kg.game_world.terrains import Terrain

        assert Entity._headless is True
        assert Terrain._headless is True
