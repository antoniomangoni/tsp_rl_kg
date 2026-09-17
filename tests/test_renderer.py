"""Regression tests for the dirty-rect, fog-of-war aware Renderer (spec g19).

These run pygame against the SDL ``dummy`` video driver so they need no display.
Pixel assertions compare a rendered tile with the same images composited by hand,
which keeps the expectations independent of the exact sprite artwork.
"""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from tsp_rl_kg.game_world.entities import (
    ENTITY_ID_OUTPOST,
    ENTITY_ID_TREE,
    Outpost,
    Tree,
    WoodPath,
)
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.game_world.world_template import WorldTemplate
from tsp_rl_kg.renderer import Renderer

TILE_SIZE = 10
GRID_SIZE = 5
SPAWN = (1, 1)
TREE_TILE = (3, 3)
OUTPOST_TILE = (0, 4)
PLAINS = 2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_display(monkeypatch: pytest.MonkeyPatch):
    """Initialise pygame against the headless SDL drivers for the duration of a test."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.display.quit()
    pygame.init()
    yield
    pygame.display.quit()


@pytest.fixture
def world_template() -> WorldTemplate:
    """All-plains 5x5 world with a tree far from spawn and an outpost in a corner."""
    terrain = tuple(tuple(PLAINS for _ in range(GRID_SIZE)) for _ in range(GRID_SIZE))
    entities = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    entities[TREE_TILE[0]][TREE_TILE[1]] = ENTITY_ID_TREE
    entities[OUTPOST_TILE[0]][OUTPOST_TILE[1]] = ENTITY_ID_OUTPOST
    return WorldTemplate(
        terrain,
        tuple(tuple(row) for row in entities),
        SPAWN,
        (OUTPOST_TILE,),
    )


@pytest.fixture
def environment(dummy_display, world_template: WorldTemplate) -> Environment:
    return world_template.restore(tile_size=TILE_SIZE, headless=False)


@pytest.fixture
def renderer(environment: Environment) -> Renderer:
    """Renderer after the initial full draw, with a radius-1 area around spawn discovered."""
    renderer = Renderer(environment, agent_control=None)
    environment.init_discovered_area(SPAWN, radius=1)
    renderer.init_render()
    return renderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tile_pixels(surface: pygame.Surface, x: int, y: int) -> np.ndarray:
    rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    return pygame.surfarray.array3d(surface.subsurface(rect)).copy()


def composite(*images: pygame.Surface) -> np.ndarray:
    """Pixels of *images* blitted on top of each other, first image at the bottom."""
    out = images[0].copy()
    for image in images[1:]:
        out.blit(image, (0, 0))
    return pygame.surfarray.array3d(out)


def assert_tile_is_fog(surface: pygame.Surface, x: int, y: int) -> None:
    assert not tile_pixels(surface, x, y).any(), f"tile {(x, y)} should be fully black fog"


def assert_tile_shows(surface: pygame.Surface, x: int, y: int, *images: pygame.Surface) -> None:
    assert np.array_equal(
        tile_pixels(surface, x, y), composite(*images)
    ), f"tile {(x, y)} does not match the expected composite"


def terrain_image(environment: Environment, x: int, y: int) -> pygame.Surface:
    return environment.terrain_object_grid[x, y].image


# ---------------------------------------------------------------------------
# Initial render
# ---------------------------------------------------------------------------


class TestInitRender:
    def test_discovered_tiles_show_terrain_and_player(
        self, environment: Environment, renderer: Renderer
    ):
        px, py = SPAWN
        assert_tile_shows(
            renderer.surface, px, py, terrain_image(environment, px, py), environment.player.image
        )
        assert_tile_shows(renderer.surface, 2, 2, terrain_image(environment, 2, 2))

    def test_entities_on_undiscovered_tiles_are_hidden(
        self, environment: Environment, renderer: Renderer
    ):
        assert isinstance(environment.terrain_object_grid[TREE_TILE].entity_on_tile, Tree)
        assert isinstance(environment.terrain_object_grid[OUTPOST_TILE].entity_on_tile, Outpost)
        assert_tile_is_fog(renderer.surface, *TREE_TILE)
        assert_tile_is_fog(renderer.surface, *OUTPOST_TILE)


# ---------------------------------------------------------------------------
# Dirty-tile updates
# ---------------------------------------------------------------------------


class TestRenderUpdatedTiles:
    def test_discovering_a_tile_reveals_its_entity(
        self, environment: Environment, renderer: Renderer
    ):
        assert environment.discover_coordinate(*TREE_TILE)
        assert environment.changed_tiles == {TREE_TILE}

        renderer.render_updated_tiles()

        tree = environment.terrain_object_grid[TREE_TILE].entity_on_tile
        assert_tile_shows(
            renderer.surface, *TREE_TILE, terrain_image(environment, *TREE_TILE), tree.image
        )

    def test_render_clears_change_tracking(self, environment: Environment, renderer: Renderer):
        environment.discover_coordinate(*TREE_TILE)

        renderer.render_updated_tiles()

        assert environment.changed_tiles == set()
        assert environment.environment_changed_flag is False

    def test_changed_but_undiscovered_tile_stays_fogged(
        self, environment: Environment, renderer: Renderer
    ):
        environment.single_environment_changed(*TREE_TILE)

        renderer.render_updated_tiles()

        assert_tile_is_fog(renderer.surface, *TREE_TILE)

    def test_no_op_when_nothing_changed(self, environment: Environment, renderer: Renderer):
        before = pygame.surfarray.array3d(renderer.surface).copy()

        renderer.render_updated_tiles()

        assert np.array_equal(pygame.surfarray.array3d(renderer.surface), before)

    def test_player_move_clears_old_tile_and_draws_new_tile(
        self, environment: Environment, renderer: Renderer
    ):
        old_x, old_y = SPAWN
        new_x, new_y = environment.move_entity(environment.player, 1, 0)
        assert (new_x, new_y) == (old_x + 1, old_y)
        assert environment.changed_tiles == {(old_x, old_y), (new_x, new_y)}

        renderer.render_updated_tiles()

        assert_tile_shows(renderer.surface, old_x, old_y, terrain_image(environment, old_x, old_y))
        assert_tile_shows(
            renderer.surface,
            new_x,
            new_y,
            terrain_image(environment, new_x, new_y),
            environment.player.image,
        )

    def test_placed_path_survives_player_leaving_the_tile(
        self, environment: Environment, renderer: Renderer
    ):
        px, py = SPAWN
        environment.place_path(px, py)
        renderer.render_updated_tiles()
        path = environment.terrain_object_grid[px, py].entity_on_tile
        assert isinstance(path, WoodPath)
        assert_tile_shows(
            renderer.surface,
            px,
            py,
            terrain_image(environment, px, py),
            path.image,
            environment.player.image,
        )

        environment.move_entity(environment.player, 0, 1)
        renderer.render_updated_tiles()

        assert_tile_shows(renderer.surface, px, py, terrain_image(environment, px, py), path.image)

    def test_does_not_redraw_the_whole_entity_group(
        self, monkeypatch: pytest.MonkeyPatch, environment: Environment, renderer: Renderer
    ):
        group_draws: list[pygame.Surface] = []
        monkeypatch.setattr(environment.entity_group, "draw", group_draws.append)
        environment.discover_coordinate(*TREE_TILE)

        renderer.render_updated_tiles()

        assert group_draws == []

    def test_no_full_group_fog_overdraw_pass(self):
        assert not hasattr(Renderer, "_overdraw_fog_on_entities")


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------


class TestRenderHeatmap:
    def test_reuses_preallocated_surface_and_tints_discovered_tiles(
        self, monkeypatch: pytest.MonkeyPatch, environment: Environment, renderer: Renderer
    ):
        environment.heat_map = np.zeros((GRID_SIZE, GRID_SIZE))
        environment.heat_map[2, 2] = 4.0  # discovered, full intensity
        environment.heat_map[TREE_TILE] = 4.0  # undiscovered, must stay fogged
        allocations: list[tuple] = []
        real_surface = pygame.Surface

        def counting_surface(*args, **kwargs):
            allocations.append(args)
            return real_surface(*args, **kwargs)

        monkeypatch.setattr(pygame, "Surface", counting_surface)

        renderer.render_heatmap(max_intensity=4.0, bool_heatmap=True)

        assert allocations == []
        assert (tile_pixels(renderer.surface, 2, 2) == renderer.heatmap_colour).all()
        assert_tile_is_fog(renderer.surface, *TREE_TILE)

    def test_disabled_heatmap_leaves_surface_untouched(
        self, environment: Environment, renderer: Renderer
    ):
        before = pygame.surfarray.array3d(renderer.surface).copy()

        renderer.render_heatmap(max_intensity=1.0, bool_heatmap=False)

        assert np.array_equal(pygame.surfarray.array3d(renderer.surface), before)


# ---------------------------------------------------------------------------
# HUD status rows
# ---------------------------------------------------------------------------


class TestRenderUi:
    STATUS_ROWS = [
        {"X": 1, "Y": 1, "Energy": 0, "Outposts": "0/1"},
        {"Wood": "0/5", "Stone": "0/5", "Target route": 12.5, "Current route": 0},
    ]

    @staticmethod
    def hud_band(renderer: Renderer) -> np.ndarray:
        rect = pygame.Rect(0, renderer.hud_top, renderer.window_width, Renderer.HUD_HEIGHT)
        return pygame.surfarray.array3d(renderer.surface.subsurface(rect)).copy()

    def test_draws_status_rows_into_hud_band(self, renderer: Renderer):
        renderer.render_ui(self.STATUS_ROWS)

        band = self.hud_band(renderer)
        is_background = (band == np.array(Renderer.HUD_BACKGROUND_COLOUR)).all(axis=-1)
        assert is_background.any(), "HUD band should be painted with the HUD background"
        assert not is_background.all(), "status text should be visible on the HUD band"

    def test_leaves_game_area_untouched(self, renderer: Renderer):
        game_rect = pygame.Rect(0, 0, renderer.game_area_width, renderer.game_area_height)
        before = pygame.surfarray.array3d(renderer.surface.subsurface(game_rect)).copy()

        renderer.render_ui(self.STATUS_ROWS)

        after = pygame.surfarray.array3d(renderer.surface.subsurface(game_rect))
        assert np.array_equal(after, before)
