import pygame

from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.game_world.environment import Environment


class Renderer:
    HUD_HEIGHT = 72
    HUD_PADDING = 8
    HUD_TEXT_COLOUR = (240, 240, 240)
    HUD_BACKGROUND_COLOUR = (24, 24, 24)
    HUD_FONT_SIZE = 22

    def __init__(self, environment: Environment, agent_control: Agent):
        self.environment = environment
        self.agent = agent_control  # Reference to the agent to access its status and inventory
        self.tile_size = environment.tile_size
        self.game_area_width = environment.width * self.tile_size
        self.game_area_height = environment.height * self.tile_size
        self.window_width = self.game_area_width
        self.window_height = self.game_area_height + self.HUD_HEIGHT
        self.hud_top = self.game_area_height

        self.heatmap_colour = (255, 0, 0)  # Colour for the heatmap overlay is red

        self.surface = pygame.display.set_mode((self.window_width, self.window_height))

        self.terrain_surface = pygame.Surface((self.game_area_width, self.game_area_height))
        self.terrain_surface.set_alpha(None)
        self.hud_surface = pygame.Surface((self.window_width, self.HUD_HEIGHT))
        self.hud_surface.fill(self.HUD_BACKGROUND_COLOUR)
        self.font = pygame.font.Font(None, self.HUD_FONT_SIZE)

        # Reusable black tile for undiscovered (fog-of-war) areas
        self.fog_tile = pygame.Surface((self.tile_size, self.tile_size))
        self.fog_tile.fill((0, 0, 0))

        self._heat_surface = pygame.Surface((self.tile_size, self.tile_size))

    def init_render(self):
        discovered = self.environment.discovered_grid
        # Draw terrain or fog onto the terrain surface
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                pos = (x * self.tile_size, y * self.tile_size)
                if discovered[x, y]:
                    terrain_tile = self.environment.terrain_object_grid[x, y]
                    self.terrain_surface.blit(terrain_tile.image, pos)
                else:
                    self.terrain_surface.blit(self.fog_tile, pos)

        # Initial blit of the terrain surface onto the main surface
        self.surface.blit(self.terrain_surface, (0, 0))
        # Initial draw still uses the sprite group because this is a one-time full render.
        self.environment.entity_group.draw(self.surface)
        for sprite in self.environment.entity_group:
            gx, gy = sprite.grid_x, sprite.grid_y
            if not discovered[gx, gy]:
                self.surface.blit(self.fog_tile, (gx * self.tile_size, gy * self.tile_size))
        pygame.display.flip()

    def render_updated_tiles(self):
        if not self.environment.environment_changed_flag:
            return

        changed_tiles = self.environment.changed_tiles

        # Go through the list of changed tiles and update them
        for x, y in changed_tiles:
            self.update_tile(x, y)
            # Blit the updated terrain tile onto the main surface
            rect = pygame.Rect(
                x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size
            )
            self.surface.blit(self.terrain_surface, rect.topleft, rect)

            terrain_tile = self.environment.terrain_object_grid[x, y]
            if self.environment.discovered_grid[x, y] and terrain_tile.entity_on_tile is not None:
                self.surface.blit(terrain_tile.entity_on_tile.image, rect.topleft)

        # Finally, update the display only for the dirty rects
        dirty_rects = [
            pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
            for x, y in changed_tiles
        ]
        pygame.display.update(dirty_rects)

        # Clear the list of changed tiles after updating
        changed_tiles.clear()
        self.environment.environment_changed_flag = False

    def render_ui(self, status_rows: list[dict[str, str | int | float]]):
        self.hud_surface.fill(self.HUD_BACKGROUND_COLOUR)
        line_height = self.font.get_linesize()
        for row_index, row in enumerate(status_rows):
            status_parts = [f"{key}: {value}" for key, value in row.items()]
            status_text = "   |   ".join(status_parts)
            text_surface = self.font.render(status_text, True, self.HUD_TEXT_COLOUR)
            y = self.HUD_PADDING + row_index * line_height
            self.hud_surface.blit(text_surface, (self.HUD_PADDING, y))
        self.surface.blit(self.hud_surface, (0, self.hud_top))
        pygame.display.update(
            pygame.Rect(0, self.hud_top, self.window_width, self.HUD_HEIGHT),
        )

    def update_tile(self, x, y):
        pos = (x * self.tile_size, y * self.tile_size)
        if not self.environment.discovered_grid[x, y]:
            self.terrain_surface.blit(self.fog_tile, pos)
            return

        # Directly access and redraw the terrain tile
        terrain_tile = self.environment.terrain_object_grid[x, y]
        self.terrain_surface.blit(terrain_tile.image, pos)

    def render_heatmap(self, max_intensity, bool_heatmap=False):
        if not bool_heatmap:
            return

        discovered = self.environment.discovered_grid
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                if not discovered[x, y]:
                    continue
                intensity = self.environment.heat_map[x, y]
                if intensity > 0:
                    alpha = int((intensity / max_intensity) * 255)  # Scale intensity to 0-255 range
                    color = (*self.heatmap_colour[:3], alpha)  # Add alpha to the heatmap color
                    self._heat_surface.set_alpha(alpha)
                    self._heat_surface.fill(color[:3])
                    self.surface.blit(self._heat_surface, (x * self.tile_size, y * self.tile_size))
        pygame.display.update()
