import pygame

from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.game_world.environment import Environment


class Renderer:
    def __init__(self, environment: Environment, agent_control: Agent):
        self.environment = environment
        self.agent = agent_control  # Reference to the agent to access its status and inventory
        self.tile_size = environment.tile_size
        self.window_width = environment.width * self.tile_size
        self.window_height = environment.height * self.tile_size

        self.heatmap_colour = (255, 0, 0)  # Colour for the heatmap overlay is red

        self.surface = pygame.display.set_mode((self.window_width, self.window_height))

        self.terrain_surface = pygame.Surface((self.window_width, self.window_height))
        self.terrain_surface.set_alpha(None)

        # Reusable black tile for undiscovered (fog-of-war) areas
        self.fog_tile = pygame.Surface((self.tile_size, self.tile_size))
        self.fog_tile.fill((0, 0, 0))

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
        # Draw entities only on discovered tiles
        self.environment.entity_group.draw(self.surface)
        self._overdraw_fog_on_entities()
        pygame.display.flip()

    def render_updated_tiles(self):
        if not self.environment.environment_changed_flag:
            return

        # Go through the list of changed tiles and update them
        for x, y in self.environment.changed_tiles_list:
            self.update_tile(x, y)
            # Blit the updated terrain tile onto the main surface
            rect = pygame.Rect(
                x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size
            )
            self.surface.blit(self.terrain_surface, rect.topleft, rect)

        # Redraw entities, then overdraw fog on any undiscovered tiles
        # to hide entity sprites that entity_group.draw() renders globally.
        self.environment.entity_group.draw(self.surface)
        self._overdraw_fog_on_entities()

        # Finally, update the display only for the dirty rects
        dirty_rects = [
            pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
            for x, y in self.environment.changed_tiles_list
        ]
        pygame.display.update(dirty_rects)

        # Clear the list of changed tiles after updating
        self.environment.changed_tiles_list.clear()
        self.environment.environment_changed_flag = False

    def update_tile(self, x, y):
        pos = (x * self.tile_size, y * self.tile_size)
        if not self.environment.discovered_grid[x, y]:
            self.terrain_surface.blit(self.fog_tile, pos)
            return

        # Directly access and redraw the terrain tile
        terrain_tile = self.environment.terrain_object_grid[x, y]
        self.terrain_surface.blit(terrain_tile.image, pos)

        # Redraw the entity if present on this tile
        if terrain_tile.entity_on_tile is not None:
            self.surface.blit(terrain_tile.entity_on_tile.image, pos)

    def _overdraw_fog_on_entities(self):
        """Cover entity sprites on undiscovered tiles with fog."""
        discovered = self.environment.discovered_grid
        for sprite in self.environment.entity_group:
            gx, gy = sprite.grid_x, sprite.grid_y
            if not discovered[gx, gy]:
                self.surface.blit(
                    self.fog_tile, (gx * self.tile_size, gy * self.tile_size)
                )

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
                    heat_rect = pygame.Surface((self.tile_size, self.tile_size))
                    heat_rect.set_alpha(alpha)
                    heat_rect.fill(color[:3])
                    self.surface.blit(heat_rect, (x * self.tile_size, y * self.tile_size))
        pygame.display.update()
