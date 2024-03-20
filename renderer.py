import pygame
import numpy as np

from environment import Environment

class Renderer:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.tile_size = environment.tile_size
        self.surface = pygame.display.set_mode((environment.width * self.tile_size, environment.height * self.tile_size))
        self.terrain_surface = pygame.Surface(self.surface.get_size())
        self.terrain_surface.set_alpha(None)

    def init_terrain(self):
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                terrain_tile = self.environment.terrain_object_grid[x, y]
                self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))

    def render(self):
        self.surface.blit(self.terrain_surface, (0, 0))
        self.environment.entity_group.draw(self.surface)
        pygame.display.flip()

    def update_tile(self, x, y):
        # Directly access and redraw the terrain tile
        terrain_tile = self.environment.terrain_object_grid[x, y]
        self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))

        # Redraw the entity if present on this tile
        if terrain_tile.entity_on_tile is not None:
            self.surface.blit(terrain_tile.entity_on_tile.image, (x * self.tile_size, y * self.tile_size))

    def update_changed_tiles(self):
        if self.environment.environment_changed_flag:
            # go through the list of changed tiles and update them
            for x, y in self.environment.changed_tiles_list:
                self.update_tile(x, y)
            self.environment.environment_changed_flag = False
            self.environment.changed_tiles_list = []
