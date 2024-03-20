import pygame

from environment import Environment
from helper_functions import time_function

class Renderer:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.tile_size = environment.tile_size
        self.surface = pygame.display.set_mode((environment.width * self.tile_size, environment.height * self.tile_size))
        self.terrain_surface = pygame.Surface(self.surface.get_size())
        self.terrain_surface.set_alpha(None)

    def init_render(self):
        # Draw the entire terrain onto the terrain surface
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                terrain_tile = self.environment.terrain_object_grid[x, y]
                self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))
        
        # Initial blit of the terrain surface onto the main surface
        self.surface.blit(self.terrain_surface, (0, 0))
        # Draw all entities for the first time
        self.environment.entity_group.draw(self.surface)
        pygame.display.flip()


    def render_updated_tiles(self):
        if not self.environment.environment_changed_flag:
            return
        # Go through the list of changed tiles and update them
        for x, y in self.environment.changed_tiles_list:
            self.update_tile(x, y)
            # Blit the updated terrain tile onto the main surface
            rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
            self.surface.blit(self.terrain_surface, rect.topleft, rect)
        
        # Now redraw entities that are within or intersect the updated tiles.
        # This is a simplified approach. A more optimized method would check for actual intersections.
        self.environment.entity_group.draw(self.surface)

        # Finally, update the display only for the dirty rects
        dirty_rects = [pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size) for x, y in self.environment.changed_tiles_list]
        pygame.display.update(dirty_rects)

        # Clear the list of changed tiles after updating
        self.environment.changed_tiles_list.clear()
        self.environment.environment_changed_flag = False


    def update_tile(self, x, y):
        # Directly access and redraw the terrain tile
        terrain_tile = self.environment.terrain_object_grid[x, y]
        self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))

        # Redraw the entity if present on this tile
        if terrain_tile.entity_on_tile is not None:
            self.surface.blit(terrain_tile.entity_on_tile.image, (x * self.tile_size, y * self.tile_size))

    # def update_changed_tiles(self):
    #     if self.environment.environment_changed_flag:
    #         # go through the list of changed tiles and update them
    #         for x, y in self.environment.changed_tiles_list:
    #             self.update_tile(x, y)
    #         self.environment.environment_changed_flag = False
    #         self.environment.changed_tiles_list.clear()
