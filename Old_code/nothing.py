import pygame
from environment import Environment
from helper_functions import time_function

class Renderer:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.tile_size = environment.tile_size
        self.surface = pygame.display.set_mode((environment.width * self.tile_size, environment.height * self.tile_size))
        self.terrain_surface = pygame.Surface(self.surface.get_size()).convert()
        self.dirty_rects = []  # List to hold rectangles that need updating

    def init_terrain(self):
        """Initializes the terrain and pre-renders it to a separate surface."""
        self.terrain_surface.fill((0, 0, 0))  # Assuming a black background, adjust as necessary
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                terrain_tile = self.environment.terrain_object_grid[x, y]
                rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                self.terrain_surface.blit(terrain_tile.image, rect.topleft)
                self.dirty_rects.append(rect)
        self.surface.blit(self.terrain_surface, (0, 0))  # Initial blit to the main surface
        pygame.display.update(self.dirty_rects)  # Update the entire screen initially
        self.dirty_rects.clear()

    @time_function
    def render(self):
        """Renders the current game state to the screen, updating only dirty rects."""
        # Render the terrain (this example assumes it doesn't change frequently)
        # If your terrain changes dynamically, you'd update and add dirty rects here as well.

        # Render entities on top of the terrain and collect dirty rects
        for entity in self.environment.entity_group.sprites():
            rect = entity.rect
            self.surface.blit(self.terrain_surface, rect.topleft, rect)  # Redraw terrain background
            self.surface.blit(entity.image, rect.topleft)  # Draw the entity
            self.dirty_rects.append(rect)

        # Update the screen for only the dirty rectangles
        pygame.display.update(self.dirty_rects)
        self.dirty_rects.clear()  # Clear the list of dirty rectangles after updating

    def update_tile(self, x, y):
        """Updates a specific tile, adding its rectangle to the dirty list."""
        terrain_tile = self.environment.terrain_object_grid[x, y]
        rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
        self.terrain_surface.blit(terrain_tile.image, rect.topleft)
        self.dirty_rects.append(rect)

        # Optionally, redraw entity here and add to dirty rects if an entity is on this tile
        # This method can be modified based on your game's specific needs.

    def update_changed_tiles(self):
        """Updates tiles that have changed, leveraging the dirty rectangles method."""
        if self.environment.environment_changed_flag:
            for x, y in self.environment.changed_tiles_list:
                self.update_tile(x, y)
            self.environment.environment_changed_flag = False
            self.environment.changed_tiles_list.clear()

