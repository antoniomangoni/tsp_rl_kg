import pygame
from terrain_manager import TerrainManager
from entity_manager import EntityManager
from helper_functions import time_function

class Renderer:
    @time_function
    def __init__(self, terrain_manager: TerrainManager, entity_manager: EntityManager, tile_size: int = 50):
        self.terrain_manager = terrain_manager
        self.entity_manager = entity_manager
        self.tile_size = tile_size
        self.surface = pygame.display.set_mode((terrain_manager.width * tile_size, terrain_manager.height * tile_size))
        self.terrain_surface = pygame.Surface(self.surface.get_size())
        self.terrain_surface.set_alpha(None)
        self.terrain_needs_update = True

    @time_function
    def render_terrain(self):
        if self.terrain_needs_update:
            print("Rendering terrain")
            for x in range(self.terrain_manager.width):
                for y in range(self.terrain_manager.height):
                    terrain = self.terrain_manager.heightmap[x, y]
                    color = self.terrain_manager.get_terrain_color(terrain)
                    print(f"Terrain color at ({x}, {y}): {color}")
                    pygame.draw.rect(self.terrain_surface, color, (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))
            self.terrain_needs_update = False

    @time_function
    def render(self):
        self.surface.blit(self.terrain_surface, (0, 0))
        print(f"Number of entities to render: {len(self.entity_manager.entity_group)}")
        self.entity_manager.entity_group.draw(self.surface)
        pygame.display.flip()
