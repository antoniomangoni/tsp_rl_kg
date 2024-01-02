import pygame
import numpy as np
import random
from typing import Dict, Tuple, Type

from terrain_manager import TerrainManager
from entity_manager import EntityManager

class Renderer:
    def __init__(self, terrain_manager: TerrainManager, entity_manager: EntityManager, tile_size: int = 50):
        self.terrain_manager = terrain_manager
        self.entity_manager = entity_manager
        self.tile_size = tile_size
        self.surface = pygame.display.set_mode((terrain_manager.width * tile_size, terrain_manager.height * tile_size))

    def render(self):
        self.surface.fill((0, 0, 0))
        for x in range(self.terrain_manager.width):
            for y in range(self.terrain_manager.height):
                terrain = self.terrain_manager.heightmap[x, y]
                color = self.terrain_manager.get_terrain_color(terrain)
                pygame.draw.rect(self.surface, color, (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))

        self.entity_manager.entity_group.draw(self.surface)
        pygame.display.flip()

    def real_time_update(self, update_interval=50):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render()
            clock.tick(update_interval)
        pygame.quit()

