import pygame
import numpy as np
import random
from typing import Dict, Tuple, Type
from entities import Tree, MossyRock, SnowyRock, Fish

class TerrainRenderer:
    def __init__(self, heightmap: np.ndarray,
                 colors: Dict[int, Tuple[int, int, int]],
                 entity_map: Dict[str, int],
                 entity_class_map: Dict[str, Type[pygame.sprite.Sprite]],
                 entity_spawn_probabilities: Dict[str, float],
                 tile_size: int = 50):
        
        pygame.init()
        self.heightmap = heightmap
        self.colors = colors
        self.entity_map = entity_map
        self.entity_class_map = entity_class_map
        self.spawn_probs = entity_spawn_probabilities
        self.width, self.height = self.heightmap.shape
        self.tile_size = tile_size
        self.surface = pygame.display.set_mode((self.width * self.tile_size, self.height * self.tile_size))
        self.entity_group = pygame.sprite.Group()
        self.populate_tiles()

    def create_entity(self, entity_type: str, x: int, y: int):
        if entity_type not in self.entity_class_map or random.random() > self.spawn_probs.get(entity_type, 1):
            return

        pixel_x, pixel_y = x * self.tile_size, y * self.tile_size
        entity_class = self.entity_class_map[entity_type]
        entity = entity_class(pixel_x, pixel_y, pixel_size=self.tile_size)
        self.entity_group.add(entity)

    def populate_tiles(self):
        self.entity_group.empty()
        for x in range(self.width):
            for y in range(self.height):
                terrain_code = self.heightmap[x, y]
                entity_type = self.get_entity_type_from_terrain(terrain_code)
                if entity_type:
                    self.create_entity(entity_type, x, y)

    def get_entity_type_from_terrain(self, terrain_code: int) -> str:
        inverse_map = {v: k for k, v in self.entity_map.items()}
        return inverse_map.get(terrain_code)

    def render(self):
        self.surface.fill((0, 0, 0))
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                color = self.colors[terrain]
                flipped_y = self.height - 1 - y
                pygame.draw.rect(self.surface, color, (x * self.tile_size, flipped_y * self.tile_size, self.tile_size, self.tile_size))
        
        self.entity_group.draw(self.surface)
        pygame.display.flip()

    def update_heightmap(self, new_heightmap: np.ndarray):
        self.heightmap = new_heightmap
        self.populate_tiles()

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

    def save_image(self, file_path: str):
        pygame.image.save(self.surface, file_path)
