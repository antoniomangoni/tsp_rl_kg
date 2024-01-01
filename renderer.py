import pygame
import numpy as np
import random
from typing import Dict, Tuple, Type

class TerrainRenderer:
    def __init__(
            self, heightmap: np.ndarray,
            terrain_colors: np.ndarray,
            entity_map: Dict[str, int],
            entity_classes: np.ndarray,
            entity_spawn_probabilities: np.ndarray,
            terrain_entity_map: Dict[int, int],
            tile_size: int = 50):
        
        pygame.init()
        self.heightmap = heightmap
        self.entity_locations = np.full((heightmap.shape), -1)
        self.terrain_colors = terrain_colors
        self.entity_map = entity_map
        self.entity_classes = entity_classes
        self.spawn_probs = entity_spawn_probabilities
        self.terrain_entity_map = terrain_entity_map
        self.width, self.height = self.heightmap.shape
        self.tile_size = tile_size
        self.surface = pygame.display.set_mode((self.width * self.tile_size, self.height * self.tile_size))
        self.entity_group = pygame.sprite.Group()
        self.populate_tiles()

    def create_entity(self, entity_type: int, x: int, y: int):
        if random.random() > self.spawn_probs[entity_type]:
            return

        pixel_x, pixel_y = x * self.tile_size, y * self.tile_size
        entity_class = self.entity_classes[entity_type]
        entity = entity_class(pixel_x, pixel_y, pixel_size=self.tile_size)
        self.entity_group.add(entity)
        self.entity_locations[x, y] = entity_type

    def populate_tiles(self):
        self.entity_group.empty()
        for x in range(self.width):
            for y in range(self.height):
                terrain_code = self.heightmap[x, y]
                entity_type = self.terrain_entity_map.get(terrain_code)
                if entity_type is not None:
                    self.create_entity(entity_type, x, y)
     
    def render(self):
        self.surface.fill((0, 0, 0))                                                
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                color = self.terrain_colors[terrain]
                pygame.draw.rect(self.surface, color, (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))
        
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

    def print(self):
        print('Terrain:')
        print(np.transpose(self.heightmap))
        print('Entities:')
        print(np.transpose(self.entity_locations))
