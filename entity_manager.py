import numpy as np
import random
import pygame
from typing import Dict, Type
from terrain_manager import TerrainManager

class EntityManager:
    def __init__(self, terrain_manager: TerrainManager,
                 entity_map: Dict[str, int],
                 terrain_entity_map: Dict[int, int],
                 entity_classes: Dict[str, Type[pygame.sprite.Sprite]],
                 spawn_probs: np.ndarray,
                 tile_size: int = 50):
        
        self.terrain_manager = terrain_manager
        self.width = terrain_manager.width
        self.height = terrain_manager.height
        self.entity_locations = np.full((self.width, self.height), -1)
        self.entity_map = entity_map
        self.terrain_entity_map = terrain_entity_map
        self.entity_classes = entity_classes
        self.spawn_probs = spawn_probs
        self.entity_group = pygame.sprite.Group()
        self.tile_size = tile_size

        self.populate_tiles()
        self.add_player()

    def populate_tiles(self):
        self.entity_group.empty()
        for x in range(self.width):
            for y in range(self.height):
                terrain_code = self.terrain_manager.heightmap[x, y]  # Use heightmap from TerrainManager
                entity_type = self.terrain_entity_map.get(terrain_code)
                if entity_type is not None:
                    self.create_entity(entity_type, x, y)

    def create_entity(self, entity_type: int, x: int, y: int):
        r = self.spawn_probs[entity_type]
        # print(f'Creating entity {entity_type} at ({x}, {y}) with probability {r}')
        if random.random() > self.spawn_probs[entity_type]:
            return

        pixel_x, pixel_y = x * self.tile_size, y * self.tile_size
        entity_class = self.entity_classes[entity_type]
        entity = entity_class(pixel_x, pixel_y, self.tile_size)
        self.entity_group.add(entity)
        self.entity_locations[x, y] = entity_type

    def add_player(self):
        empty_tiles = np.argwhere(self.entity_locations == -1)
        if empty_tiles.size == 0:
            raise ValueError("No space to add the player.")
        player_location = random.choice(empty_tiles)
        self.create_entity(self.entity_map['player'], player_location[0], player_location[1])

