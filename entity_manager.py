import numpy as np
import random
import pygame
from typing import Dict, Type
from terrain_manager import TerrainManager
from helper_functions import time_function, debug_function
from entities import Entity, Player

class EntityManager:
    @time_function
    def __init__(self, terrain_manager: TerrainManager,
                 entity_map: Dict[str, int],
                 terrain_entity_map: Dict[int, int],
                 entity_classes: np.ndarray[Type[Entity]],
                 spawn_probs: np.ndarray[float],
                 tile_size: int = 50):
        
        self.terrain_manager = terrain_manager
        self.width = terrain_manager.width
        self.height = terrain_manager.height
        self.entity_locations = np.full((self.width, self.height), 0, dtype=np.uint8)
        self.entity_map = entity_map
        self.terrain_entity_map = terrain_entity_map
        self.entity_classes = entity_classes
        self.spawn_probs = spawn_probs
        self.entity_group = pygame.sprite.Group()
        self.tile_size = tile_size

        self.populate_tiles()
        self.add_player()

    @time_function
    def populate_tiles(self):
        self.entity_group.empty()
        for x in range(self.width):
            for y in range(self.height):
                # Use heightmap from TerrainManager to get terrain code
                terrain_code = self.terrain_manager.heightmap[x, y]
                # Get the entity type which spawns on this terrain
                entity_type = self.terrain_entity_map.get(terrain_code) -1 
                if entity_type is not None and random.random() < self.spawn_probs[entity_type]: 
                    # Add entity type to the entity locations array for processing
                    self.entity_locations[x, y] = entity_type
                    # Get entity type from its one hot encoding and spawn it
                    self.create_entity(self.entity_classes[entity_type], x, y)

    @time_function
    def create_entity(self, entity_class: Entity, x: int, y: int):
        # print(f'Creating entity {entity_type} at ({x}, {y}))
        pixel_x, pixel_y = x * self.tile_size, y * self.tile_size
        entity = entity_class(pixel_x, pixel_y, self.tile_size)
        self.entity_group.add(entity) # Add entity to the sprite group for rendering

    @time_function
    def add_player(self):
        empty_tiles = np.argwhere(self.entity_locations == 0)
        if empty_tiles.size == 0:
            raise ValueError("No space to add the player.")
        # print(f'Entity locations: {self.entity_locations}')
        player_location = random.choice(empty_tiles)
        self.entity_locations[player_location[0], player_location[1]] = self.entity_map['player']
        self.create_entity(Player, player_location[0], player_location[1])
