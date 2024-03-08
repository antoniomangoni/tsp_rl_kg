import pygame
import numpy as np

from terrains import DeepWater, Water, Plains, Hills, Mountains, Snow  # Assuming terrains.py contains the Terrain classes

class TerrainManager:
    def __init__(self, heightmap: np.ndarray, tile_size: int = 50):
        self.heightmap = heightmap
        self.tile_size = tile_size
        self.width, self.height = heightmap.shape
        self.terrain_types = [DeepWater, Water, Plains, Hills, Mountains, Snow]
        self.entity_init_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.terrain_object_grid, self.terrain_index_grid = self.initialize_terrain()

    def initialize_terrain(self):
        grid = np.empty((self.width, self.height), dtype=object)
        index_grid = np.zeros((self.width, self.height), dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                terrain_code = int(self.heightmap[x, y] * (len(self.terrain_types) - 1))
                index_grid[x, y] = terrain_code
                entity_prob = self.entity_init_prob[terrain_code]
                grid[x, y] = self.place_terrain(terrain_code, x * self.tile_size, y * self.tile_size, entity_prob)
        return grid, index_grid

    def place_terrain(self, code, x, y, entity_prob):
        # Instantiate the terrain with its probability for entity spawning
        terrain = self.terrain_types[code](x, y, self.tile_size, entity_prob)
        return terrain

    def render(self, surface):
        for x in range(self.width):
            for y in range(self.height):
                terrain_tile = self.terrain_object_grid[x, y]
                surface.blit(terrain_tile.image, (terrain_tile.x, terrain_tile.y))

    def is_passable(self, x, y):
        return self.terrain_object_grid[x, y].passable

