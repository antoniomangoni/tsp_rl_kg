import pygame
import numpy as np

from terrains import DeepWater, Water, Plains, Hills, Mountains, Snow  # Assuming terrains.py contains the Terrain classes

class TerrainManager:
    def __init__(self, heightmap: np.ndarray, tile_size: int = 50):
        self.heightmap = heightmap
        self.tile_size = tile_size
        self.width, self.height = heightmap.shape
        self.terrain_grid = np.empty((self.width, self.height), dtype=object)
        self.initialize_terrain()

    def initialize_terrain(self):
        for x in range(self.width):
            for y in range(self.height):
                terrain_code = self.heightmap[x, y]
                self.terrain_grid[x, y] = self.create_terrain(terrain_code, x, y)

    def create_terrain(self, terrain_code, x, y):
        if terrain_code == 0:
            return DeepWater(x * self.tile_size, y * self.tile_size, self.tile_size)
        elif terrain_code == 1:
            return Water(x * self.tile_size, y * self.tile_size, self.tile_size)
        elif terrain_code == 2:
            return Plains(x * self.tile_size, y * self.tile_size, self.tile_size)
        elif terrain_code == 3:
            return Hills(x * self.tile_size, y * self.tile_size, self.tile_size)
        elif terrain_code == 4:
            return Mountains(x * self.tile_size, y * self.tile_size, self.tile_size)
        elif terrain_code == 5:
            return Snow(x * self.tile_size, y * self.tile_size, self.tile_size)
        else:
            raise ValueError("Invalid terrain code")

    def render(self, surface):
        for x in range(self.width):
            for y in range(self.height):
                terrain_tile = self.terrain_grid[x, y]
                surface.blit(terrain_tile.image, (terrain_tile.rect.x, terrain_tile.rect.y))

    def is_passable(self, x, y):
        return self.terrain_grid[x, y].passable