# heightmap_generator.py
import noise
import numpy as np
from typing import Dict
import random

from helper_functions import time_function

class HeightmapGenerator:
    @time_function
    def __init__(self, width: int, height: int, scale: float, terrain_thresholds: np.ndarray, octaves: int, persistence: float, lacunarity: float):
        self.width = width
        self.height = height
        self.scale = scale
        self.terrain_thresholds = terrain_thresholds
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = random.randint(0, 100)

    def save_heightmap(self, file_path: str):
        np.save(file_path, self.generate())

    @time_function
    def generate(self) -> np.ndarray:
        heightmap, min_val, max_val = self.generate_raw_heightmap()
        self.normalize_heightmap(heightmap, min_val, max_val)
        return self.classify_heightmap(heightmap)

    @time_function
    def generate_raw_heightmap(self) -> np.ndarray:
        heightmap = np.zeros((self.width, self.height), dtype=float)
        min_val = 1.0
        max_val = 0.0
        for x in range(self.width):
            for y in range(self.height):
                height = noise.pnoise2(x / self.scale, 
                                        y / self.scale, 
                                        octaves=self.octaves, 
                                        persistence=self.persistence, 
                                        lacunarity=self.lacunarity, 
                                        repeatx=self.width, 
                                        repeaty=self.height, 
                                        base=self.seed)
                height = (height + 1) / 2  # Normalize to [0, 1]
                heightmap[x, y] = height
                min_val = min(min_val, height)
                max_val = max(max_val, height)
        return heightmap, min_val, max_val

    def normalize_heightmap(self, heightmap: np.ndarray, min_val: float, max_val: float):
        heightmap -= min_val
        heightmap /= (max_val - min_val)

    @time_function
    def classify_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        int_heightmap = np.zeros(heightmap.shape, dtype=int)
        num_thresholds = len(self.terrain_thresholds)

        for index, threshold in enumerate(self.terrain_thresholds):
            if index == 0:
                mask = heightmap < threshold
            elif index == num_thresholds - 1:
                mask = heightmap >= self.terrain_thresholds[index - 1]
            else:
                mask = (heightmap >= self.terrain_thresholds[index - 1]) & (heightmap < threshold)
            
            int_heightmap[mask] = index

        return int_heightmap


