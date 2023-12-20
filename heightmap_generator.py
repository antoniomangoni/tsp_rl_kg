# heightmap_generator.py
import noise
import numpy as np
from typing import Dict
import random

class HeightmapGenerator:
    def __init__(self, width: int, height: int, scale: float, thresholds: Dict[str, float], octaves: int, persistence: float, lacunarity: float):
        self.width = width
        self.height = height
        self.scale = scale
        self.thresholds = thresholds
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = random.randint(0, 100)

    def save_heightmap(self, file_path: str):
        np.save(file_path, self.generate())

    def generate(self) -> np.ndarray:
        heightmap, min_val, max_val = self.generate_raw_heightmap()
        self.normalize_heightmap(heightmap, min_val, max_val)
        return self.classify_heightmap(heightmap)

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

    def classify_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        int_heightmap = np.zeros(heightmap.shape, dtype=int)
        last_threshold = 0
        for index, (terrain, threshold) in enumerate(sorted(self.thresholds.items(), key=lambda x: x[1])):
            mask = (heightmap >= last_threshold) & (heightmap < threshold)
            int_heightmap[mask] = index
            last_threshold = threshold
        int_heightmap[heightmap >= last_threshold] = len(self.thresholds) - 1 # Snow by default
        return int_heightmap

    def decode_terrain(self, code: int) -> str:
        terrain_mapping = {
            0: 'strong_water',
            1: 'water',
            2: 'plains',
            3: 'hills',
            4: 'mountains',
            5: 'snow',
            6: 'river'
        }
        return terrain_mapping.get(code, "unknown")

    def printHeightmap(self):
        print(self.heightmap)