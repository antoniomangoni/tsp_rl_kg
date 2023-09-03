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
                                        base=42)
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
        int_heightmap[heightmap >= last_threshold] = len(self.thresholds)  # Snow by default
        return int_heightmap

    def find_river_path(self, heightmap: np.ndarray):
        def get_neighbors(point):
            x, y = point
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and (dx, dy) != (0, 0):
                        neighbors.append((nx, ny))
            return neighbors

        river_path = []
        snow_points = np.argwhere(heightmap == 5)  # Assuming 5 is the code for 'snow'
        
        if snow_points.size == 0:
            print("No snow point to initiate river")
            return

        # Randomly choose one snow point to start
        current_point = tuple(snow_points[random.randint(0, len(snow_points) - 1)])

        while True:
            river_path.append(current_point)
            neighbors = get_neighbors(current_point)
            next_point = min(neighbors, key=lambda p: heightmap[p] if heightmap[p] < heightmap[current_point] else float('inf'))
            
            if heightmap[next_point] >= heightmap[current_point]:
                break
                        
            current_point = next_point
                
        for point in river_path:
            heightmap[point] = 6  # Assuming 6 is the code for 'river'

    def decode_terrain(self, code: int) -> str:
        terrain_mapping = {
            0: 'deep_water',
            1: 'water',
            2: 'plains',
            3: 'hills',
            4: 'mountains',
            5: 'snow',
            6: 'river'
        }
        return terrain_mapping.get(code, "unknown")
