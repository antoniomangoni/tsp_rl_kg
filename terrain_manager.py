import numpy as np

class TerrainManager:
    def __init__(self, heightmap: np.ndarray, terrain_colors: np.ndarray):
        self.heightmap = heightmap
        self.terrain_colors = terrain_colors
        self.width, self.height = heightmap.shape

    def get_terrain_color(self, terrain_code: int):
        return self.terrain_colors[terrain_code]
