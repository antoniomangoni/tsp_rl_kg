# main.py
from renderer import TerrainRenderer
from heightmap_generator import HeightmapGenerator
import numpy as np

if __name__ == '__main__':
    thresholds = {'strong_water': 0.2, 'water': 0.38, 'plains': 0.5, 'hills': 0.7, 'mountains': 0.85, 'snow': 1.0}
    colors = {0: (0, 0, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (0, 128, 0), 4: (128, 128, 128), 5: (255, 255, 255), 6: (80, 127, 255)}

    heightmap_generator = HeightmapGenerator(width=1000, height=1000, scale=30, thresholds=thresholds, octaves=6, persistence=0.4, lacunarity=3.0)
    heightmap = heightmap_generator.generate()
    print(np.unique(heightmap))

    renderer = TerrainRenderer(heightmap, colors, tile_size=6)
    renderer.real_time_update()
