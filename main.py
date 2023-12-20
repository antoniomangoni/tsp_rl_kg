from renderer import TerrainRenderer
from heightmap_generator import HeightmapGenerator
import numpy as np

if __name__ == '__main__':
    thresholds = {'strong_water': 0.2, 'water': 0.38, 'plains': 0.5, 'hills': 0.7, 'mountains': 0.9, 'snow': 1.0}
    colors = {0: (0, 0, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (0, 128, 0), 4: (128, 128, 128), 5: (255, 255, 255)}
    entity_map = {'fish': 0, 'tree': 1, 'mossy rock': 2, 'snowy rock': 3}
    entity_spawn_probabilities = {'fish': 0.2, 'tree': 0.5, 'mossy rock': 0.2, 'snowy rock': 0.5}

    heightmap_generator = HeightmapGenerator(width=3, height=3, scale=10, thresholds=thresholds, octaves=3, persistence=0.2, lacunarity=1.0)
    heightmap = heightmap_generator.generate()

    adjusted_heightmap = np.rot90(heightmap, k=1)
    print('Heightmap is: ')
    print(adjusted_heightmap)

    renderer = TerrainRenderer(heightmap, colors, entity_map, entity_spawn_probabilities, tile_size=50)
    print('game world is: ')
    print(renderer.entity_array)
    renderer.real_time_update()
    
    # for i in range(10):
    #     # Create a new heightmap and save it to a file in the folder '7x7'. The images in a subfolder named '7x7/img' and the heightmaps in a subfolder named '7x7/maps'
    #     heightmap_generator = HeightmapGenerator(width=350, height=350, scale=5, thresholds=thresholds, octaves=3, persistence=0.2, lacunarity=1.0)
    #     heightmap = heightmap_generator.generate()
    #     heightmap_generator.save_heightmap(f"7x7/maps/{i}.npy")
    #     renderer = TerrainRenderer(heightmap, colors, tile_size=50)
    #     # renderer.real_time_update()
    #     renderer.save_image(f"7x7/img/{i}.png")