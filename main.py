from renderer import TerrainRenderer
from heightmap_generator import HeightmapGenerator
import numpy as np
from entities import Tree, MossyRock, SnowyRock, Fish

if __name__ == '__main__':
    terrain_map = {
        'strong water': 0,
        'water': 1,
        'plains': 2,
        'hills': 3,
        'mountains': 4,
        'snow': 5
    }
    terrain_thresholds = np.array([0.2, 0.38, 0.5, 0.7, 0.9, 1.0])
    terrain_colors = np.array([(0, 0, 128), (0, 0, 255), (0, 255, 0), (0, 128, 0), (128, 128, 128), (255, 255, 255)])

    entity_map = {
        'fish': 0,
        'tree': 1,
        'mossy rock': 2,
        'snowy rock': 3
    }

    entity_spawn_probabilities = np.array([0.2, 0.4, 0.4, 0.5])
    entity_classes = np.array([Fish, Tree, MossyRock, SnowyRock])

    terrain_entity_map = {
        terrain_map['strong water']: entity_map['fish'],
        terrain_map['water']: entity_map['fish'],
        terrain_map['plains']: entity_map['tree'],
        terrain_map['hills']: entity_map['tree'],
        terrain_map['mountains']: entity_map['mossy rock'],
        terrain_map['snow']: entity_map['snowy rock']
    }

    heightmap_generator = HeightmapGenerator(
        width=4, 
        height=4,
        scale=10,
        terrain_thresholds=terrain_thresholds,
        octaves=3, 
        persistence=0.2, 
        lacunarity=1.0
    )
    
    heightmap = heightmap_generator.generate()
    
    renderer = TerrainRenderer(heightmap, terrain_colors,
                               entity_map, entity_classes,
                               entity_spawn_probabilities,
                               terrain_entity_map,
                               tile_size=50)
    
    renderer.print()

    renderer.real_time_update()
