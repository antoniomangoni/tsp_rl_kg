import numpy as np
import pygame

from heightmap_generator import HeightmapGenerator
from entities import Tree, MossyRock, SnowyRock, Fish, Player
from terrain_manager import TerrainManager
from entity_manager import EntityManager
from renderer import Renderer

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
        'empty': 0,  # Empty tile
        'fish': 1,
        'tree': 2,
        'mossy rock': 3,
        'snowy rock': 4,
        'player': 5
    }

    entity_spawn_probabilities = np.array([0.15, 0.35, 0.4, 0.4, 1.0])
    entity_classes = np.array([Fish, Tree, MossyRock, SnowyRock, Player])

    terrain_entity_map = {
        terrain_map['strong water']: entity_map['fish'],
        terrain_map['water']: entity_map['fish'],
        terrain_map['plains']: entity_map['tree'],
        terrain_map['hills']: entity_map['tree'],
        terrain_map['mountains']: entity_map['mossy rock'],
        terrain_map['snow']: entity_map['snowy rock']
    }

    map_size = 30
    heightmap_generator = HeightmapGenerator(
        width=map_size, 
        height=map_size,
        scale=10,
        terrain_thresholds=terrain_thresholds,
        octaves=3, 
        persistence=0.2, 
        lacunarity=1.0
    )

    heightmap = heightmap_generator.generate()
    terrain_manager = TerrainManager(heightmap, terrain_colors)
    tile_size = 30
    entity_manager = EntityManager(terrain_manager, entity_map, terrain_entity_map,
                                   entity_classes, entity_spawn_probabilities, tile_size=tile_size)
    
    renderer = Renderer(terrain_manager, entity_manager, tile_size=tile_size)

    PLAYER_MOVED = pygame.USEREVENT + 1
    ENVIRONMENT_CHANGED = pygame.USEREVENT + 2

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((map_size * tile_size, map_size * tile_size))   
    pygame.display.set_caption("Game World")
    running = True

    renderer.render_terrain()
    renderer.render()

    while running:
        should_redraw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            elif event.type == PLAYER_MOVED or event.type == ENVIRONMENT_CHANGED:
                should_redraw = True
                renderer.terrain_needs_update = True  # If terrain needs to be updated
        if should_redraw:
            renderer.render()
            should_redraw = False
    
    height_array = heightmap.transpose()
    entity_array = entity_manager.entity_locations.transpose()

    print('Heightmap:')
    print(height_array)
    print('Entity locations:')
    print(entity_array)
    print('Game closed')