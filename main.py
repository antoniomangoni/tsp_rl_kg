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
        'player': 5,
        'outpost': 6,
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

    map_size = 5
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
    tile_size = 1000 // map_size
    entity_manager = EntityManager(terrain_manager, entity_map, terrain_entity_map,
                                   entity_classes, entity_spawn_probabilities, tile_size=tile_size,
                                   number_of_outposts=4, outpost_terrain=[2, 3])
    
    renderer = Renderer(terrain_manager, entity_manager, tile_size=tile_size)

    def handle_input():
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pygame.event.post(pygame.event.Event(PLAYER_MOVED, {"direction": "LEFT"}))
        elif keys[pygame.K_RIGHT]:
            pygame.event.post(pygame.event.Event(PLAYER_MOVED, {"direction": "RIGHT"}))
        elif keys[pygame.K_UP]:
            pygame.event.post(pygame.event.Event(PLAYER_MOVED, {"direction": "UP"}))
        elif keys[pygame.K_DOWN]:
            pygame.event.post(pygame.event.Event(PLAYER_MOVED, {"direction": "DOWN"}))

    PLAYER_MOVED = pygame.USEREVENT + 1
    ENVIRONMENT_CHANGED = pygame.USEREVENT + 2

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((map_size * tile_size, map_size * tile_size))   
    pygame.display.set_caption("Game World")
    running = True

    renderer.render_terrain()
    renderer.render()

    render_flag = False
    # In the main game loop
    while running:
        handle_input()
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == PLAYER_MOVED:
                direction = event.dict['direction']
                render_flag = entity_manager.move_player(direction)

        if render_flag:
            renderer.render()
            render_flag = False

    
    height_array = heightmap.transpose()
    entity_array = entity_manager.entity_locations.transpose()

    print('Heightmap:')
    print(height_array)
    print('Entity locations:')
    print(entity_array)

    print('Game closed')