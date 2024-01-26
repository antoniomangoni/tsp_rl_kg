import numpy as np
import os
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
        'fish': 0,
        'tree': 1,
        'mossy rock': 2,
        'snowy rock': 3,
        'player': 4
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
    heightmap_generator = HeightmapGenerator(
        width=5, 
        height=5,
        scale=10,
        terrain_thresholds=terrain_thresholds,
        octaves=3, 
        persistence=0.2, 
        lacunarity=1.0
    )

    heightmap = heightmap_generator.generate()
    terrain_manager = TerrainManager(heightmap, terrain_colors)
    entity_manager = EntityManager(terrain_manager, entity_map, terrain_entity_map,
                                   entity_classes, entity_spawn_probabilities, tile_size=50)
    renderer = Renderer(terrain_manager, entity_manager, tile_size=50)

    PLAYER_MOVED = pygame.USEREVENT + 1
    ENVIRONMENT_CHANGED = pygame.USEREVENT + 2

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 800))

    running = True
    renderer.render()
    while running:
        should_redraw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            elif event.type == PLAYER_MOVED or event.type == ENVIRONMENT_CHANGED:
                should_redraw = True
        if should_redraw:
            renderer.render()
            should_redraw = False

    height_array = heightmap.transpose()
    entity_array = entity_manager.entity_locations.transpose() + 1
