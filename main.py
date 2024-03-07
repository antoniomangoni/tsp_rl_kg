import numpy as np
import pygame
from heightmap_generator import HeightmapGenerator
from entities import Tree, MossyRock, SnowyRock, Fish, Player, Outpost, WoodPath
from terrain_manager import TerrainManager
from entity_manager import EntityManager
from renderer import Renderer

if __name__ == '__main__':
    pygame.init()

    # Configuration Parameters
    map_size = 10  # Adjust map size as needed
    tile_size = 100

    # Generate heightmap
    heightmap_generator = HeightmapGenerator(width=map_size, 
                                             height=map_size, 
                                             scale=10, 
                                             terrain_thresholds=np.array([0.2, 0.38, 0.5, 0.7, 0.9, 1.0]), 
                                             octaves=3, persistence=0.2, lacunarity=2.0)
    heightmap = heightmap_generator.generate()

    # Initialize TerrainManager with generated heightmap
    terrain_manager = TerrainManager(heightmap, tile_size)

    # Initialize EntityManager
    entity_manager = EntityManager(terrain_manager, 
                                   number_of_outposts=4)

    # Initialize Renderer 
    renderer = Renderer(terrain_manager, entity_manager)

    screen = pygame.display.set_mode((map_size * tile_size, map_size * tile_size))
    pygame.display.set_caption("Game World")

    renderer.render_terrain()  # Consider optimizing to avoid re-rendering static terrain
    renderer.render()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                renderer.terrain_needs_update = True
                if event.key == pygame.K_LEFT:
                    entity_manager.move_player("LEFT")
                elif event.key == pygame.K_RIGHT:
                    entity_manager.move_player("RIGHT")
                elif event.key == pygame.K_UP:
                    entity_manager.move_player("UP")
                elif event.key == pygame.K_DOWN:
                    entity_manager.move_player("DOWN")

        # Render updates

        renderer.terrain_needs_update = False
        pygame.display.flip()

    pygame.quit()
    print('Game closed')
