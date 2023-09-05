# renderer.py
import pygame
import numpy as np
from typing import Dict
from heightmap_generator import HeightmapGenerator
from pixel_art import Tree
import random

class TerrainRenderer:
    def __init__(self, heightmap: np.ndarray, colors: Dict[int, tuple]):
        pygame.init()
        self.heightmap = heightmap
        self.colors = colors
        self.width, self.height = self.heightmap.shape
        self.surface = pygame.display.set_mode((self.width, self.height))

        self.tree_group = pygame.sprite.Group()
        self.populate_tiles()

    def populate_tiles(self):
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                if terrain == 3 and random.random() < 0.25:
                    tree = Tree(x, y, self.tree_image, pixel_size=100)
                    self.tree_group.add(tree)

    def render(self):
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                color = self.colors[terrain]
                self.surface.set_at((x, y), color)

                if terrain == 3:
                # Use a condition to control the density of trees
                    if random.random() < 0.25:  
                        self.surface.blit(self.tree_image, (x, y))

        pygame.display.flip()

    def update_heightmap(self, new_heightmap: np.ndarray):
        self.heightmap = new_heightmap
        self.render()

    def real_time_update(self, update_interval=50):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_heightmap(self.heightmap)  # Placeholder; replace with real-time updating logic
            clock.tick(update_interval)
            
        pygame.quit()

if __name__ == '__main__':
    thresholds = {'deep_water': 0.15, 'water': 0.30, 'plains': 0.5, 'hills': 0.7, 'mountains': 0.85, 'snow': 1.0}
    colors = {0: (0, 0, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (0, 128, 0), 4: (128, 128, 128), 5: (255, 255, 255), 6: (80, 127, 255)}

    generator = HeightmapGenerator(800, 800, 350.0, thresholds, 6, 0.6, 1.8)
    heightmap = generator.generate()
    # generator.find_river_path(heightmap)  # This will be part of the `river.py`
    renderer = TerrainRenderer(heightmap, colors)
    renderer.real_time_update()
