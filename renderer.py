# renderer.py
import pygame
import numpy as np
from typing import Dict
from heightmap_generator import HeightmapGenerator
from entities import Tree
import random

class TerrainRenderer:
    def __init__(self, heightmap: np.ndarray, colors: Dict[int, tuple], tile_size=10):
        pygame.init()
        self.heightmap = heightmap
        self.colors = colors
        self.width, self.height = self.heightmap.shape
        self.tile_size = tile_size
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.tree_image = pygame.image.load('Pixel_Art/tree_1.png')  # Load the tree image
        self.tree_instances = []  # Create an empty list to store tree instances
        self.populate_tiles()

    def populate_tiles(self):
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                if terrain == 3 and random.random() < 0.25:
                    tree = Tree(x * self.tile_size, y * self.tile_size, self.tree_image, pixel_size=self.tile_size)
                    self.tree_instances.append(tree)


    def render(self):
        self.surface.fill((0, 0, 0))  # Clear the surface before rendering
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                color = self.colors[terrain]
                pygame.draw.rect(self.surface, color, (x * self.tile_size , y * self.tile_size , self.tile_size , self.tile_size ))  # Draw a rectangle to represent each tile; adjust the size and coordinates multiplication factor as needed
                
        for tree in self.tree_instances:  # Render tree instances
            self.surface.blit(tree.image, tree.rect.topleft)
        
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

