import pygame
import numpy as np
from typing import Dict
from entities import Tree, Rock, Fish
import random

class TerrainRenderer:
    def __init__(self, heightmap: np.ndarray, colors: Dict[int, tuple], entity_map: Dict[str, int], entity_spawn_probabilities: Dict[str, int], tile_size=50, ):
        # Initialization
        pygame.init()
        self.heightmap = heightmap
        self.colors = colors
        self.entity_map = entity_map
        self.spawn_probs(entity_spawn_probabilities)
        self.width, self.height = self.heightmap.shape
        self.tile_size = int(tile_size)
        # self.game_world = np.zeros((2, self.width, self.height), dtype=int)
        self.entity_array = np.zeros((self.width, self.height), dtype=int)
        self.surface = pygame.display.set_mode((self.width * self.tile_size, self.height * self.tile_size))
        self.load_images()
        self.entity_instances = []  # Storing entity instances
        self.populate_tiles()

    def spawn_probs(self, entity_spawn_probabilities: Dict[str, int]):
        self.fish_spawn_prob = entity_spawn_probabilities['fish']
        self.tree_spawn_prob = entity_spawn_probabilities['tree']
        self.moss_rock_spawn_prob = entity_spawn_probabilities['mossy rock']
        self.snow_rock_spawn_prob = entity_spawn_probabilities['snowy rock']
        
    def load_images(self):
        self.images = {
            'tree': pygame.image.load('Pixel_Art/tree_1.png'),
            'rock_moss': pygame.image.load('Pixel_Art/rock_moss.png'),
            'rock_snow': pygame.image.load('Pixel_Art/rock_snow.png'),
            'fish': pygame.image.load('Pixel_Art/fish.png'),
            'player': pygame.image.load('Pixel_Art/player_image.png'),
        }

    def create_entity(self, entity_type, x, y):
        # Creating entities based on type and location with spawn probabilities
        pixel_x, pixel_y = x * self.tile_size, y * self.tile_size
        if entity_type == 'fish' and random.random() < self.fish_spawn_prob:
            self.entity_array[x, self.height - 1 - y] = self.entity_map['fish']
            self.entity_instances.append(
                Fish(pixel_x, pixel_y, self.images['fish'], pixel_size=self.tile_size))
        elif entity_type == 'tree' and random.random() < self.tree_spawn_prob:
            self.entity_array[x, self.height - 1 - y] = self.entity_map['tree']
            self.entity_instances.append(
                Tree(pixel_x, pixel_y, self.images['tree'], pixel_size=self.tile_size))
        elif entity_type == 'rock_moss' and random.random() < self.moss_rock_spawn_prob:
            self.entity_array[x, self.height - 1 - y] = self.entity_map['mossy rock']
            self.entity_instances.append(
                Rock(pixel_x, pixel_y, self.images['rock_moss'], pixel_size=self.tile_size))
        elif entity_type == 'rock_snow' and random.random() < self.snow_rock_spawn_prob:
            self.entity_array[x, self.height - 1 - y] = self.entity_map['snowy rock']
            self.entity_instances.append(
                Rock(pixel_x, pixel_y, self.images['rock_snow'], pixel_size=self.tile_size))
        else:
            self.entity_array[x, y] = -1

    def populate_tiles(self):
        # Populating the tiles based on the heightmap
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                if terrain == 0:
                    self.create_entity('fish', x, y)
                elif terrain == 1:
                    self.create_entity('fish', x, y)
                elif terrain == 2:
                    self.create_entity('rock_moss', x, y)
                elif terrain == 3:
                    self.create_entity('tree', x, y)
                elif terrain == 4:
                    self.create_entity('rock_snow', x, y)
                else:
                    # self.game_world[1, x, y] = -1
                    self.entity_array[x, y] = -1

    def render(self):
        # Clear the surface and draw the terrain
        self.surface.fill((0, 0, 0))
        for x in range(self.width):
            for y in range(self.height):
                terrain = self.heightmap[x, y]
                color = self.colors[terrain]
                # Flip y-coordinate for rendering
                flipped_y = self.height - 1 - y
                pygame.draw.rect(self.surface, color, (x * self.tile_size, flipped_y * self.tile_size, self.tile_size, self.tile_size))
        
        # Render entities
        for x in range(self.width):
            for y in range(self.height):
                entity_type = self.entity_array[x, y]
                if entity_type != -1:
                    entity = self.entity_instances[entity_type]
                    flipped_y = self.height - 1 - y
                    self.surface.blit(entity.image, (x * self.tile_size, flipped_y * self.tile_size))
        
        pygame.display.flip()

    def update_heightmap(self, new_heightmap: np.ndarray):
        self.heightmap = new_heightmap
        #self.game_world[0] = new_heightmap
        self.entity_array = new_heightmap
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

    def save_image(self, file_path: str):
        pygame.image.save(self.surface, file_path)