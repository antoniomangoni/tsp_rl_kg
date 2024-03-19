import pygame
import os
from random import choice

class Entity(pygame.sprite.Sprite):
    _images = {}

    def __init__(self, x, y, art, tile_size):
        super().__init__()
        # Ensure the image is loaded
        if art not in self._images:
            full_path = f'Pixel_Art/{art}'
            if os.path.exists(full_path):  
                self._images[art] = pygame.transform.scale(pygame.image.load(full_path), (tile_size, tile_size))
            else:
                raise FileNotFoundError(f"Image file {art} not found in Pixel_Art directory.")
        
        self.image = self._images[art]
        self.grid_x = x
        self.grid_y = y
        self.tile_size = tile_size
        self.screen_x = x * tile_size
        self.screen_y = y * tile_size
        self.rect = self.image.get_rect()
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y
        self.id = None

    def move(self, dx, dy):
        # Update the logical grid position
        self.grid_x += dx
        self.grid_y += dy
        # Update the screen (pixel) position based on the new grid position
        self.screen_x = self.grid_x * self.tile_size
        self.screen_y = self.grid_y * self.tile_size
        # Update the rect position for drawing and collision detection
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y


class Player(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='player.png', tile_size=tile_size)
        self.id = 7

    def random_move(self):
        self.move(*choice([(1, 0), (-1, 0), (0, 1), (0, -1)]))

class Outpost(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='outpost_2.png', tile_size=tile_size)
        self.id = 5

class WoodPath(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='wood_path.png', tile_size=tile_size)
        self.id = 6

class Fish(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='fish.png', tile_size=tile_size)
        self.id = 1

class Tree(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='tree_1.png', tile_size=tile_size)
        self.id = 2

class Rock(Entity):
    def __init__(self, x, y, art, tile_size):
        super().__init__(x, y, art, tile_size)

class MossyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='rock_moss.png', tile_size=tile_size)
        self.id = 3

class SnowyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='rock_snow.png', tile_size=tile_size)
        self.id = 4


