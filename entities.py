from typing import Any
import pygame

class Entity(pygame.sprite.Sprite):
    def __init__(self, x, y, art='Pixel_Art/red.png', tile_size=100):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(art), (tile_size, tile_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Player(Entity):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/player.png', tile_size=tile_size)
        self.tile_size = tile_size
        self.x = x // tile_size
        self.y = y // tile_size

    def move(self, dx, dy):
        self.rect.x += dx * self.tile_size 
        self.rect.y += dy * self.tile_size
        self.x += dx
        self.y += dy

class Outpost(Entity):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/outpost_2.png', tile_size=tile_size)

class Tree(Entity):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/tree_1.png', tile_size=tile_size)

class Rock(Entity):
    def __init__(self, x, y, art, tile_size=100):
        super().__init__(x, y, art, tile_size)

class MossyRock(Rock):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/rock_moss.png', tile_size=tile_size)

class SnowyRock(Rock):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/rock_snow.png', tile_size=tile_size)

class Fish(Entity):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, art='Pixel_Art/fish.png', tile_size=tile_size)
