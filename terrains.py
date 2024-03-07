from typing import Any
import pygame

class Terrain():
    def __init__(self, x, y, colour = None, tile_size=100):
        super().__init__()
        self.colour = colour
        self.image = pygame.Surface((tile_size, tile_size))
        self.image.fill(self.colour)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # Terrain cell properties
        self.elevation = None
        self.energy_requirement = None
        self.path = False
        self.entity = None


class DeepWater(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(0, 0, 128), tile_size=tile_size)
        self.elevation = 0
        self.energy_requirement = 10

class Water(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(0, 0, 255), tile_size=tile_size)
        self.elevation = 1
        self.energy_requirement = 6

class Plains(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(0, 255, 0), tile_size=tile_size)
        self.elevation = 2
        self.energy_requirement = 2

class Hills(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(0, 128, 0), tile_size=tile_size)
        self.elevation = 3
        self.energy_requirement = 3

class Mountains(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(128, 128, 128), tile_size=tile_size)
        self.elevation = 4
        self.energy_requirement = 5

class Snow(Terrain):
    def __init__(self, x, y, tile_size=100):
        super().__init__(x, y, colour=(255, 255, 255), tile_size=tile_size)
        self.elevation = 5
        self.energy_requirement = 7