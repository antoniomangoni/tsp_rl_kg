import random
import pygame

from entities import Tree, MossyRock, SnowyRock, Fish, WoodPath

class Terrain():
    def __init__(self, x, y, entity_init_prob, colour=None, tile_size=100):
        self.colour = colour
        self.image = pygame.Surface((tile_size, tile_size))
        self.image.fill(self.colour)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # Terrain cell properties
        self.elevation = None
        self.energy_requirement = None
        self.passable = True
        self.entity_type = None
        self.entity_prob = entity_init_prob
        self.entity_on_tile = None  # To keep track of the entity on this tile

    def spawn_entity(self):
        if random.random() < self.entity_prob:
            entity = self.entity_type(self.rect.x, self.rect.y, self.rect.width)
            self.entity_on_tile = entity  # Record the entity on this tile
            if isinstance(entity, WoodPath):
                self.passable = True
                self.energy_requirement -= 2
            else:
                self.passable = False
            return entity
        
    def remove_entity(self):
        self.entity_on_tile = None
        self.passable = True
        if isinstance(self.entity_type, WoodPath):
            self.energy_requirement += 2

class DeepWater(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(0, 0, 128), tile_size=tile_size)
        self.elevation = 0
        self.energy_requirement = 10
        self.entity_type = Fish


class Water(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(0, 0, 255), tile_size=tile_size)
        self.elevation = 1
        self.energy_requirement = 6
        self.entity_type = Fish

class Plains(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(0, 255, 0), tile_size=tile_size)
        self.elevation = 2
        self.energy_requirement = 2
        self.entity_type = Tree

class Hills(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(0, 128, 0), tile_size=tile_size)
        self.elevation = 3
        self.energy_requirement = 3
        self.entity_type = Tree

class Mountains(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(128, 128, 128), tile_size=tile_size)
        self.elevation = 4
        self.energy_requirement = 5
        self.entity_type = MossyRock

class Snow(Terrain):
    def __init__(self, x, y, entity_init_prob, tile_size=100):
        super().__init__(x, y, entity_init_prob, colour=(255, 255, 255), tile_size=tile_size)
        self.elevation = 5
        self.energy_requirement = 7
        self.entity_type = SnowyRock