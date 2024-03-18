import random
import pygame

from entities import Tree, MossyRock, SnowyRock, Fish, WoodPath
class Terrain:
    def __init__(self, x, y, tile_size, entity_prob):
        self.grid_x = x
        self.grid_y = y
        self.tile_size = tile_size
        self.screen_x = x * tile_size
        self.screen_y = y * tile_size
        self.colour = None
        self.image = None
        self.elevation = None
        self.energy_requirement = None
        self.passable = True
        self.entity_type = None
        self.entity_index = None
        self.entity_prob = entity_prob
        self.entity_on_tile = None

    def create_image(self):
        image = pygame.Surface((self.tile_size, self.tile_size))
        image.fill(self.colour)
        return image
    
    def add_entity(self, entity):
        self.entity_on_tile = entity
        self.entity_index = entity.id
        self.adjust_for_entity(entity)

    def adjust_for_entity(self, entity):
        if isinstance(entity, WoodPath):
            self.passable = True
            self.energy_requirement = max(0, self.energy_requirement - 2)
        else:
            self.passable = False

    def remove_entity(self):
        if isinstance(self.entity_on_tile, WoodPath):
            self.energy_requirement += 2
        self.entity_on_tile = None
        self.passable = True


class DeepWater(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (0, 0, 128)
        self.image = self.create_image()
        self.elevation = 0
        self.energy_requirement = 10
        self.entity_type = Fish
        self.entity_prob = entity_prob

class Water(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (0, 0, 255)
        self.image = self.create_image()
        self.elevation = 1
        self.energy_requirement = 6
        self.entity_type = Fish
        self.entity_prob = entity_prob

class Plains(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (0, 255, 0)
        self.image = self.create_image()
        self.elevation = 2
        self.energy_requirement = 2
        self.entity_type = Tree
        self.entity_prob = entity_prob

class Hills(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (0, 128, 0)
        self.image = self.create_image()
        self.elevation = 3
        self.energy_requirement = 3
        self.entity_type = Tree
        self.entity_prob = entity_prob

class Mountains(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (128, 128, 128)
        self.image = self.create_image()
        self.elevation = 4
        self.energy_requirement = 5
        self.entity_type = MossyRock
        self.entity_prob = entity_prob

class Snow(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        self.colour = (255, 255, 255)
        self.image = self.create_image()
        self.elevation = 5
        self.energy_requirement = 4
        self.entity_type = SnowyRock
        self.entity_prob = entity_prob
