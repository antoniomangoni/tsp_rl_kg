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
        self.passable = False

    def add_path(self, path):
        self.entity_on_tile = path
        self.entity_index = path.id
        self.passable = True
        self.energy_requirement = max(0, self.energy_requirement - 2)

    def adjust_for_entity(self, entity):
        if isinstance(entity, WoodPath):
            self.passable = True
            self.energy_requirement = max(0, self.energy_requirement - 2)
        else:
            # This means that when a player moves onto a tile with an entity it is not passable,
            # when the player leaves it becomes passable again.
            self.passable = False

    def remove_entity(self):
        if isinstance(self.entity_on_tile, WoodPath):
            self.energy_requirement += 2
        self.entity_on_tile = None
        self.passable = True

    def set_colour(self, id):
        if id == 0:
            return (0, 0, 128)
        elif id == 1:
            return (0, 0, 255)
        elif id == 2:
            return (0, 255, 0)
        elif id == 3:
            return (0, 128, 0)
        elif id == 4:
            return (128, 128, 128)
        elif id == 5:
            return (255, 255, 255)
        else:
            return (0, 0, 0)

class DeepWater(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 0
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 10
        self.entity_type = Fish
        self.entity_prob = entity_prob

    def shallow(self):
        self.__class__ = Water
        self.colour = (0, 0, 255)
        self.image = self.create_image()
        self.elevation = 1
        self.energy_requirement = 6
        self.entity_type = Fish

class Water(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 1
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 6
        self.entity_type = Fish
        self.entity_prob = entity_prob

    def land_fill(self):
        self.__class__ = Plains
        self.colour = (0, 255, 0)
        self.image = self.create_image()
        self.elevation = 2
        self.energy_requirement = 2
        self.entity_type = Tree

class Plains(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 2
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 2
        self.entity_type = Tree
        self.entity_prob = entity_prob

class Hills(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 3
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 3
        self.entity_type = Tree
        self.entity_prob = entity_prob

class Mountains(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 4
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 5
        self.entity_type = MossyRock
        self.entity_prob = entity_prob

class Snow(Terrain):
    def __init__(self, x, y, tile_size, entity_prob):
        super().__init__(x, y, tile_size, entity_prob)
        
        self.elevation = 5
        self.colour = self.set_colour(self.elevation)
        self.image = self.create_image()
        self.energy_requirement = 4
        self.entity_type = SnowyRock
        self.entity_prob = entity_prob
