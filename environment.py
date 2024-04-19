import random
import pygame
import numpy as np

from entities import Player, Outpost, WoodPath, Fish, Tree, MossyRock, SnowyRock
from terrains import DeepWater, Water, Plains, Hills, Mountains, Snow

class Environment:
    def __init__(self, heightmap: np.ndarray, tile_size: int = 50, number_of_outposts: int = 3):
        self.heightmap = heightmap
        self.terrain_index_grid = np.zeros_like(self.heightmap)
        self.entity_index_grid = np.zeros_like(self.heightmap)
        self.terrain_object_grid = np.zeros_like(self.heightmap, dtype=object)
        self.tile_size = tile_size
        self.width, self.height = heightmap.shape
        self.number_of_outposts = number_of_outposts
        self.entity_group = pygame.sprite.LayeredUpdates() # pygame.sprite.Group()
        self.terrain_definitions = {
            0: {'class': DeepWater, 'entity_prob': 0.4},
            1: {'class': Water, 'entity_prob': 0.3},
            2: {'class': Plains, 'entity_prob': 0.3},
            3: {'class': Hills, 'entity_prob': 0.4},
            4: {'class': Mountains, 'entity_prob': 0.4},
            5: {'class': Snow, 'entity_prob': 0.4},
        }

        self.suitable_terrain_locations = {'Plains': [], 'Hills': [], 'Mountains': [], 'Snow': []}
        self.outpost_locations = [] # List of (x, y) coordinates for each outpost, no need to use an array here

        self.initialize_environment()
        self.add_outposts()
        self.player = self.init_player()

        self.environment_changed_flag = False
        self.changed_tiles_list = []

    def initialize_environment(self):
        for (x, y), terrain_code in np.ndenumerate(self.heightmap):
            self.terrain_index_grid[x, y] = terrain_code
            terrain_info = self.terrain_definitions.get(terrain_code)
            if terrain_info:
                terrain_class = terrain_info['class']
                entity_prob = terrain_info['entity_prob']
                # Instantiate the terrain with its corresponding properties
                self.terrain_object_grid[x, y] = terrain_class(x, y, self.tile_size, entity_prob)
                # Add entity to terrain if entity_prob is met
                self.init_entity(self.terrain_object_grid[x, y], x, y)

            else:
                raise ValueError(f"Invalid terrain code: {terrain_code}")
    
    def init_entity(self, terrain, x, y):
        if random.random() < terrain.entity_prob:
            entity_type = terrain.entity_type
            entity = entity_type(x, y, self.tile_size)
            self.entity_group.add(entity)
            self.entity_index_grid[x, y] = entity.id
            self.terrain_object_grid[x, y].add_entity(entity)
        else:
            if isinstance(terrain, Plains): # terrain.evelation == isinstance(terrain, Plains):
                self.suitable_terrain_locations['Plains'].append((x, y))
            if isinstance(terrain, Hills):
                self.suitable_terrain_locations['Hills'].append((x, y))
            if isinstance(terrain, Mountains):
                self.suitable_terrain_locations['Mountains'].append((x, y))
            elif isinstance(terrain, Snow):
                self.suitable_terrain_locations['Snow'].append((x, y))

    def add_outposts(self):
        possible_locations = self.suitable_terrain_locations['Plains'] + self.suitable_terrain_locations['Hills']
        selected_locations = random.sample(possible_locations, min(len(possible_locations), self.number_of_outposts))

        for x, y in selected_locations:
            outpost = Outpost(x, y, self.tile_size)
            self.entity_group.add(outpost)
            self.terrain_object_grid[x, y].add_entity(outpost)
            self.outpost_locations.append((x, y))

            # remove the location from the list of suitable locations
            if (x, y) in self.suitable_terrain_locations['Plains']:
                self.suitable_terrain_locations['Plains'].remove((x, y))
            if (x, y) in self.suitable_terrain_locations['Hills']:
                self.suitable_terrain_locations['Hills'].remove((x, y))
            if (x, y) in self.suitable_terrain_locations['Mountains']:
                self.suitable_terrain_locations['Mountains'].remove((x, y))
            if (x, y) in self.suitable_terrain_locations['Snow']:
                self.suitable_terrain_locations['Snow'].remove((x, y))

        return possible_locations

    def init_player(self):
        location = random.choice(self.suitable_terrain_locations['Plains'] + self.suitable_terrain_locations['Hills'])
        player = Player(location[0], location[1], self.tile_size)
        self.entity_group.add(player, layer=2)
        self.entity_index_grid[location[0], location[1]] = player.id
        return player
    
    def update_terrain_passability(self, x, y, entity):
        terrain = self.terrain_object_grid[x, y]
        if isinstance(entity, WoodPath):
            terrain.passable = True
            terrain.energy_requirement = max(0, terrain.energy_requirement - 2)

    def move_entity(self, entity, dx, dy):
        new_x, new_y = entity.grid_x + dx, entity.grid_y + dy
        if not self.is_move_valid(new_x, new_y):
            return
        old_x, old_y = entity.grid_x, entity.grid_y

        self.terrain_object_grid[old_x, old_y].remove_entity()
        entity.move(dx, dy)
        self.terrain_object_grid[new_x, new_y].add_entity(entity)
        # We never remove the entity from the entity_group, so no need to re-add it
        self.environment_changed(old_x, old_y, new_x, new_y)
    
    def delete_entity(self, entity):
        x, y = entity.grid_x, entity.grid_y
        self.terrain_object_grid[x, y].remove_entity()
        self.entity_group.remove(entity)
        self.entity_index_grid[x, y] = 0
        self.single_environment_changed(x, y)

    def is_move_valid(self, x: int, y: int) -> bool:
        return self.within_bounds(x, y) and self.terrain_object_grid[x, y].passable

    def within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
    
    def environment_changed(self, old_x, old_y, new_x, new_y):
        self.environment_changed_flag = True
        self.changed_tiles_list.append((old_x, old_y))
        self.changed_tiles_list.append((new_x, new_y))

    def single_environment_changed(self, x, y):
        self.environment_changed_flag = True
        self.changed_tiles_list.append((x, y))

    def place_path(self, x, y):
        wood_path = WoodPath(x, y, self.tile_size)
        self.entity_group.add(wood_path)
        self.entity_index_grid[x, y] = wood_path.id
        self.terrain_object_grid[x, y].add_path(wood_path)
        self.single_environment_changed(x, y)

    def drop_rock_in_water(self, x, y, fill_type):
        if fill_type == 0:
            self.terrain_object_grid[x, y].shallow()
        elif fill_type == 1:
            self.terrain_object_grid[x, y].land_fill()
        self.single_environment_changed(x, y)

    def get_neighbours(self, x, y):
        neighbours = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if self.within_bounds(new_x, new_y):
                    neighbours.append((new_x, new_y))
        return neighbours

    def environment_gamestate(self):
        return self.terrain_index_grid, self.entity_index_grid
    
    def print_entities(self):
        # number_of_fish = 0
        number_of_trees = 0
        number_of_mossy_rocks = 0
        number_of_snowy_rocks = 0
        number_of_outposts = 0
        number_of_wood_paths = 0
        number_of_players = 0

        for entity in self.entity_group:
            if isinstance(entity, Player):
                number_of_players += 1
            elif isinstance(entity, Outpost):
                number_of_outposts += 1
            elif isinstance(entity, WoodPath):
                number_of_wood_paths += 1
            # elif isinstance(entity, Fish):
            #     number_of_fish += 1
            elif isinstance(entity, Tree):
                number_of_trees += 1
            elif isinstance(entity, MossyRock):
                number_of_mossy_rocks += 1
            elif isinstance(entity, SnowyRock):
                number_of_snowy_rocks += 1

        print(f"Players: {number_of_players}")
        print(f"Outposts: {number_of_outposts}")
        print(f"Wood Paths: {number_of_wood_paths}")
        print(f"Fish: {number_of_fish}")
        print(f"Trees: {number_of_trees}")
        print(f"Mossy Rocks: {number_of_mossy_rocks}")
        print(f"Snowy Rocks: {number_of_snowy_rocks}")
        