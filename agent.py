from random import choice
import pygame
import numpy as np

from environment import Environment
from entities import Fish, Tree, MossyRock, SnowyRock, Outpost, WoodPath
from terrains import DeepWater, Water

class Agent:
    def __init__(self, environment : Environment):
        self.environment = environment
        self.agent = self.environment.player
        self.running = True

        self.energy_max = 100
        self.resouce_max = 5
        self.hunger_thirst_max = 20

        self.energy = 100 # Decrease when moving or collecting resources, increase when resting
        self.hunger = 0 # Eat fish to derease
        self.thirst = 0 # Drink water to decrease
        self.wood = 0
        self.stone = 0
        self.fish = 0
        self.water = 0

    def agent_step(self):
        self.energy -= self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y].energy_requirement
        self.hunger += 1
        self.thirst += 1
        if self.energy <= 0 or self.hunger >= 20 or self.thirst >= 20:
            self.running = False
        if self.running:
            self.agent_action(self.agent_model.predict(self.environment.terrain_object_grid, self.agent.grid_x, self.agent.grid_y))

    def agent_action(self, action):
        action = choice(range(11))
        if action == 0:
            self.agent_move()
        elif action == 1:
            self.rest()
        elif action == 2:
            self.build_path()
        elif action == 3:
            self.place_rock()
        elif action == 4:
            self.collect_resource(0, 1)
        elif action == 5:
            self.collect_resource(0, -1)
        elif action == 6:
            self.collect_resource(1, 0)
        elif action == 7:
            self.collect_resource(-1, 0)
        elif action == 8:
            self.collect_water()
        elif action == 9:
            self.drink()
        elif action == 10:
            self.eat()

    def agent_move(self):
        (dx, dy) = choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.environment.move_entity(self.agent, dx, dy)

    def rest(self):
        self.energy += 20
        self.energy = min(self.energy_max, self.energy)

    def build_path(self):
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water):
            return
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            return
        if self.wood >= 1:
            self.wood -= 1
            self.environment.place_path(self.agent.grid_x, self.agent.grid_y)

    def place_rock(self):
        if self.stone < 1:
            return
        place = -1
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            place = 0
        elif isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water):
            place = 1
        else:
            return
        self.stone -= 1
        # print(f'Placing - Stone inventory: {self.stone}')
        self.environment.drop_rock_in_water(self.agent.grid_x, self.agent.grid_y, place)

    def collect_resource(self, dx, dy):
        x, y = self.agent.grid_x + dx, self.agent.grid_y + dy
        if self.environment.within_bounds(x, y) is False:
            return
        resource = self.environment.terrain_object_grid[x, y].entity_on_tile
        if resource is None or isinstance(resource, Outpost) or isinstance(resource, WoodPath):
            return
        else:
            if isinstance(resource, Fish):
                if self.fish >= self.resouce_max:
                    return
                self.fish += 1
            elif isinstance(resource, Tree):
                if self.wood >= self.resouce_max:
                    return
                self.wood += 1
            elif isinstance(resource, MossyRock):
                if self.stone >= self.resouce_max:
                    return
                self.stone += 1
                # print(f'Collecting - Stone inventory: {self.stone}')
            elif isinstance(resource, SnowyRock):
                if self.stone >= self.resouce_max:
                    return
                self.stone += 1
            self.environment.delete_entity(resource)

    def collect_water(self):
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water):
            self.water += 1
        elif isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            self.water += 2

    def drink(self):
        self.rest()
        if self.water > 0:
            self.water -= 1
            self.thirst = max(0, self.thirst - 5)

    def eat(self):
        self.rest()
        if self.fish > 0:
            self.fish -= 1
            self.hunger = max(0, self.hunger - 5)