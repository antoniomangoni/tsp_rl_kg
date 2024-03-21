from random import choice
import pygame
import numpy as np

from environment import Environment
from entities import Fish, Tree, MossyRock, SnowyRock
from terrains import DeepWater, Water

class AgentModel:
    def __init__(self, environment : Environment):
        self.environment = environment
        self.agent = self.environment.player
        self.running = True

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
        # print(f'Action: {action}')
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            if self.stone >= 1:
                action = 3
                print("Placing rock")
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
        self.energy = min(100, self.energy)

    def build_path(self):
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water):
            return
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            return
        if self.wood >= 1:
            self.wood -= 1
            self.environment.place_path(self.agent.grid_x, self.agent.grid_y)

    def place_rock(self):
        if self.stone >= 1:
            self.stone -= 1
            self.environment.place_rock(self.agent.grid_x, self.agent.grid_y)

    def collect_resource(self, dx, dy):
        x, y = self.agent.grid_x + dx, self.agent.grid_y + dy
        if (x < 0 or x >= self.environment.width) or (y < 0 or y >= self.environment.height):
            return
        resource = self.environment.terrain_object_grid[x, y].entity_on_tile
        collected = False
        if isinstance(resource, Fish):
            if self.fish < 5:
                self.fish += 1
                collected = True
        elif isinstance(resource, Tree):
            if self.wood < 5:
                self.wood += 1
                collected = True
        elif isinstance(resource, MossyRock):
            if self.stone < 5:
                self.stone += 1
                collected = True
        elif isinstance(resource, SnowyRock):
            if self.stone < 5:
                self.stone += 1
                collected = True
        if collected:
            self.environment.delete_entity(resource)
            self.energy -= self.environment.terrain_object_grid[x, y].energy_requirement

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