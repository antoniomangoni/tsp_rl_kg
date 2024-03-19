from random import choice
import pygame
import numpy as np

from environment import Environment

class AgentModel:
    def __init__(self):
        self.environment = Environment()
        self.agent = self.environment.player
        self.running = True

    def random_move(self):
        new_coords = self.environment.player.random_move()
        self.environment.move_entity(self.agent, *new_coords)