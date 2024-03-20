from random import choice
import pygame
import numpy as np

from environment import Environment

class AgentModel:
    def __init__(self, environment : Environment):
        self.environment = environment
        self.agent = self.environment.player
        self.running = True

    def agent_move(self):
        (dx, dy) = choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.environment.move_entity(self.agent, dx, dy)