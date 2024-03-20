import numpy as np
import pygame
from heightmap_generator import HeightmapGenerator
from environment import Environment
from agent_model import AgentModel
from renderer import Renderer

class GameManager:
    def __init__(self, map_size=3, tile_size=200):
        self.map_size = map_size
        self.tile_size = 1000 // map_size # 1000 is the maximum window size, so we want to scale the tile size to fit the window
        self.environment = None
        # self.player = None

        self.agent_model = None
        self.agent = None

        self.renderer = None
        self.running = True

        self.init_pygame()
        self.load_resources()
        self.initialize_components()
        # self.test()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Game World")
    
    def load_resources(self):
        # If you have any resources to load, do it here
        pass

    def initialize_components(self):
        # Generate heightmap
        heightmap_generator = HeightmapGenerator(
            width=self.map_size, 
            height=self.map_size, 
            scale=10, 
            terrain_thresholds=np.array([0.2, 0.38, 0.5, 0.7, 0.9, 1.0]), 
            octaves=3, persistence=0.2, lacunarity=2.0
        )
        heightmap = heightmap_generator.generate()
        self.environment = Environment(heightmap, self.tile_size, number_of_outposts=3)
        self.agent_model = AgentModel(self.environment)
        self.agent = self.agent_model.agent
        self.renderer = Renderer(self.environment)
        self.screen = pygame.display.set_mode((self.map_size * self.tile_size, self.map_size * self.tile_size))

    def test(self):
        for _ in range(100):
            try:
                self.initialize_components()
            except Exception as e:
                print(e)

    def handle_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        self.agent.random_move()   
        self.renderer.update_changed_tiles()

    def render(self):
        self.renderer.render()
        pygame.display.flip()

    def run(self):
        self.renderer.render_terrain()
        # self.render()
        while self.running:
            pygame.time.delay(300)
            self.handle_keyboard()
            self.update()
            self.render()

        pygame.quit()
        print('Game closed')

if __name__ == '__main__':
    game_manager = GameManager()
    game_manager.run()