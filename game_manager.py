import numpy as np
import pygame
from heightmap_generator import HeightmapGenerator
from environment import Environment
from agent_model import AgentModel
from renderer import Renderer
from target import Target_Manager

class GameManager:
    def __init__(self, map_size=3, tile_size=200):
        self.map_size = map_size
        self.tile_size = 1000 // map_size # 1000 is the maximum window size, so we want to scale the tile size to fit the window
        self.environment = None
        # self.player = None

        self.agent_model = None
        self.agent = None
        self.target_manager = None

        self.renderer = None
        self.running = True

        self.init_pygame()
        self.load_resources()
        self.initialize_components()
        self.target_path_energy = self.target_manager.get_energy_required(self.target_manager.shortest_path)
        print(f"Energy required for the target path: {self.target_path_energy}")
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
        self.target_manager = Target_Manager(self.environment)

    def initialise_rendering(self):
        self.renderer = Renderer(self.environment)
        self.screen = pygame.display.set_mode((self.map_size * self.tile_size, self.map_size * self.tile_size))
        self.renderer.init_render()

    def handle_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        self.agent_model.agent_action(11)
        # self.agent_model.agent_move()

    def render(self):
        self.renderer.render()
        pygame.display.flip()

    def run(self):
        self.initialise_rendering()
        while self.running:
            # pygame.time.delay(500)
            self.handle_keyboard()
            self.update()
            self.renderer.render_updated_tiles()

        pygame.quit()
        print('Game closed')
        self.environment.print_entities()

if __name__ == '__main__':
    game_manager = GameManager()
    game_manager.run()