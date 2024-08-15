import numpy as np
import pygame
import random

from heightmap_generator import HeightmapGenerator
from environment import Environment
from agent import Agent
from renderer import Renderer
from target import Target_Manager

from knowledge_graph import KnowledgeGraph

class GameManager:
    def __init__(self, num_tiles=32, screen_size=800, vision_range=2, plot=False):
        self.num_tiles = num_tiles
        self.tile_size: int = screen_size // num_tiles
        self.environment = None
        self.agent_controler = None
        self.agent = None
        self.target_manager = None
        self.route_energy_list = []
        self.vision_range = vision_range
        self.renderer = None
        self.running = True
        self.plot = plot

        self.initialize_components()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Game World")

    def initialize_components(self):
        # Generate heightmap
        heightmap_generator = HeightmapGenerator(
            width=self.num_tiles, 
            height=self.num_tiles, 
            scale=10, 
            terrain_thresholds=np.array([0.1, 0.2, 0.5, 0.7, 0.9, 1.0]), 
            octaves=3, persistence=0.2, lacunarity=2.0
        )
        heightmap = heightmap_generator.generate()
        self.environment = Environment(heightmap, self.tile_size, number_of_outposts=3)

        self.agent_controler = Agent(self.environment, self.vision_range)
        self.agent = self.agent_controler.agent
        
        self.target_manager = Target_Manager(self.environment)

    def init_knowledge_graph(self, kg_completeness):
        self.kg_class = KnowledgeGraph(self.environment, self.vision_range, kg_completeness, self.plot)
        self.agent_controler.get_kg(self.kg_class)

    def initialise_rendering(self):
        self.renderer = Renderer(self.environment, self.agent_controler)
        self.screen = pygame.display.set_mode((self.num_tiles * self.tile_size, self.num_tiles * self.tile_size))
        self.renderer.init_render()

    def rerender(self):
        self.renderer.render_updated_tiles()
        # self.renderer.render_heatmap(self.target_manager.min_path_length, bool_heatmap=True)
        pygame.display.flip()

    def start_game(self, kg_completeness=0.5):
        self.init_pygame()
        self.init_knowledge_graph(kg_completeness)
        self.initialise_rendering()

    def end_game(self):
        self.running = False
        pygame.quit()


    #####################################################################################
    #   This is the main game loop that runs the game when the model is not being used  #
    #####################################################################################
 
    def game_step(self):
        self.agent_controler.agent_action(random.randint(0, 10))
        # self.environment.update_heat_map(self.agent.grid_x, self.agent.grid_y, self.target_manager.min_path_length)
        self.rerender()

    def run(self):
        self.start_game()
        while self.running:
            self.game_step()
            # pygame.time.wait(1000)
            # save the surface to an image
            pygame.image.save(self.screen, "small_game_background.jpeg")
            exit()

        pygame.quit()
        print('Game closed')
        self.environment.print_environment()
        self.kg_class.visualise_graph()