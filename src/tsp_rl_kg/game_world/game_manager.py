import random

import numpy as np
import pygame
from loguru import logger

from tsp_rl_kg.config import GameManagerConfig
from tsp_rl_kg.game_world.actions import ActionType
from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.game_world.heightmap_generator import HeightmapGenerator
from tsp_rl_kg.graph.projection import CompletenessProjection, ProjectionPolicy
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph
from tsp_rl_kg.renderer import Renderer
from tsp_rl_kg.rl.target import Target_Manager


class GameManager:
    KEY_TO_ACTION: dict[int, ActionType] = {
        pygame.K_a: ActionType.MOVE_LEFT,
        pygame.K_d: ActionType.MOVE_RIGHT,
        pygame.K_w: ActionType.MOVE_UP,
        pygame.K_s: ActionType.MOVE_DOWN,
        pygame.K_q: ActionType.SCOUT,
        pygame.K_e: ActionType.BUILD_PATH,
        pygame.K_r: ActionType.PLACE_ROCK,
        pygame.K_i: ActionType.COLLECT_UP,
        pygame.K_k: ActionType.COLLECT_DOWN,
        pygame.K_l: ActionType.COLLECT_RIGHT,
        pygame.K_j: ActionType.COLLECT_LEFT,
    }

    def __init__(
        self,
        config: GameManagerConfig | None = None,
        plot: bool = False,
        feature_encoder=None,
        # Legacy positional args — prefer config
        num_tiles: int | None = None,
        screen_size: int | None = None,
        vision_range: int | None = None,
    ):
        if config is not None:
            self._config = config
        else:
            self._config = GameManagerConfig(
                num_tiles=num_tiles if num_tiles is not None else 32,
                screen_size=screen_size if screen_size is not None else 800,
                vision_range=vision_range if vision_range is not None else 2,
            )
        self.num_tiles = self._config.num_tiles
        self.tile_size: int = self._config.screen_size // self._config.num_tiles
        self.environment = None
        self.agent_controler = None
        self.agent = None
        self.target_manager = None
        self.route_energy_list = []
        self.renderer = None
        self.running = True
        self.plot = plot
        self.feature_encoder = feature_encoder
        self.headless = self._config.headless
        self.human_mode = self._config.human_mode
        self.use_random_human_actions = self._config.use_random_human_actions
        self.target_fps = self._config.target_fps
        self.clock: pygame.time.Clock | None = None
        self.vision_range = self._config.vision_range
        self.initialize_components()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Game World")
        self.clock = pygame.time.Clock()

    def initialize_components(self):
        # Generate heightmap
        heightmap_generator = HeightmapGenerator(
            width=self.num_tiles,
            height=self.num_tiles,
            scale=10,
            terrain_thresholds=np.array([0.1, 0.2, 0.5, 0.7, 0.9, 1.0]),
            octaves=3,
            persistence=0.2,
            lacunarity=2.0,
        )
        heightmap = heightmap_generator.generate()
        self.environment = Environment(
            heightmap, self.tile_size, number_of_outposts=3, headless=self.headless
        )

        self.agent_controler = Agent(self.environment, self.vision_range)
        self.agent = self.agent_controler.agent

        self.target_manager = Target_Manager(self.environment)

    def init_knowledge_graph(self, projection: ProjectionPolicy):
        self.kg_class = KnowledgeGraph(
            self.environment,
            self.vision_range,
            plot=self.plot,
            feature_encoder=self.feature_encoder,
            projection=projection,
        )
        self.agent_controler.get_kg(self.kg_class)

    def initialise_rendering(self):
        self.renderer = Renderer(self.environment, self.agent_controler)
        self.screen = pygame.display.set_mode(
            (self.num_tiles * self.tile_size, self.num_tiles * self.tile_size)
        )
        self.renderer.init_render()

    def rerender(self):
        if self.headless:
            return
        self.renderer.render_updated_tiles()
        # self.renderer.render_heatmap(self.target_manager.min_path_length, bool_heatmap=True)
        pygame.display.flip()

    def start_game(self, kg_completeness=0.5, projection: ProjectionPolicy | None = None):
        if not self.headless:
            self.init_pygame()
        if projection is None:
            projection = CompletenessProjection(kg_completeness, self.vision_range, self.num_tiles)
        self.init_knowledge_graph(projection)
        if not self.headless:
            self.initialise_rendering()

    def end_game(self):
        self.running = False
        if not self.headless:
            pygame.quit()

    #####################################################################################
    #   This is the main game loop that runs the game when the model is not being used  #
    #####################################################################################

    def game_step(self):
        action: ActionType | None = None
        if self.human_mode:
            action = (
                random.choice(list(ActionType))
                if self.use_random_human_actions
                else self._poll_human_action()
            )
        else:
            action = random.choice(list(ActionType))

        if action is not None:
            self.agent_controler.agent_action(action)
        # self.environment.update_heat_map(
        #     self.agent.grid_x, self.agent.grid_y,
        #     self.target_manager.min_path_length
        # )
        self.rerender()

    def _poll_human_action(self) -> ActionType | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end_game()
                return None
            if event.type == pygame.KEYDOWN and event.key in self.KEY_TO_ACTION:
                return self.KEY_TO_ACTION[event.key]
        return None

    def run(self):
        i = 0
        self.start_game()
        while self.running:
            self.game_step()
            if self.clock is not None:
                self.clock.tick(self.target_fps)
            # pygame.time.wait(1000)
            # save the surface to an image
            if i % 10 == 0:
                pygame.image.save(self.screen, f"game_world_{i}.jpeg")
            i += 1
            # exit()

        pygame.quit()
        logger.info("Game closed")
        self.environment.print_environment()
        self.kg_class.visualise_graph()
