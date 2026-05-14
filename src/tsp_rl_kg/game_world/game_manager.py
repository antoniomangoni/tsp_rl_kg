import random
from pathlib import Path

import numpy as np
import pygame
from loguru import logger

from tsp_rl_kg.config import GameManagerConfig
from tsp_rl_kg.game_world.actions import ActionType
from tsp_rl_kg.game_world.agent import Agent
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.game_world.heightmap_generator import HeightmapGenerator
from tsp_rl_kg.game_world.play_recorder import PlayRecorder
from tsp_rl_kg.graph.projection import CompletenessProjection, ProjectionPolicy
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph
from tsp_rl_kg.renderer import Renderer
from tsp_rl_kg.rl.target import Target_Manager


class GameManager:
    KEY_TO_ACTION: dict[int, ActionType] = {
        pygame.K_d: ActionType.MOVE_RIGHT,
        pygame.K_a: ActionType.MOVE_LEFT,
        pygame.K_s: ActionType.MOVE_UP,
        pygame.K_w: ActionType.MOVE_DOWN,
        pygame.K_q: ActionType.SCOUT,
        pygame.K_e: ActionType.BUILD_PATH,
        pygame.K_r: ActionType.PLACE_ROCK,
        pygame.K_k: ActionType.COLLECT_UP,
        pygame.K_i: ActionType.COLLECT_DOWN,
        pygame.K_l: ActionType.COLLECT_RIGHT,
        pygame.K_j: ActionType.COLLECT_LEFT,
    }
    HUMAN_CONTROL_LINES: tuple[str, ...] = (
        "Human controls:",
        "  A: move left",
        "  D: move right",
        "  W: move up",
        "  S: move down",
        "  Q: scout",
        "  E: build path",
        "  R: place rock",
        "  I/J/K/L: collect from adjacent tiles",
    )

    def __init__(
        self,
        config: GameManagerConfig | None = None,
        plot: bool = False,
        feature_encoder=None,
        # Legacy positional args - prefer config
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
        self.route_energy_list = []
        self.visited_outposts: set[tuple[int, int]] = set()
        self.route_start_energy = 0
        self.running = True
        self.plot = plot
        self.feature_encoder = feature_encoder
        self.headless = self._config.headless
        self.human_mode = self._config.human_mode
        self.use_random_human_actions = self._config.use_random_human_actions
        self.target_fps = self._config.target_fps
        self.clock: pygame.time.Clock | None = None
        self.vision_range = self._config.vision_range
        self.max_steps = self._config.max_steps
        self.recorder: PlayRecorder | None = None
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
        self.screen = self.renderer.surface
        self.renderer.init_render()
        self.renderer.render_ui(self._build_status())

    # TODO: This is a bit hacky - we should separate the status building
    # from the rendering, but this is a quick way to get the status info
    # into the recorder without tightly coupling it to the renderer.
    def _build_status(self) -> list[dict[str, str | int | float]]:
        agent = self.agent_controler
        current_route_energy = agent.energy_spent - self.route_start_energy
        return [
            {
                "X": self.agent.grid_x,
                "Y": self.agent.grid_y,
                "Energy": agent.energy_spent,
                "Outposts": (
                    f"{len(self.visited_outposts)}/{len(self.environment.outpost_locations)}"
                ),
            },
            {
                "Wood": f"{agent.wood}/{agent.resource_max}",
                "Stone": f"{agent.stone}/{agent.resource_max}",
                "Best route": self.target_manager.target_route_energy,
                "Current route": current_route_energy,
            },
        ]

    def _update_route_tracking(self) -> None:
        """Track outpost visits during human/random play.

        ``Current route`` energy accumulates from ``route_start_energy``; once
        every outpost has been visited the finished route is recorded in
        ``route_energy_list`` and the tracker resets for the next trip.
        """
        position = (self.agent.grid_x, self.agent.grid_y)
        if position not in self.environment.outpost_locations:
            return
        self.visited_outposts.add(position)
        if len(self.visited_outposts) == len(self.environment.outpost_locations):
            route_energy = self.agent_controler.energy_spent - self.route_start_energy
            self.route_energy_list.append(route_energy)
            self.visited_outposts.clear()
            self.route_start_energy = self.agent_controler.energy_spent

    def rerender(self):
        if self.headless:
            return
        self.renderer.render_updated_tiles()
        self.renderer.render_ui(self._build_status())
        # self.renderer.render_heatmap(self.target_manager.min_path_length, bool_heatmap=True)
        pygame.display.flip()

    def start_game(self, kg_completeness=0.5, projection: ProjectionPolicy | None = None):
        if not self.headless:
            self.init_pygame()
        if projection is None:
            projection = CompletenessProjection(kg_completeness, self.vision_range, self.num_tiles)
        self.init_knowledge_graph(projection)
        if self.human_mode and not self.use_random_human_actions:
            for line in self.HUMAN_CONTROL_LINES:
                logger.info(line)
        self.recorder = PlayRecorder(self._config)
        self.recorder.write_run_start(
            player_pos=(self.agent.grid_x, self.agent.grid_y),
            discovered_tiles=PlayRecorder.count_discovered_tiles(self.environment.discovered_grid),
        )
        logger.info(f"Play recorder initialised at {self.recorder.paths.run_dir}")
        if not self.headless:
            self.initialise_rendering()

    def end_game(self):
        self.running = False
        if not self.headless:
            pygame.quit()

    #####################################################################################
    #   This is the main game loop that runs the game when the model is not being used  #
    #####################################################################################

    def game_step(self) -> ActionType | None:
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
            self._update_route_tracking()
        # self.environment.update_heat_map(
        #     self.agent.grid_x, self.agent.grid_y,
        #     self.target_manager.min_path_length
        # )
        self.rerender()
        return action

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
        end_reason = "quit_event"
        self.start_game()
        while self.running and (self.max_steps is None or i < self.max_steps):
            action = self.game_step()
            if self.clock is not None:
                self.clock.tick(self.target_fps)
            # pygame.time.wait(1000)
            # save the surface to an image
            frame_path: str | None = None
            if i % 10 == 0 and not self.headless:
                assert self.recorder is not None
                frame_file = Path(self.recorder.paths.visual_dir) / f"game_world_{i}.jpeg"
                pygame.image.save(self.screen, frame_file.as_posix())
                frame_path = frame_file.as_posix()
            if action is not None and self.recorder is not None:
                self.recorder.append_step(
                    step_index=i,
                    action=action,
                    player_pos=(self.agent.grid_x, self.agent.grid_y),
                    energy_spent=self.agent_controler.energy_spent,
                    wood=self.agent_controler.wood,
                    stone=self.agent_controler.stone,
                    discovered_tiles=PlayRecorder.count_discovered_tiles(
                        self.environment.discovered_grid
                    ),
                    frame_path=frame_path,
                )
            i += 1
            # exit()

        if self.max_steps is not None and i >= self.max_steps:
            end_reason = "max_steps_reached"
        if self.recorder is not None:
            self.recorder.write_run_end(end_reason=end_reason)
        pygame.quit()
        logger.info("Game closed")
        self.environment.print_environment()
        self.kg_class.visualise_graph()
