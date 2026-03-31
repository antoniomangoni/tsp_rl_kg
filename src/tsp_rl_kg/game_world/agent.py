from tsp_rl_kg.config import AgentConfig
from tsp_rl_kg.game_world.actions import COLLECT_DELTAS, MOVEMENT_DELTAS, ActionType
from tsp_rl_kg.game_world.entities import MossyRock, Outpost, SnowyRock, Tree, WoodPath
from tsp_rl_kg.game_world.environment import Environment
from tsp_rl_kg.game_world.terrains import DeepWater, Water
from tsp_rl_kg.knowledge.knowledge_graph import KnowledgeGraph as KG


class Agent:
    def __init__(
        self,
        environment: Environment,
        vision_range: int,
        agent_config: AgentConfig | None = None,
    ):
        if agent_config is None:
            agent_config = AgentConfig()
        self.environment = environment
        self.terrain_id_grid = self.environment.terrain_index_grid
        self.entity_id_grid = self.environment.entity_index_grid
        self.kg = None
        self.agent = self.environment.player
        self.agent_step_count = 0

        self.resource_max = agent_config.resource_max
        self.vision_range = vision_range

        self.energy_spent = 0
        self.action_energy_cost = agent_config.action_energy_cost
        self._scout_vision_multiplier = agent_config.scout_vision_multiplier

        self.wood = 0
        self.stone = 0

    def reset_agent(self):
        self.reset_energy_spent()
        self.wood = 0
        self.stone = 0
        self.agent = self.environment.player

    def get_kg(self, kg: KG):
        self.kg = kg

    def agent_action(self, action: int) -> None:
        self.agent_step_count += 1
        action = ActionType(action)
        if action in MOVEMENT_DELTAS:
            dx, dy = MOVEMENT_DELTAS[action]
            self.move_agent(dx, dy)
        elif action in COLLECT_DELTAS:
            dx, dy = COLLECT_DELTAS[action]
            self.collect_resource(dx, dy)
            self.energy_spent += self.action_energy_cost
        elif action is ActionType.SCOUT:
            self.scout()
            self.energy_spent += self.action_energy_cost
        elif action is ActionType.BUILD_PATH:
            self.build_path()
            self.energy_spent += self.action_energy_cost
        elif action is ActionType.PLACE_ROCK:
            self.place_rock()
            self.energy_spent += self.action_energy_cost
        else:
            raise ValueError(f"Invalid action: {action}")

    def reset_energy_spent(self):
        self.energy_spent = 0
        self.agent_step_count = 0

    def move_agent(self, dx, dy):
        new_x, new_y = self.environment.move_entity(self.agent, dx, dy)
        self.kg.move_player_node(new_x, new_y)
        self.energy_spent += self.environment.terrain_object_grid[new_x, new_y].energy_requirement

    def scout(self):
        """Looking at the environment is a deliberate action."""
        """ Adding a terrain node automatically adds the corresponding entity node"""

        discovered_now = 0
        vision = int(self.vision_range * self._scout_vision_multiplier)

        for y in range(self.agent.grid_y - vision, self.agent.grid_y + vision + 1):
            for x in range(self.agent.grid_x - vision, self.agent.grid_x + vision + 1):
                if self.environment.within_bounds(x, y):
                    newly_discovered = self.environment.discover_coordinate(x, y)
                    if newly_discovered:
                        discovered_now += 1

        return discovered_now

    def build_path(self):
        if (self.agent.grid_x, self.agent.grid_y) in self.environment.outpost_locations:
            return
        if isinstance(
            self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water
        ):
            return
        if isinstance(
            self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater
        ):
            return
        if self.wood >= 1:
            self.wood -= 1
            self.environment.place_path(self.agent.grid_x, self.agent.grid_y)
            self.kg.build_path_node(self.agent.grid_x, self.agent.grid_y)

    def place_rock(self):
        if self.stone < 1:
            return
        place = -1
        if isinstance(
            self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater
        ):
            place = 0
        elif isinstance(
            self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water
        ):
            place = 1
        else:
            return
        self.stone -= 1
        self.environment.drop_rock_in_water(self.agent.grid_x, self.agent.grid_y, place)
        self.kg.elevate_terrain_node(self.agent.grid_x, self.agent.grid_y)

    def collect_resource(self, dx, dy):
        x, y = self.agent.grid_x + dx, self.agent.grid_y + dy
        assert (x, y) != (self.agent.grid_x, self.agent.grid_y)
        if self.environment.within_bounds(x, y) is False:
            return
        if self.entity_id_grid[x, y] == 0:
            return
        resource = self.environment.terrain_object_grid[x, y].entity_on_tile
        if resource is None or isinstance(resource, Outpost) or isinstance(resource, WoodPath):
            return
        else:
            if isinstance(resource, Tree):
                if self.wood >= self.resource_max:
                    return
                self.wood += 1
            elif isinstance(resource, MossyRock):
                if self.stone >= self.resource_max:
                    return
                self.stone += 1
            elif isinstance(resource, SnowyRock):
                if self.stone >= self.resource_max:
                    return
                self.stone += 1
            self.environment.delete_entity(resource)
            self.kg.remove_entity_node(x, y)
