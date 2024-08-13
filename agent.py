from environment import Environment
from entities import Tree, MossyRock, SnowyRock, Outpost, WoodPath
from terrains import DeepWater, Water
from knowledge_graph import KnowledgeGraph as KG

class Agent:
    def __init__(self, environment : Environment, vision_range : int):
        self.environment = environment
        self.terrain_id_grid = self.environment.terrain_index_grid
        self.entity_id_grid = self.environment.entity_index_grid
        self.kg = None
        self.agent = self.environment.player

        self.resouce_max = 5
        self.vision_range = vision_range 

        self.energy_spent = 0
        self.action_energy_cost = 3

        self.wood = 0
        self.stone = 0
        self.movement_actions = {
            0: (-1, 0),  # Left
            1: (1, 0),   # Right
            2: (0, 1),   # Up
            3: (0, -1),  # Down
        }
        self.other_actions = {
            4: self.scout,
            5: self.build_path,
            6: self.place_rock,
            7: lambda: self.collect_resource(0, 1),
            8: lambda: self.collect_resource(0, -1),
            9: lambda: self.collect_resource(1, 0),
            10: lambda: self.collect_resource(-1, 0),
        }
    
    def reset_agent(self):
        self.wood = 0
        self.stone = 0
        self.energy_spent = 0
        self.agent = self.environment.player

    def get_kg(self, kg : KG):
        self.kg = kg

    def agent_action(self, action):
        if action in self.movement_actions:
            dx, dy = self.movement_actions[action]
            self.move_agent(dx, dy)
        elif action in self.other_actions:
            self.other_actions[action]()
            self.energy_spent += self.action_energy_cost
        else:
            raise ValueError("Invalid action")
        
    def reset_energy_spent(self):
        self.energy_spent = 0

    def move_agent(self, dx, dy):
        new_x, new_y = self.environment.move_entity(self.agent, dx, dy)
        self.kg.move_player_node(new_x, new_y)
        self.energy_spent += self.environment.terrain_object_grid[new_x, new_y].energy_requirement

    def scout(self):
        """ Looking at the environment is a deliberate action. """
        """ Adding a terrain node automatically adds the corresponding entity node"""

        discovered_now = 0
        vision = self.vision_range * 2
        
        for y in range(self.agent.grid_y - vision, self.agent.grid_y + vision + 1):
            for x in range(self.agent.grid_x - vision, self.agent.grid_x + vision + 1):
                if self.environment.within_bounds(x, y):
                    boo = self.kg.discover_this_coordinate(x, y)
                    if boo:
                        discovered_now += 1

        return discovered_now

    def build_path(self):
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], Water):
            return
        if isinstance(self.environment.terrain_object_grid[self.agent.grid_x, self.agent.grid_y], DeepWater):
            return
        if self.wood >= 1:
            self.wood -= 1
            self.environment.place_path(self.agent.grid_x, self.agent.grid_y)
            self.kg.build_path_node(self.agent.grid_x, self.agent.grid_y)

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
                if self.wood >= self.resouce_max:
                    return
                self.wood += 1
            elif isinstance(resource, MossyRock):
                if self.stone >= self.resouce_max:
                    return
                self.stone += 1
            elif isinstance(resource, SnowyRock):
                if self.stone >= self.resouce_max:
                    return
                self.stone += 1
            self.environment.delete_entity(resource)
            self.kg.remove_entity_node(x, y)

"""
These are not implemented as they do not fall within the scope of the project.
They have been left here for future reference.


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

"""
