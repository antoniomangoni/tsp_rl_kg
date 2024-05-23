import pygame

from environment import Environment
from agent import Agent

class Renderer:
    def __init__(self, environment: Environment, agent_control: Agent):
        self.environment = environment
        self.agent = agent_control  # Reference to the agent to access its status and inventory
        self.tile_size = environment.tile_size
        self.window_width = environment.width * self.tile_size
        self.window_height = environment.height * self.tile_size

        self.heatmap_colour = (255, 0, 0)  # Colour for the heatmap overlay is red

        self.ui_height = 100  # Height for the UI panel at the bottom
        self.surface = pygame.display.set_mode((self.window_width, self.window_height + self.ui_height))
        
        self.terrain_surface = pygame.Surface((self.window_width, self.window_height))
        self.terrain_surface.set_alpha(None)

    def init_render(self):
        # Draw the entire terrain onto the terrain surface
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                terrain_tile = self.environment.terrain_object_grid[x, y]
                self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))
        
        # Initial blit of the terrain surface onto the main surface
        self.surface.blit(self.terrain_surface, (0, 0))
        # Draw all entities for the first time
        self.environment.entity_group.draw(self.surface)
        self.render_ui()
        pygame.display.flip()

    def render_updated_tiles(self):
        if not self.environment.environment_changed_flag:
            return

        # Go through the list of changed tiles and update them
        for x, y in self.environment.changed_tiles_list:
            self.update_tile(x, y)
            # Blit the updated terrain tile onto the main surface
            rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
            self.surface.blit(self.terrain_surface, rect.topleft, rect)
        
        # Now redraw entities that are within or intersect the updated tiles.
        # This is a simplified approach. A more optimized method would check for actual intersections.
        self.environment.entity_group.draw(self.surface)

        # Finally, update the display only for the dirty rects
        dirty_rects = [pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size) for x, y in self.environment.changed_tiles_list]
        pygame.display.update(dirty_rects)

        # Clear the list of changed tiles after updating
        self.environment.changed_tiles_list.clear()
        self.environment.environment_changed_flag = False
        self.render_changed_ui()

    def update_tile(self, x, y):
        # Directly access and redraw the terrain tile
        terrain_tile = self.environment.terrain_object_grid[x, y]
        self.terrain_surface.blit(terrain_tile.image, (x * self.tile_size, y * self.tile_size))

        # Redraw the entity if present on this tile
        if terrain_tile.entity_on_tile is not None:
            self.surface.blit(terrain_tile.entity_on_tile.image, (x * self.tile_size, y * self.tile_size))

    def render_heatmap(self, max_intensity, bool_heatmap=False):
        if not bool_heatmap:
            return

        for x in range(self.environment.width):
            for y in range(self.environment.height):
                intensity = self.environment.heat_map[x, y]
                if intensity > 0:
                    alpha = int((intensity / max_intensity) * 255)  # Scale intensity to 0-255 range
                    color = (*self.heatmap_colour[:3], alpha)  # Add alpha to the heatmap color
                    heat_rect = pygame.Surface((self.tile_size, self.tile_size))
                    heat_rect.set_alpha(alpha)
                    heat_rect.fill(color[:3])
                    self.surface.blit(heat_rect, (x * self.tile_size, y * self.tile_size))
        pygame.display.update()

    def render_ui(self):
        # First, clear the UI area to ensure a clean slate for UI rendering
        self.clear_ui_area()
        
        # Render the status bars for energy, hunger, and thirst
        self.render_status_bars()
        
        # Render the inventory text
        self.render_inventory_text()

    def render_changed_ui(self):
        # Update the status bars and inventory text
        self.render_status_bars()
        self.render_inventory_text()

    def clear_ui_area(self):
        # Fill the UI background to clear previous frame's UI elements
        pygame.draw.rect(self.surface, (0, 0, 0), (0, self.window_height, self.window_width, self.ui_height))

    def render_status_bars(self):
        # Define bar properties
        bars_info = [
            (self.agent.energy, self.agent.energy_max, (255, 0, 0), "Energy"),
            (self.agent.hunger, self.agent.hunger_thirst_max, (0, 255, 0), "Hunger"),
            (self.agent.thirst, self.agent.hunger_thirst_max, (0, 0, 255), "Thirst")
        ]
        y_offset = 10
        for value, max_value, color, label in bars_info:
            self.render_bar(20, self.window_height + y_offset, value, max_value, color, label)
            y_offset += 25

    def render_inventory_text(self):
        # Render inventory items as text
        inventory_items = [
            ('fish', self.agent.fish),
            ('water', self.agent.water),
            ('wood', self.agent.wood),
            ('stone', self.agent.stone)
        ]
        inventory_start_x = 300
        inventory_start_y = self.window_height + 10
        font = pygame.font.Font(None, 24)
        
        for index, (item, quantity) in enumerate(inventory_items):
            text = font.render(f"{item.capitalize()}: {quantity}", True, (255, 255, 255))
            self.surface.blit(text, (inventory_start_x, inventory_start_y + index * 25))

    def render_bar(self, x, y, value, max_value, color, label):
        # Draw the status bar for a single attribute
        bar_length = 200
        bar_height = 20
        fill_length = (value / max_value) * bar_length
        # Background bar
        pygame.draw.rect(self.surface, (255, 255, 255), (x, y, bar_length, bar_height), 2)
        # Filled bar
        pygame.draw.rect(self.surface, color, (x, y, fill_length, bar_height))
        # Label text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, (255, 255, 255))
        self.surface.blit(text, (x + bar_length + 5, y))
