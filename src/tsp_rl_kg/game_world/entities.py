import pygame
import os
from pathlib import Path

class Entity(pygame.sprite.Sprite):
    _images = {}
    
    def __init__(self, x, y, art, tile_size):
        super().__init__()
        if art not in self._images:
            module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            
            possible_paths = [
                module_dir / "assets" / "pixel_art" / art,                 # Within the module
                module_dir.parent / "assets" / "pixel_art" / art,          # One level up
                module_dir.parent.parent / "assets" / "pixel_art" / art,   # Two levels up (package parent)
                module_dir.parent.parent.parent / "assets" / "pixel_art" / art,  # Three levels up (repository root)
                Path("assets") / "pixel_art" / art                         # Relative to working directory
            ]
                        
            for path in possible_paths:
                if path.exists():
                    self._images[art] = pygame.transform.scale(
                        pygame.image.load(str(path)), 
                        (tile_size, tile_size)
                    )
                    break
            else:  # This runs if no paths were found
                paths_str = "\n - ".join(str(p) for p in possible_paths)
                raise FileNotFoundError(f"Did not find image: {art}. Tried:\n - {paths_str}")


        
        self.image = self._images[art]
        self.grid_x = x
        self.grid_y = y
        self.tile_size = tile_size
        self.screen_x = x * tile_size
        self.screen_y = y * tile_size
        self.rect = self.image.get_rect()
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y
        self.id = None

    def move(self, dx, dy):
        # Update the logical grid position
        self.grid_x += dx
        self.grid_y += dy
        # Update the screen (pixel) position based on the new grid position
        self.screen_x = self.grid_x * self.tile_size
        self.screen_y = self.grid_y * self.tile_size
        # Update the rect position for drawing and collision detection
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y

class Player(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='player.png', tile_size=tile_size)
        self.id = 7
        self.name = 'Player'

class Outpost(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='outpost_2.png', tile_size=tile_size)
        self.id = 5
        self.name = 'Outpost'

class WoodPath(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='wood_path.png', tile_size=tile_size)
        self.id = 6
        self.name = 'Wood Path'

class Fish(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='fish.png', tile_size=tile_size)
        self.id = 1
        self.name = 'Fish'

class Tree(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='tree_1.png', tile_size=tile_size)
        self.id = 2
        self.name = 'Tree'

class Rock(Entity):
    def __init__(self, x, y, art, tile_size):
        super().__init__(x, y, art, tile_size)

class MossyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='rock_moss.png', tile_size=tile_size)
        self.id = 3
        self.name = 'Mossy Rock'

class SnowyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art='rock_snow.png', tile_size=tile_size)
        self.id = 4
        self.name = 'Snowy Rock'


