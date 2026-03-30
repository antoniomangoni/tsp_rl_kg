import os
from pathlib import Path

import pygame

# Canonical entity IDs used in the knowledge-graph node features.
ENTITY_ID_FISH = 1
ENTITY_ID_TREE = 2
ENTITY_ID_MOSSY_ROCK = 3
ENTITY_ID_SNOWY_ROCK = 4
ENTITY_ID_OUTPOST = 5
ENTITY_ID_WOOD_PATH = 6
ENTITY_ID_PLAYER = 7


class Entity(pygame.sprite.Sprite):
    _headless = False
    _images = {}

    def __init__(self, x, y, art, tile_size):
        self.grid_x = x
        self.grid_y = y
        self.tile_size = tile_size
        self.screen_x = x * tile_size
        self.screen_y = y * tile_size
        self.id = None

        if self._headless:
            return

        super().__init__()
        if art not in self._images:
            module_dir = Path(os.path.dirname(os.path.abspath(__file__)))

            possible_paths = [
                module_dir / "assets" / "pixel_art" / art,
                module_dir.parent / "assets" / "pixel_art" / art,
                module_dir.parent.parent / "assets" / "pixel_art" / art,
                module_dir.parent.parent.parent / "assets" / "pixel_art" / art,
                Path("assets") / "pixel_art" / art,
            ]

            for path in possible_paths:
                if path.exists():
                    self._images[art] = pygame.transform.scale(
                        pygame.image.load(str(path)),
                        (tile_size, tile_size),
                    )
                    break
            else:
                paths_str = "\n - ".join(str(p) for p in possible_paths)
                raise FileNotFoundError(f"Did not find image: {art}. Tried:\n - {paths_str}")

        self.image = self._images[art]
        self.rect = self.image.get_rect()
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y

    def move(self, dx, dy):
        self.grid_x += dx
        self.grid_y += dy
        self.screen_x = self.grid_x * self.tile_size
        self.screen_y = self.grid_y * self.tile_size
        if not self._headless:
            self.rect.x = self.screen_x
            self.rect.y = self.screen_y


class Player(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="player.png", tile_size=tile_size)
        self.id = ENTITY_ID_PLAYER
        self.name = "Player"


class Outpost(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="outpost_2.png", tile_size=tile_size)
        self.id = ENTITY_ID_OUTPOST
        self.name = "Outpost"


class WoodPath(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="wood_path.png", tile_size=tile_size)
        self.id = ENTITY_ID_WOOD_PATH
        self.name = "Wood Path"


class Fish(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="fish.png", tile_size=tile_size)
        self.id = ENTITY_ID_FISH
        self.name = "Fish"


class Tree(Entity):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="tree_1.png", tile_size=tile_size)
        self.id = ENTITY_ID_TREE
        self.name = "Tree"


class Rock(Entity):
    def __init__(self, x, y, art, tile_size):
        super().__init__(x, y, art, tile_size)


class MossyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="rock_moss.png", tile_size=tile_size)
        self.id = ENTITY_ID_MOSSY_ROCK
        self.name = "Mossy Rock"


class SnowyRock(Rock):
    def __init__(self, x, y, tile_size):
        super().__init__(x, y, art="rock_snow.png", tile_size=tile_size)
        self.id = ENTITY_ID_SNOWY_ROCK
        self.name = "Snowy Rock"
