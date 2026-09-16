from importlib.resources import as_file, files

import pygame

# Canonical entity IDs used in the knowledge-graph node features.
ENTITY_ID_FISH = 1
ENTITY_ID_TREE = 2
ENTITY_ID_MOSSY_ROCK = 3
ENTITY_ID_SNOWY_ROCK = 4
ENTITY_ID_OUTPOST = 5
ENTITY_ID_WOOD_PATH = 6
ENTITY_ID_PLAYER = 7


class BaseEntity:
    """Lightweight entity base with position/id/name - no pygame dependency."""

    def __init__(self, x, y, tile_size, *, headless=False):
        self._headless = headless
        self.grid_x = x
        self.grid_y = y
        self.tile_size = tile_size
        self.screen_x = x * tile_size
        self.screen_y = y * tile_size
        self.id = None
        self.name = None

    def move(self, dx, dy):
        self.grid_x += dx
        self.grid_y += dy
        self.screen_x = self.grid_x * self.tile_size
        self.screen_y = self.grid_y * self.tile_size


class Entity(BaseEntity, pygame.sprite.Sprite):
    """Renderable entity. Loads sprite images when not in headless mode."""

    _images = {}

    def __init__(self, x, y, art, tile_size, *, headless=False):
        BaseEntity.__init__(self, x, y, tile_size, headless=headless)

        if self._headless:
            return

        pygame.sprite.Sprite.__init__(self)
        key = (art, tile_size)
        if key not in self._images:
            resource = files("tsp_rl_kg").joinpath("assets", "pixel_art", art)
            with as_file(resource) as path:
                self._images[key] = pygame.transform.scale(
                    pygame.image.load(str(path)), (tile_size, tile_size)
                )
        self.image = self._images[key]
        self.rect = self.image.get_rect()
        self.rect.x = self.screen_x
        self.rect.y = self.screen_y

    def move(self, dx, dy):
        super().move(dx, dy)
        if not self._headless:
            self.rect.x = self.screen_x
            self.rect.y = self.screen_y


class Player(Entity):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="player.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_PLAYER
        self.name = "Player"


class Outpost(Entity):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="outpost_2.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_OUTPOST
        self.name = "Outpost"


class WoodPath(Entity):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="wood_path.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_WOOD_PATH
        self.name = "Wood Path"


class Fish(Entity):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="fish.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_FISH
        self.name = "Fish"


class Tree(Entity):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="tree_1.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_TREE
        self.name = "Tree"


class Rock(Entity):
    def __init__(self, x, y, art, tile_size, *, headless=False):
        super().__init__(x, y, art, tile_size, headless=headless)


class MossyRock(Rock):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="rock_moss.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_MOSSY_ROCK
        self.name = "Mossy Rock"


class SnowyRock(Rock):
    def __init__(self, x, y, tile_size, *, headless=False):
        super().__init__(x, y, art="rock_snow.png", tile_size=tile_size, headless=headless)
        self.id = ENTITY_ID_SNOWY_ROCK
        self.name = "Snowy Rock"
