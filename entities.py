import pygame

class Entity(pygame.sprite.Sprite):
    def __init__(self, x, y, art='Pixel_Art/red.png', pixel_size=100):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(art), (pixel_size, pixel_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Player(Entity):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, pixel_size=pixel_size)
        # Create a transparent surface
        self.image = pygame.Surface((pixel_size, pixel_size), pygame.SRCALPHA)
        # Draw a red circle on the surface
        pygame.draw.circle(self.image, (255, 0, 0), (pixel_size // 2, pixel_size // 2), pixel_size // 2)
        # Set the rect attribute for the sprite
        self.rect = self.image.get_rect(topleft=(x, y))

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

class Outpost(Entity):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/outpost.png', pixel_size=pixel_size)

class Tree(Entity):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/tree_1.png', pixel_size=pixel_size)

class Rock(Entity):
    def __init__(self, x, y, art, pixel_size=100):
        super().__init__(x, y, art, pixel_size)

class MossyRock(Rock):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/rock_moss.png', pixel_size=pixel_size)

class SnowyRock(Rock):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/rock_snow.png', pixel_size=pixel_size)

class Fish(Entity):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/fish.png', pixel_size=pixel_size)
