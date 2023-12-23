import pygame

class Entity(pygame.sprite.Sprite):
    def __init__(self, x, y, art='Pixel_Art/red.png', pixel_size=100):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(art), (pixel_size, pixel_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Player(Entity):
    def __init__(self, x, y, art, pixel_size=100):
        super().__init__(x, y, art, pixel_size)

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

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
