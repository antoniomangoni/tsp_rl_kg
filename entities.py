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
        self.down_image = pygame.transform.scale(pygame.image.load('Pixel_Art/player-down.png'), (pixel_size, pixel_size))
        self.up_image = pygame.transform.scale(pygame.image.load('Pixel_Art/player-up.png'), (pixel_size, pixel_size))
        self.left_image = pygame.transform.scale(pygame.image.load('Pixel_Art/player-left.png'), (pixel_size, pixel_size))
        self.right_image = pygame.transform.scale(pygame.image.load('Pixel_Art/player-right.png'), (pixel_size, pixel_size))
        self.sleep_image = pygame.transform.scale(pygame.image.load('Pixel_Art/player-sleep.png'), (pixel_size, pixel_size))
        super().__init__(x, y, art= 'Pixel_Art/player.png', pixel_size=pixel_size)

    # Can only move in 4 directions
    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy
        
        if dx == 1:
            self.image = self.right_image
            return
        elif dx == -1:
            self.image = self.left_image
            return
        elif dy == 1:
            self.image = self.down_image
            return
        elif dy == -1:
            self.image = self.up_image
            return

class Outpost(Entity):
    def __init__(self, x, y, pixel_size=100):
        super().__init__(x, y, art='Pixel_Art/outpost_2.png', pixel_size=pixel_size)

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
