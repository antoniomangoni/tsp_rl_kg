import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, image, pixel_size=100):
        super().__init__()
        # just a red rectangle for now
        self.image = pygame.transform.scale(image, (pixel_size, pixel_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Tree(pygame.sprite.Sprite):
    def __init__(self, x, y, image, pixel_size=100):
        super().__init__()
        self.image = pygame.transform.scale(image, (pixel_size, pixel_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Rock(pygame.sprite.Sprite):
    def __init__(self, x, y, image, pixel_size=100):
        super().__init__()
        self.image = pygame.transform.scale(image, (pixel_size, pixel_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
