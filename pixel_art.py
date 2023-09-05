# pixel_art.py
import pygame

class Tree(pygame.sprite.Sprite):
    def __init__(self, x, y, image, pixel_size=100):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.pixel_size = pixel_size
        self.rect.topleft = (x, y)

        
        self.image = pygame.image.load('C:/Users/anton/Dev/ABM/Pixel_Art/tree_1.png')
        self.image = pygame.transform.scale(self.tree_image, (self.pixel_size, self.pixel_size))
