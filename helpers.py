# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:09:46 2021

@author: Mika Sipilä
"""
import pygame
import numpy as np
import os

def dist(point1, point2):
    return( np.sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2))
    
def load_images():

    def load_image(img_file_name):

        file_name = os.path.join(os.path.dirname(__file__), 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'dirt': load_image('dirt.png'),
            'robot': load_image('robot.png')}
    
def rot_center(image, angle):
    """rotate a Surface, maintaining position."""
        
    loc = image.get_rect().center  
    rot_sprite = pygame.transform.rotate(image, angle)
    rot_sprite.get_rect().center = loc
    return rot_sprite