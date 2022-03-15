# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:35:52 2021

@author: Mika Sipilä
"""
import math
import numpy as np
import sympy
from random import randint
from collections import deque
from helpers import *

import pygame
from pygame.locals import *
import copy
import os

from DuelingDDQN import Agent

import cv2

FPS = 1
BLOCK_SIZE = 20
MAP_SIZE = 20
MAP_WIDTH = (MAP_SIZE + 2)*BLOCK_SIZE
IMG_WIDTH = 22
IMG_SIZE = (15, 15)
MAX_MOVES = 1000

class Block():
    
    def __init__(self, x, y, clean_counter = 0):
        self.x = x
        self.y = y
        self.clean_counter = clean_counter
        self.counter = 0
    
    def update_clean_counter(self):
        if self.clean_counter > 0:
            self.clean_counter -= 1/400
        self.counter += 1
        
class Area():
    
    def __init__(self, obstacle_file = None):
        self.obstacles = []
        if obstacle_file == None:
            self.generate_obstacles()
        else:
            self.obstacles_from_file(obstacle_file)
        self.dirt = []
        self.generate_dirt()
        self.map = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        self.blocked_squares = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        self.robots = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        self.unknown_squares = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        self.empty_squares = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        
    def generate_obstacles(self, amount = 0):
        amount = np.random.randint(1,3)/10.0
        for i in range(0, MAP_SIZE + 2):
            self.obstacles.append((0,i))
            self.obstacles.append((i,0))
            self.obstacles.append((MAP_SIZE + 1,i))
            self.obstacles.append((i, MAP_SIZE + 1))
        obstacle_count = round(MAP_SIZE*MAP_SIZE*amount)
        for i in range(0,obstacle_count):
            new_obstacle = tuple(np.random.randint(1, MAP_SIZE, size=2))
            while new_obstacle in self.obstacles:
                new_obstacle = tuple(np.random.randint(1, MAP_SIZE, size=2))
            self.obstacles.append((new_obstacle[0],new_obstacle[1]))
        
    def obstacles_from_file(self, file_name):
        obstacle_map = np.loadtxt(file_name, dtype='i', delimiter=',')
        #print(obstacle_map)
        num_rows, num_cols = obstacle_map.shape
        for i in range(num_rows):
            for j in range(num_cols):
                if obstacle_map[i,j] == 1:
                    self.obstacles.append((i,j))
        
    def generate_dirt(self, amount = 0.1):
        dirt_count = round(MAP_SIZE*MAP_SIZE*amount)
        for i in range(0,dirt_count):
            new_dirt = tuple(np.random.randint(1, MAP_SIZE, size=2))
            while new_dirt in self.obstacles or new_dirt in self.dirt:
                new_dirt = tuple(np.random.randint(1, MAP_SIZE, size=2))
            self.dirt.append(new_dirt)
            
            
class RobotPopulation():
    
    def __init__(self, count, area):
        self.cleaning_robots = []
        self.squares_visited = []
        self.blocked_squares = []
        self.unknown_squares = []
        self.initial_pos = (1,1)  
        self.area = area
        self.cleaned = 0
        for i in range(0,count):
            new_pos = tuple(np.random.randint(1, MAP_SIZE, size=2))
            while new_pos in self.area.obstacles or new_pos in self.area.dirt:
                new_pos = tuple(np.random.randint(0, MAP_SIZE, size=2))
            self.cleaning_robots.append(CleaningRobot(new_pos, self, self.area))
            self.squares_visited.append(Block(new_pos[0],new_pos[1]))
            self.area.map[new_pos[0], new_pos[1]] = 0.4
            self.area.empty_squares[new_pos[1], new_pos[0]] = 1
        self.update_unknown_squares()            
    
    def perform_actions(self):
        for robot in self.cleaning_robots:
            robot.perform_action()
        self.update_unknown_squares()
    
    def update_unknown_squares(self):
        self.area.unknown_squares = np.zeros(shape=(MAP_SIZE + 2, MAP_SIZE + 2))
        for robot in self.cleaning_robots:
            for i in (-1,0,1):
                for j in (-1,0,1):
                    unknown_pos = (robot.pos[0] + i, robot.pos[1] + j)
                    if unknown_pos not in map(lambda o: (o.x, o.y), self.squares_visited) and unknown_pos not in self.area.obstacles:
                        if unknown_pos not in self.unknown_squares:
                            self.unknown_squares.append(Block(unknown_pos[0], unknown_pos[1]))
                        self.area.map[unknown_pos[0], unknown_pos[1]] = 0.6
                        self.area.unknown_squares[unknown_pos[1], unknown_pos[0]] = 1
    
    def update_cleanable_squares(self):
        for block in self.squares_visited:
            block.update_clean_counter()
            if block.clean_counter <= 0:
                self.unknown_squares.append(Block(block.x,block.y))
        self.squares_visited = [block for block in self.squares_visited if (block.clean_counter > 0)]

class CleaningRobot():
    
    def __init__(self, pos, population, area):
        self.pos = pos
        self.state_dict = ["move", "change_direction", "clean"]
        self.state = "move"
        self.direction = (1,0)
        self.population = population
        self.area = area
        self.moves = 0
        self.cleaned_dirt = 0
        self.original_img = load_images()['robot']
        self.img = self.original_img
        self.next_direction = None
        self.reward_for_action = 0
        self.prev_positions = []
        self.prev_frames = []
        self.previous_actions = []
        self.obsercation = None
        self.new_observation = None
        self.no_new_things_found_counter = 0
        
    def change_direction(self):
        if self.direction[0] == 0:
            if self.direction[1] == 1:
                self.direction = (-1, 0)
                self.img = pygame.transform.flip(self.original_img, True, False)
            else:
                self.direction = (1, 0)
                self.img = self.original_img
        else:
            if self.direction[0] == 1:
                self.direction = (0, 1)
                self.img = rot_center(self.original_img, 270)
            else:
                self.direction = (0, -1)
                self.img = rot_center(self.original_img, 90)
        self.state = "move"
    
    def change_direction_to(self, direction = None):
        if direction == None:
            self.change_direction
        elif not direction in [(0,1), (0,-1), (1,0), (-1,0)]:
            return
        while self.direction != direction:
            self.change_direction()

    def move(self):
        next_pos = (self.pos[0] + self.direction[0], self.pos[1] + self.direction[1])
        #power = 10
        #i = -1
        #while len(self.prev_positions) >= abs(i) and next_pos != self.prev_positions[i] and power > 0:
        #    power -= 1
        #    i -= 1
        #if power > 0:
        #    self.reward_for_action -= 1.5**(power/2)
        #if len(self.population.unknown_squares) > 0:
        #    if self.nearest_unknown(next_pos)[1] < self.nearest_unknown(self.pos)[1]:
        #        self.reward_for_action += 5
        if next_pos in map(lambda o: (o.x, o.y), self.population.squares_visited):
            blocks = [block for block in self.population.squares_visited if (block.x == next_pos[0], block.y == next_pos[1])]
            self.reward_for_action -= blocks[0].clean_counter
            blocks[0].clean_counter = 1
            blocks[0].counter = 0
            self.no_new_things_found_counter += 1

        if next_pos in self.area.obstacles or next_pos in map(lambda o: o.pos, self.population.cleaning_robots):
            self.state = "change_direction"
            if next_pos not in self.population.blocked_squares:
                self.no_new_things_found_counter = 0
                self.reward_for_action += 2
                self.population.blocked_squares.append(next_pos)
                self.area.map[next_pos[0], next_pos[1]] = 0.8
                self.area.blocked_squares[next_pos[1], next_pos[0]] = 1
                self.prev_positions.append(self.pos)
            else:
                self.reward_for_action -= 1
                self.no_new_things_found_counter += 1
            next_pos = self.pos
        elif next_pos in self.area.dirt:
            self.state = "clean"
            if next_pos not in map(lambda o: (o.x, o.y), self.population.squares_visited):
                self.population.squares_visited.append(Block(next_pos[0], next_pos[1], 1))
                self.no_new_things_found_counter = 0
            self.area.empty_squares[self.pos[1], self.pos[0]] = 1
            self.area.robots[next_pos[1], next_pos[0]] = 1
            self.area.robots[self.pos[1], self.pos[0]] = 0
            self.area.map[self.pos[0], self.pos[1]] = 0.2
            self.pos = next_pos
            self.prev_positions.append(next_pos)
            self.area.map[next_pos[0], next_pos[1]] = 0.4
            self.area.empty_squares[next_pos[1], next_pos[0]] = 1
        else:
            if next_pos not in map(lambda o: (o.x, o.y), self.population.squares_visited):
                self.population.squares_visited.append(Block(next_pos[0], next_pos[1], 1))
                self.no_new_things_found_counter = 0
            self.area.empty_squares[self.pos[1], self.pos[0]] = 1
            self.area.robots[next_pos[1], next_pos[0]] = 1
            self.area.robots[self.pos[1], self.pos[0]] = 0
            self.pos = next_pos
            self.prev_positions.append(next_pos)
        if next_pos in map(lambda o: (o.x, o.y), self.population.unknown_squares):
            self.population.unknown_squares = [block for block in self.population.unknown_squares if not (block.x == next_pos[0] and block.y == next_pos[1])]
            #self.area.unknown_squares[next_pos[1], next_pos[0]] = 0
            if next_pos not in self.area.obstacles:
                self.reward_for_action += 1

    def clean(self):
        if self.pos in self.area.dirt:
            self.area.dirt.remove(self.pos)
            self.cleaned_dirt += 1
            self.population.cleaned += 1
        
    def perform_action(self, direction):
        self.reward_for_action = 0
        self.clean()
        self.change_direction_to(direction)
        self.move()
        #self.previous_frames = np.stack((self.nearest_blocks(), self.previous_frames[0,:,:], self.previous_frames[1,:,:], self.previous_frames[2,:,:]))
        return self.reward_for_action
        
        
    def nearest_unknown(self, pos = None):
        if pos == None:
            pos = self.pos
        distance_map = np.full((MAP_SIZE + 2,MAP_SIZE + 2), 1000, dtype=int)
        distance_map[pos[0], pos[1]] = 0
        current_dist = 0
        nearest_block = None
        done = False
        while not done:
            if len(self.population.unknown_squares) == 0:
                print(self.population.unknown_squares)
                return((self.pos[0],self.pos[1]), 0, (1,0))
            row_indices, col_indices = np.where(distance_map == current_dist)
            for k in range(len(row_indices)):
                i = row_indices[k]
                j = col_indices[k]
                real_pos = (i,j)
                if real_pos in map(lambda o: (o.x, o.y), self.population.unknown_squares):
                    done = True
                    nearest_block = real_pos
                    break
                if not done:
                    if (real_pos[0] + 1, real_pos[1]) not in self.area.obstacles:
                        if real_pos[0] + 1 in range(0,MAP_SIZE + 2) and real_pos[1] in range(0, MAP_SIZE + 2):
                            if distance_map[i + 1,j] > current_dist + 1:
                                distance_map[i + 1,j] = current_dist + 1
                    if (real_pos[0] - 1, real_pos[1]) not in self.area.obstacles:
                        if real_pos[0] - 1 in range(0,MAP_SIZE + 2) and real_pos[1] in range(0, MAP_SIZE + 2):
                            if distance_map[i - 1,j] > current_dist + 1:
                                distance_map[i - 1,j] = current_dist + 1
                    if (real_pos[0], real_pos[1] + 1) not in self.area.obstacles:
                        if real_pos[0] in range(0,MAP_SIZE + 2) and real_pos[1] + 1 in range(0, MAP_SIZE + 2):
                            if distance_map[i,j + 1] > current_dist + 1:
                                distance_map[i,j + 1] = current_dist + 1
                    if (real_pos[0], real_pos[1] - 1) not in self.area.obstacles:
                        if real_pos[0] in range(0,MAP_SIZE + 2) and real_pos[1] - 1 in range(0, MAP_SIZE + 2):
                            if distance_map[i,j - 1] > current_dist + 1:
                                distance_map[i,j - 1] = current_dist + 1
            if not done:
                current_dist += 1
        neighbours = [(nearest_block[0], nearest_block[1])]
        for dist in range(1,current_dist):
            for curr_block in neighbours:
                for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
                    if distance_map[curr_block[0] + direction[0], curr_block[1] + direction[1]] == current_dist - dist:
                        neighbours.append((curr_block[0] + direction[0], curr_block[1] + direction[1]))
                neighbours.remove(curr_block)
        if len(neighbours) <= 0:
            return((self.pos[0],self.pos[1]), 0, (1,0))
            
        return((nearest_block, current_dist, (neighbours[0][0] - pos[0], neighbours[0][1] - pos[1])))
    
    def direction_to(self, point):
        dir1 = 0
        dir2 = 0
        if ( point[0] - self.pos[0] >= 0): dir1 = 1
        else: dir1 = -1 
        if (point[1] - self.pos[1] >= 0): dir2 = 1
        else: dir2 = -1
        if (abs(point[0] - self.pos[0]) >= abs(point[1] - self.pos[1])):
            return (dir1,0)
        else:
            return (0,dir2)
        
    def nearest_blocks(self):
        blocks = np.ones(shape=(9,9))
        for i in range(9):
            for j in range(9):
                block = (self.pos[0] - 4 + i, self.pos[1] - 4 + j)
                visited_blocks = [vis_block for vis_block in self.population.squares_visited if (vis_block.x == block[0] and vis_block.y == block[1])]
                if block in self.population.blocked_squares:
                    blocks[j,i] = -1
                elif block in map(lambda o: o.pos, self.population.cleaning_robots):
                    blocks[j,i] = 0
                elif len(visited_blocks) > 0:
                    blocks[j,i] = 0.5 - visited_blocks[0].clean_counter/2
        return blocks
    
    def nearest_blocks_one_hot(self):
        blocks1 = np.zeros(shape=(5,5))
        blocks2 = np.zeros(shape=(5,5))
        blocks3 = np.zeros(shape=(5,5))
        blocks4 = np.zeros(shape=(7,5))
        for i in range(3):
            for j in range(3):
                block_i = self.pos[0] + 2 - i
                block_j = self.pos[1] + 2 - j
                if block_i >= 0 and block_i < MAP_SIZE + 2 and block_j >= 0 and block_j < MAP_SIZE + 2:
                    blocks1[i,j] = self.area.blocked_squares[block_i, block_j]
                    blocks2[i,j] = self.area.empty_squares[block_i, block_j]
                    blocks3[i,j] = self.area.unknown_squares[block_i, block_j]
                    blocks4[i,j] = self.area.robots[block_i, block_j]
        return np.stack((blocks1, blocks2, blocks3, blocks4))
    
    def get_dist_to(self, block_type, direction, max_dist = 1):
        for dist in range(1, max_dist + 1):
            if block_type == "obstacle":
                if (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in self.area.obstacles:
                    return (1 - (dist-1)/max_dist)
                elif (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in map(lambda o: (o.x, o.y), self.population.unknown_squares):
                    return 0
            if block_type == "unknown":
                if (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in map(lambda o: (o.x, o.y), self.population.unknown_squares):
                    return (1 - (dist-1)/max_dist)
            elif (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in self.area.obstacles:
                return 0
            if block_type == "known":
                if (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in map(lambda o: (o.x, o.y), self.population.squares_visited):
                    return (1 - (dist-1)/max_dist)
            elif (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in self.area.obstacles:
                return 0
            elif (round(self.pos[0] + dist*direction[0]), round(self.pos[1] + dist*direction[1])) in map(lambda o: (o.x, o.y), self.population.unknown_squares):
                return 0
        return(0)
    
    def previous_position_penalty(self, direction):
        next_pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
        if next_pos in self.area.obstacles:
            next_pos = self.pos
        penalty = 20
        i = -1
        while len(self.prev_positions) >= abs(i) and next_pos != self.prev_positions[i] and penalty > 0:
            penalty -= 1
            i -= 1
        if len(self.prev_positions) >= abs(i):
            return(penalty/20)
        else: 
            return 0
    
    def get_clean_counter(self, direction):
        blocks = [block for block in self.population.squares_visited if (block.x == self.pos[0] + direction[0] and block.y == self.pos[1] + direction[1])]
        if len(blocks) > 0:
            return(blocks[0].clean_counter)
        else:
            return 0
    
    def get_clean_counter_for_distance(self, direction, dist = 20):
        clean_counters = []
        i = 1
        while (self.pos[0] + direction[0]*i, self.pos[1] + direction[1]*i) not in self.area.obstacles:
            clean_counters.append(self.get_clean_counter((direction[0]*i, direction[1]*i)))
            i += 1
        if len(clean_counters) <= 0:
            return 0
        return(min(clean_counters))
        
    def distances_to_observation(self):
        d1 = self.get_dist_to("obstacle", (-1,0))
        d2 = self.get_dist_to("obstacle", (-1/np.sqrt(2),1/np.sqrt(2)))
        d3 = self.get_dist_to("obstacle", (0,1))
        d4 = self.get_dist_to("obstacle", (1/np.sqrt(2),1/np.sqrt(2)))
        d5 = self.get_dist_to("obstacle", (1,0))
        d6 = self.get_dist_to("obstacle", (1/np.sqrt(2),-1/np.sqrt(2)))
        d7 = self.get_dist_to("obstacle", (0,-1))
        d8 = self.get_dist_to("obstacle", (-1/np.sqrt(2),-1/np.sqrt(2)))
        d1_u = self.get_dist_to("unknown", (-1,0))
        d2_u = self.get_dist_to("unknown",(-1/np.sqrt(2),1/np.sqrt(2)))
        d3_u = self.get_dist_to("unknown", (0,1))
        d4_u = self.get_dist_to("unknown",(1/np.sqrt(2),1/np.sqrt(2)))
        d5_u = self.get_dist_to("unknown", (1,0))
        d6_u = self.get_dist_to("unknown",(1/np.sqrt(2),-1/np.sqrt(2)))
        d7_u = self.get_dist_to("unknown", (0,-1))
        d8_u = self.get_dist_to("unknown",(-1/np.sqrt(2),-1/np.sqrt(2)))
        d1_k = self.get_dist_to("known", (-1,0))
        d2_k = self.get_dist_to("known", (-1/np.sqrt(2),1/np.sqrt(2)))
        d3_k = self.get_dist_to("known", (0,1))
        d4_k = self.get_dist_to("known", (1/np.sqrt(2),1/np.sqrt(2)))
        d5_k = self.get_dist_to("known", (1,0))
        d6_k = self.get_dist_to("known", (1/np.sqrt(2),-1/np.sqrt(2)))
        d7_k = self.get_dist_to("known", (0,-1))
        d8_k = self.get_dist_to("known", (-1/np.sqrt(2),-1/np.sqrt(2)))
        #d1_10 = self.get_dist_to("obstacle", (-1,0), 10)
        #d2_10 = self.get_dist_to("obstacle", (-1,1), 10)
        #d3_10 = self.get_dist_to("obstacle", (0,1), 10)
        #d4_10 = self.get_dist_to("obstacle", (1,1), 10)
        #d5_10 = self.get_dist_to("obstacle", (1,0), 10)
        #d6_10 = self.get_dist_to("obstacle", (1,-1), 10)
        #d7_10 = self.get_dist_to("obstacle", (0,-1), 10)
        #d8_10 = self.get_dist_to("obstacle", (-1,-1), 10)
        d1_u_10 = self.get_dist_to("unknown", (-1,0), 20)
        d2_u_10 = self.get_dist_to("unknown",(-1/np.sqrt(2),1/np.sqrt(2)), 20)
        d3_u_10 = self.get_dist_to("unknown", (0,1), 20)
        d4_u_10 = self.get_dist_to("unknown",(1/np.sqrt(2),1/np.sqrt(2)), 20)
        d5_u_10 = self.get_dist_to("unknown", (1,0), 20)
        d6_u_10 = self.get_dist_to("unknown",(1/np.sqrt(2),-1/np.sqrt(2)), 20)
        d7_u_10 = self.get_dist_to("unknown", (0,-1), 20)
        d8_u_10 = self.get_dist_to("unknown",(-1/np.sqrt(2),-1/np.sqrt(2)), 20)
        #clean_counter1 = self.get_clean_counter((-1,0))
        #clean_counter2 = self.get_clean_counter((-1,1))
        #clean_counter3 = self.get_clean_counter((0,1))
        #clean_counter4 = self.get_clean_counter((1,1))
        #clean_counter5 = self.get_clean_counter((1,0))
        #clean_counter6 = self.get_clean_counter((1,-1))
        #clean_counter7 = self.get_clean_counter((0,-1))
        #clean_counter8 = self.get_clean_counter((-1,-1))
        #clean_counter1_20 = self.get_clean_counter_for_distance((-1,0))
        #clean_counter2_20 = self.get_clean_counter_for_distance((-1,1))
        #clean_counter3_20 = self.get_clean_counter_for_distance((0,1))
        #clean_counter4_20 = self.get_clean_counter_for_distance((1,1))
        #clean_counter5_20 = self.get_clean_counter_for_distance((1,0))
        #clean_counter6_20 = self.get_clean_counter_for_distance((1,-1))
        #clean_counter7_20 = self.get_clean_counter_for_distance((0,-1))
        #clean_counter8_20 = self.get_clean_counter_for_distance((-1,-1))
        penalty1 = self.previous_position_penalty((0,-1))
        penalty2 = self.previous_position_penalty((1,0))
        penalty3 = self.previous_position_penalty((0,1))
        penalty4 = self.previous_position_penalty((-1,0))
        return(np.array((d1,d2,d3,d4,d5,d6,d7,d8,d1_u,d2_u,d3_u,d4_u,d5_u,d6_u,d7_u,d8_u,d1_k,d2_k,d3_k,d4_k,d5_k,d6_k,d7_k,d8_k,d1_u_10,d2_u_10,d3_u_10,d4_u_10,d5_u_10,d6_u_10,d7_u_10,d8_u_10,penalty1, penalty2, penalty3, penalty4)))
        #return(np.append(arr, self.nearest_blocks().flatten().reshape(-1)))
        #clean_counter1,clean_counter2,clean_counter3,clean_counter4,clean_counter5,clean_counter6,clean_counter7,clean_counter8,clean_counter1_20,clean_counter2_20,clean_counter3_20,clean_counter4_20,clean_counter5_20,clean_counter6_20,clean_counter7_20,clean_counter8_20,
def blit_map(display_surface, population, area, background, visited_img, dirt_img, obstacle_img, score_font, agent, episode, score):
    
    display_surface.blit(background, (0, 0))
    for square in map(lambda o: (o.x, o.y), population.squares_visited):
        display_surface.blit(visited_img, (square[0]*BLOCK_SIZE, square[1]*BLOCK_SIZE))
    for dirt in area.dirt:
        if dirt in map(lambda o: (o.x, o.y), population.squares_visited):
            display_surface.blit(dirt_img, (dirt[0]*BLOCK_SIZE, dirt[1]*BLOCK_SIZE))
    for robot in population.cleaning_robots:
        display_surface.blit(robot.img, (robot.pos[0]*BLOCK_SIZE, robot.pos[1]*BLOCK_SIZE))
    for obstacle in population.blocked_squares:
        #if obstacle in population.blocked_squares:
        display_surface.blit(obstacle_img, (obstacle[0]*BLOCK_SIZE, obstacle[1]*BLOCK_SIZE))
            
    score_surface = score_font.render("Cleaned: " + str(len(population.squares_visited)), True, (255, 0, 0))
    epsilon_surface = score_font.render('Epsilon: %.3f' % agent.epsilon, True, (255, 0, 0))
    episode_surface = score_font.render("Episode: " + str(episode), True, (255, 0, 0))
    score2_surface = score_font.render("Score: %.2f" % score, True, (255, 0, 0))

    display_surface.blit(episode_surface, (10, 10))
    display_surface.blit(epsilon_surface, (10, 30))
    display_surface.blit(score_surface, (10, 50))
    display_surface.blit(score2_surface, (10, 70))
            
def main():
    lr = 0.0005
    n_games = 500
    num_agents = 1
    scores = []
    eps_history = []
    agent = Agent(lr = lr, gamma=0.85, epsilon=1, 
                input_dims=[9,9,1],
                n_actions=4, mem_size=30000, batch_size=128,
                eps_end=0.001, replace = 512, epsilon_dec = 0.0005,
                q_eval_model_file='q_eval_dueling_ddqn2',
                q_target_model_file='q_next_dueling_ddqn2'
                )
    #agent2 = Agent(lr = lr, gamma=0.85, epsilon=1, 
    #            input_dims=[19,19,1],
    #            n_actions=4, mem_size=30000, batch_size=128,
    #            eps_end=0.001, replace = 512, epsilon_dec = 0.0005,
    #            q_eval_model_file='q_eval_dueling_ddqn2',
    #            q_target_model_file='q_next_dueling_ddqn2'
    #            )
    
    pygame.init()
    
    best_score = 100
    
    for i in range(n_games):
        
        display_surface = pygame.display.set_mode((MAP_WIDTH, MAP_WIDTH))
        pygame.display.set_caption('Cleaning robots')
    
        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 16, bold=True)  # default font
        frame_clock = 0
        robot_img = load_images()['robot']
        dirt_img = load_images()['dirt']
        obstacle_img = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), SRCALPHA)
        obstacle_img.convert()   # speeds up blitting
        obstacle_img.fill((0,0,0))
        background = pygame.Surface((MAP_WIDTH, MAP_WIDTH), SRCALPHA)
        background.convert()
        background.fill((190,190,190))
        visited_img = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), SRCALPHA)
        visited_img.convert()   # speeds up blitting
        visited_img.fill((220,220,220))
        #if i % 7 == 0:
        #    area = Area('house2.txt')
        #elif i % 8 == 0:
        #if i % 2 == 0:
        #    area = Area('house1.txt')
        #else:
        area = Area('house2.txt')
        #    area = Area()
        population = RobotPopulation(num_agents, area)
        
        blit_map(display_surface, population, area, background, visited_img, dirt_img, obstacle_img, score_font, agent, i, 0)
        
        observation = None
        score = 0
        move = 0
        done = False
        while not done:
            #clock.tick(FPS)
            
            if frame_clock == 0:
                #pygame.image.save(display_surface, "observation.png")
                #image = cv2.imread("observation.png", 0)
                #image = cv2.resize(image, IMG_SIZE)
                #frame = population.cleaning_robots[0].nearest_blocks()
                #population.cleaning_robots[0].previous_frames = np.stack((frame,frame,frame,frame))
                #observation = population.cleaning_robots[0].distances_to_observation().reshape(-1,1)
                #observation = np.asarray(image).reshape(IMG_WIDTH,IMG_WIDTH,1)/255
                for robot in population.cleaning_robots:
                    robot.observation = robot.nearest_blocks().reshape(9,9,1)
            
            rewards = []
            for j in range(len(population.cleaning_robots)):
                if i*500 + move <= 0:
                    direction = population.cleaning_robots[i].nearest_unknown()[2]
                    if direction == (0,1):
                        action = 1
                    elif direction == (-1,0):
                        action = 2
                    elif direction == (0,-1):
                        action = 3
                    else:
                        action = 0
                else:
                    #if j == 0:
                    action = agent.choose_action(population.cleaning_robots[j].observation)
                    #else:
                    #    action = agent2.choose_action(population.cleaning_robots[j].observation)
                if action == 1:
                    direction = (0,1) #down
                elif action == 2:
                    direction = (-1,0) #left
                elif action == 3:
                    direction = (0,-1) #up
                else:
                    direction = (1,0) #right
                rewards.append(population.cleaning_robots[j].perform_action(direction))
            population.update_unknown_squares()
            
            population.update_cleanable_squares()

            total_reward = 0
            for reward in rewards:
                total_reward += reward
            score += total_reward

            blit_map(display_surface, population, area, background, visited_img, dirt_img, obstacle_img, score_font, agent, i, score)
            
            pygame.event.pump()
            pygame.display.flip()
    
            frame_clock += 1
            #pygame.image.save(display_surface, "observation.png")
            #image = cv2.imread("observation.png", 0)
            #image = cv2.resize(image, IMG_SIZE)
            #observation_ = np.asarray(image).reshape(IMG_WIDTH,IMG_WIDTH,1)/255
            #population.cleaning_robots[0].nearest_unknown()
            #observation_ = population.cleaning_robots[0].distances_to_observation().reshape(-1,1)
            for robot in population.cleaning_robots:
                    robot.new_observation = robot.nearest_blocks().reshape(9,9,1)
            #print(observation_)
            #print(population.cleaning_robots[0].previous_frames.astype('float32'))
            if move >= MAX_MOVES or population.cleaning_robots[0].no_new_things_found_counter >= 20:
                done = True
                population.cleaning_robots[0].reward_for_action -= 20

            for j in range(len(population.cleaning_robots)):
                robot = population.cleaning_robots[j]
                if j == 0:
                    agent.store_transition(robot.observation, action, robot.reward_for_action + total_reward/2, robot.new_observation, done)
                    agent.learn()
                    robot.observation = robot.new_observation
                else:
                    agent2.store_transition(robot.observation, action, robot.reward_for_action + total_reward/2, robot.new_observation, done)
                    agent2.learn()
                    robot.observation = robot.new_observation
            
            #eps_history.append(agent.epsilon)
            move += 1
        scores.append(score)
        avg_score = np.mean(scores[-10:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode: ', i, 'score %.3f' % score,
                    'average_score %.2f' % avg_score,
                    'epsilon: %.3f' % agent.epsilon)

    print(i)
    print(population.cleaned)                        
                                     
if __name__ == '__main__':
    main()
        
        
        
        
        