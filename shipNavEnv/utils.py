#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:26:47 2020

@author: gfo
"""
import numpy as np
import math
from gym.envs.classic_control.rendering import Color
from pyglet.gl import glColor4f

# COLORS
clist = []
clist.append((0,0,255))
clist.append((0,255,0))
clist.append((255,0,0))
clist.append((0,255,255))
clist.append((255,0,255))
clist.append((255,255,0))

def getColor(idx:int=0,alpha:float=1.0,clist:list=clist)->tuple:
    white = np.array([255,255,255])
    alpha = np.clip(alpha,0.6,1.0)
    color = (alpha * np.array(clist[idx%len(clist)]) + (1-alpha)*white)/255
    return tuple(color)

def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255

def draw_random_in_list(arr):
    index = np.random.randint(0, len(arr))
    obj = arr[index]
    del arr[index]
    return obj

def int_round_iter(iterable):
    return map(lambda x: int(round(x)), iterable)

def calc_angle_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    return math.atan2(dy, dx)

def get_path_dist(path):
    dist = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dist += np.linalg.norm((x2 - x1, y2 - y1))
    return dist

class DynamicColor(Color):
    def __init__(self, fn):
        self.fn = fn

    def enable(self):
        pass

    def disable(self): #FIXME Kinda hacky but render call enable in reverse and disable in correct order. Since the color is the first attr, using enable will override our dynamic color
        #print("In dynamic")
        #print(self.fn)
        #print(self.fn())
        glColor4f(*self.fn(), 1)
