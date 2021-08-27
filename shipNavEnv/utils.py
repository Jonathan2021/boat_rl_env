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
from gym.envs.classic_control import rendering
import pyglet

# COLORS
clist = []
clist.append((0,0,255))
clist.append((0,255,0))
clist.append((255,0,0))
clist.append((0,255,255))
clist.append((255,0,255))
clist.append((255,255,0))

def getColor(idx:int=0,alpha:float=1.0,clist:list=clist)->tuple:
    """ Get a color in a list and apply an alpha """
    white = np.array([255,255,255])
    alpha = np.clip(alpha,0.6,1.0)
    color = (alpha * np.array(clist[idx%len(clist)]) + (1-alpha)*white)/255
    return tuple(color)

def rgb(r, g, b):
    """ Get scaled r g b in [-1, 1] """
    return float(r) / 255, float(g) / 255, float(b) / 255

def draw_random_in_list(arr):
    """ Pop a random element from a list """
    index = np.random.randint(0, len(arr))
    obj = arr[index]
    del arr[index]
    return obj

def int_round_iter(iterable):
    """ return a iterable of ints rounded to the closest point from another iterable """
    return map(lambda x: int(round(x)), iterable)

def calc_angle_two_points(p1, p2):
    """ Get the angle of the vector formed by 2 points """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    return math.atan2(dy, dx)

def get_path_dist(path):
    """ Get the accumulated distance of a path from a list of checkpoints """
    dist = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dist += np.linalg.norm((x2 - x1, y2 - y1))
    return dist

def make_half_circle(radius, init_angle = 0, res=15, filled=True):
    """ Make a half circle geom (for render) """
    points = []
    for i in range(res):
        ang = init_angle + (math.pi * i / (res-1))
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, False)
