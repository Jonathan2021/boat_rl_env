#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:26:47 2020

@author: gfo
"""
import numpy as np

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
