#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:46:23 2020

@author: gfo
"""

from shipNavEnv.Worlds import ShipsOnlyWorld, ShipsOnlyWorldLidar
from shipNavEnv.envs import ShipNavRocks, ShipNavRocksLidar

"""

STATE VARIABLES
The state consists of the following variables:
    - ship's velocity on it's sway axis
    - ship's velocity on its surge axis
    - angular velocity
    - thruster angle normalized
    - distance to target (ship's frame) normalized
    - target bearing normalized
    - distance to rock n°1 normalized
    - bearing to rock n°1 normalized
    - ...
    - distance to rock n°k normalized
    - bearing to rock n°k normalized
    - ...
    - distance to last rock normalized
    - bearing to last rock normalized

all state variables are roughly in the range [-1, 1] (distances are normalized)
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - no action
"""


# gym env class
class ShipNavMultiShipsRadius(ShipNavRocks):

    possible_kwargs = ShipNavRocks.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 0})
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': possible_kwargs['fps']
    }


    def _build_world(self):
        return ShipsOnlyWorld(self.n_ships, {'obs_radius': self.obs_radius})

class ShipNavMultiShipsLidar(ShipNavRocksLidar):
    possible_kwargs = ShipNavRocksLidar.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 0})

    def _build_world(self):
        return ShipsOnlyWorldLidar(self.n_ships, self.n_lidars)
