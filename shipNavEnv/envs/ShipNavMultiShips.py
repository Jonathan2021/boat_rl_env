#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:46:23 2020

@author: gfo
"""

from shipNavEnv.Worlds import ShipsOnlyWorld
from shipNavEnv.envs import ShipNavRocks

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
class ShipNavMultiShips(ShipNavRocks):

    FPS = 30          # simulation framerate
    MAX_TIME = 200    # max steps for a simulation
    MAX_STEPS = MAX_TIME * FPS  # max steps for a simulation
    SHIP_STATE_LENGTH = 6
    WORLD_STATE_LENGTH = 2

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def _read_kwargs(self, **kwargs):
        n_ships_default = 0
        self.n_ships = kwargs.get('n_ships', n_ships_default)

        n_ships_obs_default = self.n_ships
        self.n_ships_obs = kwargs.get('n_ships_obs', n_ships_obs_default)
        self.n_obstacles_obs = self.n_ships_obs

        obs_radius_default = 200
        self.obs_radius = kwargs.get('obs_radius', obs_radius_default)
        
        fps_default = self.FPS
        self.fps = kwargs.get('fps', fps_default)

        display_traj_default = False
        self.display_traj = kwargs.get('display_traj', display_traj_default)

        display_traj_T_default = 0.1
        self.display_traj_T = kwargs.get('display_traj_T', display_traj_T_default)


    def _build_world(self):
        return ShipsOnlyWorld(self.n_ships, {'obs_radius': self.obs_radius})
