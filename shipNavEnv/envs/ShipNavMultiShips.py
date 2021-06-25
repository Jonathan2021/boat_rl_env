#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:46:23 2020

@author: gfo
"""

from shipNavEnv.Worlds import ShipsOnlyWorld, ShipsOnlyWorldLidar
from shipNavEnv.envs import ShipNavRocks, ShipNavRocksLidar
import numpy as np
from gym import spaces

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
    SINGLE_OBSTACLE_LENGTH = 6

    possible_kwargs = ShipNavRocks.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 0, 'scale':ShipsOnlyWorld.SCALE, 'waypoints':False})
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': possible_kwargs['fps']
    }


    def _build_world(self):
        return ShipsOnlyWorld(self.n_ships, self.scale, {'obs_radius': self.obs_radius})

    def _get_ships_obstacles(self):
        obstacles = self.world.get_obstacles(rocks=False)
        obstacles.sort(key=lambda x: (0 if x.seen else 1, abs(x.bearing_from_ship) + abs(x.bearing_to_ship) + x.distance_to_ship))
        for obs in obstacles[self.n_obstacles_obs:]:
            obs.seen = False

        self.obstacles = obstacles[:self.n_obstacles_obs]
        return self.obstacles

    def _get_obstacles(self):
        return self._get_ships_obstacles()

    def _get_obstacle_state(self):
        ship = self.world.ship
        state = ShipNavRocks._get_obstacle_state(self)
        for i in range(self.n_obstacles_obs):
            if i < len(self.obstacles) and self.obstacles[i].seen:
                state.append(self.obstacles[i].bearing_to_ship / np.pi)
                v_x, v_y = ship.body.GetLocalVector(self.obstacles[i].body.linearVelocity)
                state.append(v_x / ship.Vmax) # FIXME won't be exactly between -1 and 1 if Vmax of other ships is diff
                state.append(v_y / ship.Vmax)
                state.append(self.obstacles[i].body.angularVelocity / ship.Rmax) # same problem here
            else:
                state.append(np.random.uniform(-1,1))
                state.append(np.random.uniform(-1,1))
                state.append(np.random.uniform(-1,1))
                state.append(np.random.uniform(-1,1))
        return state
        

class ShipNavMultiShipsLidar(ShipNavRocksLidar):
    possible_kwargs = ShipNavRocksLidar.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 0, 'scale':ShipsOnlyWorld.SCALE, 'waypoints':False})

    def _build_world(self):
        return ShipsOnlyWorldLidar(self.n_ships, self.n_lidars, self.scale, {'obs_radius': self.obs_radius}, waypoint_support=self.waypoints)

class ShipNavMultiShipsLidarRadar(ShipNavMultiShipsRadius):
    possible_kwargs = ShipNavMultiShipsLidar.possible_kwargs.copy()
    possible_kwargs.update(ShipNavMultiShipsRadius.possible_kwargs)

    def _get_obs_space(self):
        return ShipNavRocksLidar._get_obs_space(self)

    def _build_world(self):
        return ShipNavMultiShipsLidar._build_world(self)

    def _get_state(self):
        return np.concatenate((ShipNavMultiShipsRadius._get_state(self), ShipNavRocksLidar._get_lidar_state(self)), dtype=np.float32)
