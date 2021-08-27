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
    - Radar (r,g,b) image of surrondings (if applicable)
    - Lidar fraction of the distance (if applicable)
    - Fraction of time left to episode end

all state variables are roughly standardized in the range [-1, 1] and centered at 0
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - no action
"""


# gym env class
class ShipNavMultiShipsRadius(ShipNavRocks):
    """
    Env with ship obstacles only and without lidars (only radar)
    """
    SINGLE_OBSTACLE_LENGTH = 6 # Length of entries for an obstacle (ship) in the old radar (Tabular).

    possible_kwargs = ShipNavRocks.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 50, 'scale':ShipsOnlyWorld.SCALE, 'waypoints':False})
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': possible_kwargs['fps']
    }

    def _build_world(self):
        """ Build a world full of ships in the given scale and with the given radar radius """
        return ShipsOnlyWorld(self.n_ships, self.scale, {'obs_radius': self.obs_radius})

    def _get_ships_obstacles(self):
        """ Get the sorted list of a predefined number of neighbouring ships. (for old radar version). """

        obstacles = self.world.get_obstacles(rocks=False) # Get the ships
        obstacles.sort(key=lambda x: (0 if x.seen else 1, abs(x.bearing_from_ship) + abs(x.bearing_to_ship) + x.distance_to_ship)) # Sort by seen, then on how much they are going one towards the other, then distance
        for obs in obstacles[self.n_obstacles_obs:]:
            obs.seen = False

        self.obstacles = obstacles[:self.n_obstacles_obs] # Only keep the predefined number of observable obstacles
        return self.obstacles

    def _get_obstacles(self):
        return self._get_ships_obstacles() # wrapper calls wanted logic

    def _get_obstacle_state(self):
        """ Build state (Old radar) from the list of obstacles """
        ship = self.world.ship
        state = ShipNavRocks._get_obstacle_state(self) # Get the states like rocks (x, y and bearing)
        for i in range(self.n_obstacles_obs): # For the predefined number of obstacles
            if i < len(self.obstacles) and self.obstacles[i].seen: # If we indeed have at least i + 1 obstacles and it was seen
                state.append(self.obstacles[i].bearing_to_ship / np.pi) # Add bearing
                v_x, v_y = ship.body.GetLocalVector(self.obstacles[i].body.linearVelocity)
                state.append(v_x / ship.Vmax) # Add speed in local X axis FIXME won't be exactly between -1 and 1 if Vmax of other ships is diff
                state.append(v_y / ship.Vmax) # speed in local Y
                state.append(self.obstacles[i].body.angularVelocity / ship.Rmax) # same problem here
            else: # Not seen or not enough obstacles in the radius etc.
                state.append(np.random.uniform(-1,1)) # Fill in with random values
                state.append(np.random.uniform(-1,1))
                state.append(np.random.uniform(-1,1))
                state.append(np.random.uniform(-1,1))
        return state
        

class ShipNavMultiShipsLidar(ShipNavRocksLidar):
    """
    Env with ship obstacles but Lidar instead of radar.
    """
    possible_kwargs = ShipNavRocksLidar.possible_kwargs.copy()
    possible_kwargs.update({'n_ships': 0, 'scale':ShipsOnlyWorld.SCALE, 'waypoints':False})

    def _build_world(self):
        """ Build radar world """
        return ShipsOnlyWorldLidar(self.n_ships, self.n_lidars, self.scale, {'obs_radius': self.obs_radius}, waypoint_support=self.waypoints)

class ShipNavMultiShipsLidarRadar(ShipNavMultiShipsRadius, ShipNavMultiShipsLidar):
    """ Env with ship obstacles but mixes radar and lidar """
    possible_kwargs = ShipNavMultiShipsLidar.possible_kwargs.copy()
    possible_kwargs.update(ShipNavMultiShipsRadius.possible_kwargs)


    # FIXME Not clear because inheriting from everywhere but the following functions actually work. The software design is bogus and should be changed
    def _get_obs_space(self):
        return ShipNavRocksLidar._get_obs_space(self) # Get observation space of including lidar (will use radar as well with this call)

    def _build_world(self):
        return ShipNavMultiShipsLidar._build_world(self)

    def _get_state(self):
        return ShipNavRocksLidar._get_state(self)
