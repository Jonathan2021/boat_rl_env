#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:19:52 2020

@author: Gilles Foinet
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import (circleShape, fixtureDef, polygonShape, contactListener)
import gym
from gym import spaces
from gym.utils import seeding
from shipNavEnv.utils import getColor, rgb
from shipNavEnv.Bodies import Ship, Rock
from shipNavEnv.Worlds import RockOnlyWorld

"""
The objective of this environment is control a ship to reach a target

STATE VARIABLES
The state consists of the following variables:
    - angular velocity normalized
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

MAX_STEPS = 1000    # max steps for a simulation
FPS = 60            # simulation framerate

# return (distance, bearing) tuple of target relatively to ship
def getDistanceBearing(ship,target):
    COGpos = ship.GetWorldPoint(ship.localCenter)
    x_distance = (target.position[0] - COGpos[0])
    y_distance = (target.position[1] - COGpos[1])
    localPos = ship.GetLocalVector((x_distance,y_distance))
    distance = np.linalg.norm(localPos)
    bearing = np.arctan2(localPos[0], localPos[1])
    return (distance, bearing)

# gym env class
class ShipNavRocks(gym.Env):

    def __init__(self,**kwargs):
        
        # Configuration
        # FIXME: Should move kwargs access in some configure function, can keep it in dict form (with defaults) and then iterate on keys
        #FIXME: Defaults should be in var
        self._read_kwargs(**kwargs)
           
        self.seed()
        self.world = RockOnlyWorld(self.n_rocks, {'obs_radius': self.obs_radius})
        
        # inital conditions
        # FIXME: Redundant with reset()
        self.episode_number = 0
        self.stepnumber = 0
        self.state = []
        self.reward = 0
        self.episode_reward = 0
        self.drawlist = None
        self.traj = []
        self.state = None
        
        # Observation are continuous in [-1, 1] 
        self.observation_space = spaces.Box(-1.0,1.0,shape=(4 +2*self.n_rocks_obs,), dtype=np.float32)
        
        # Left or Right (or nothing)
        self.action_space = spaces.Discrete(3)
       
        self.reset()

    def _read_kwargs(self, **kwargs):
        n_rocks_default = 0
        self.n_rocks = kwargs.get('n_rocks', n_rocks_default)

        n_rocks_obs_default = self.n_rocks
        self.n_rocks_obs = kwargs.get('n_rocks_obs', n_rocks_obs_default)

        obs_radius_default = 200
        self.obs_radius = kwargs.get('obs_radius', obs_radius_default)
        
        fps_default = FPS
        self.fps = kwargs.get('FPS', fps_default)

        display_traj_default = False
        self.display_traj = kwargs.get('display_traj', display_traj_default)

        display_traj_T_default = 0.1
        self.display_traj_T = kwargs.get('display_traj_T', display_traj_T_default)
         


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.world.reset()


        self.stepnumber = 0
        self.episode_reward = 0

        return self.step(2)[0] #FIXME Doesn't that mean we already do one time step ? Expected behavior ?

    def step(self, action):
        ship = self.world.ship

        done = False
        state = []

        #print('ACTION %d' % action)
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        # implement action
        # thruster angle and throttle saturation
        if action == 0:
            ship.steer(1, self.fps)
        elif action == 1:
            ship.steer(-1, self.fps)
        #else:
        #    print("Doing nothing !")

        ship.thrust(0, self.fps)

        self.world.step(self.fps)

        # Normalized ship states
        #state += list(np.asarray(self.ship.body.GetLocalVector(self.ship.body.linearVelocity))/Ship.Vmax)
        state.append(ship.body.angularVelocity/Ship.Rmax)
        state.append(ship.thruster_angle / Ship.THRUSTER_MAX_ANGLE)
        state.append(self.world.get_ship_target_standard_dist())
        state.append(self.world.get_ship_target_standard_bearing())
        
        obstacles = self.world.get_obstacles()

        # sort rocks from closest to farthest
        obstacles.sort(key=lambda x: (0 if x.seen else 1, x.distance_to_ship))
        
        for i in range(self.n_rocks_obs):
            if i < len(obstacles):
                state.append(self.world.get_ship_standard_dist(obstacles[i]))
                state.append(self.world.get_ship_standard_bearing(obstacles[i]))
            else:
                state.append(1)
                state.append(0)
        
        #FIXME Separate function
        # REWARD -------------------------------------------------------------------------------------------------------
        self.reward = 0
        #print(distance_t)
        
        if ship.is_hit(): # FIXME Will not know if hit is new or not !
            if self.world.target in ship.hit_with:
                self.reward = +10 #high positive reward. hitting target is good
                #print("Hit target, ending")
                done = True
            else:
                pass
                self.reward = -0.5 #high negative reward. hitting anything else than target is bad
            #done = True
        else:   # general case, we're trying to reach target so being close should be rewarded
            pass
            #self.reward = - 1/ MAX_STEPS
            #self.reward = - ((2* distance_t / norm_pos)  - 1) / MAX_STEPS # FIXME Macro instead of magic number
            #print(self.reward)
        
        # limits episode to MAX_STEPS
        if self.stepnumber >= MAX_STEPS:
            #self.reward = -1
            done = True

        self.episode_reward += self.reward

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1
        self.state = np.array(state, dtype=np.float32)
        
        #FIXME separate function
        #render trajectory
        if self.display_traj:
            if self.stepnumber % int(self.display_traj_T*self.fps) == 0: #FIXME If fps is low then int(<1) -> Division by 0 error. Should Take math.ceil instead or something.
                self.traj.append(COGpos) #FIXME

        # print(state)
        # if done: 
        #     print("Returning reward %d" % self.reward)
        return self.state, self.reward, done, {}

    def render(self, mode='human', close=False):
        #print([d.userData for d in self.drawlist])
        return self.world.render(mode, close)
