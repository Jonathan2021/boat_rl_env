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
from shipNavEnv.utils import getColor, rgb, draw_random_in_list
from shipNavEnv.Worlds import RockOnlyWorld, RockOnlyWorldLidar

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


# gym env class
class ShipNavRocks(gym.Env):
    MAX_STEPS = 1000    # max steps for a simulation
    FPS = 60            # simulation framerate

    def __init__(self,**kwargs):
        
        # Configuration
        # FIXME: Should move kwargs access in some configure function, can keep it in dict form (with defaults) and then iterate on keys
        #FIXME: Defaults should be in var
        self._read_kwargs(**kwargs)
           
        self.seed()
        self.world = self._build_world()
        
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
        self.obstacles = []
        
        # Observation are continuous in [-1, 1] 
        self.observation_space = self._get_obs_space()
        
        # Left or Right (or nothing)
        self.action_space = spaces.Discrete(3)
       
        self.reset()

    def _build_world(self):
        return RockOnlyWorld(self.n_rocks, self.rock_scale, {'obs_radius': self.obs_radius})

    def _get_obs_space(self):
        return spaces.Box(-1.0,1.0,shape=(4 + 2 * self.n_rocks_obs,), dtype=np.float32)


    def _read_kwargs(self, **kwargs):
        n_rocks_default = 0
        self.n_rocks = kwargs.get('n_rocks', n_rocks_default)

        rock_scale_default = RockOnlyWorld.ROCK_SCALE_DEFAULT
        self.rock_scale = kwargs.get('rock_scale', rock_scale_default)

        n_rocks_obs_default = self.n_rocks
        self.n_rocks_obs = kwargs.get('n_rocks_obs', n_rocks_obs_default)
        self.n_obstacles_obs = self.n_rocks_obs

        obs_radius_default = 800
        self.obs_radius = kwargs.get('obs_radius', obs_radius_default)
        
        fps_default = self.FPS
        self.fps = kwargs.get('self.FPS', fps_default)

        display_traj_default = False
        self.display_traj = kwargs.get('display_traj', display_traj_default)

        display_traj_T_default = 0.1
        self.display_traj_T = kwargs.get('display_traj_T', display_traj_T_default)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.world.reset()
        self.obstacles = []


        self.stepnumber = 0
        self.episode_reward = 0

        return self._get_state()
    
    def _take_actions(self, action):
        ship = self.world.ship
        
        #print('ACTION %d' % action)
        #assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        # implement action
        # thruster angle and throttle saturation
        if action is not None:
            if action == 0:
                ship.steer(1, self.fps)
            elif action == 1:
                ship.steer(-1, self.fps)
        #else:
        #    print("Doing nothing !")

        #ship.thrust(0, self.fps)

    def _get_obstacles(self):
        return self._get_obstacles_conservative()

    def _get_obstacles_conservative(self):
        obstacles = self.world.get_obstacles()
        # sort rocks from closest to farthest
        obstacles.sort(key=lambda x: (0 if x.seen else 1, x.distance_to_ship))

        new_seen = []
        for obs in obstacles:
            if obs.seen and obs not in self.obstacles: # Obstacle is seen and isn't in previously observed obstacles
                new_seen.append(obs)
            
        new_obstacles = []
        #print("new seen %s" % new_seen)
        for obs in self.obstacles:
            if not obs.seen and new_seen: # If old obstacle seen is not seen anymore, and some new to add
                new_obstacles.append(new_seen[0]) #Add closest one instead of a random one
                del new_seen[0]
                # new_obstacles.append(draw_random_in_list(new_seen))
            else:
                # If still valid
                # If no new to replace, keep the old one (it's seen value should have been reset and will be handled in the function getting state)
                new_obstacles.append(obs)
            
        new_obstacles += new_seen # If still some new seen left, add them at the tail
        self.obstacles = new_obstacles

        #FIXME Should it be handled in the world itself ?
        for obs in self.obstacles[self.n_obstacles_obs:]: # see only limited amount of obstacles
            obs.seen = False

        self.obstacles = self.obstacles[:self.n_obstacles_obs] # keep track for next iteration of that limited amount
        
        return self.obstacles

    def _get_ship_state(self):
        ship = self.world.ship

        state = []
        #state.append(ship.body.linearVelocity)
        state.append(ship.body.angularVelocity/ship.Rmax)
        state.append(ship.thruster_angle / ship.THRUSTER_MAX_ANGLE)
        state.append(self.world.get_ship_target_standard_dist())
        state.append(self.world.get_ship_target_standard_bearing())
        return state        


    def _get_state(self):
        ship = self.world.ship
        state = self._get_ship_state()
        obstacles = self._get_obstacles()
        
        for i in range(self.n_obstacles_obs):
            if i < len(obstacles) and obstacles[i].seen:
                state.append(np.clip(2 * self.world.get_ship_dist(obstacles[i]) / ship.obs_radius - 1, -1, 1))
                state.append(np.clip(self.world.get_ship_standard_bearing(obstacles[i]), -1, 1))
            else:
                state.append(1)
                state.append(np.random.uniform(-1, 1))
        return np.array(state, dtype=np.float32)
    
    def _get_reward_done(self):
        ship = self.world.ship

        reward = 0
        done = False
        #print(distance_t)
        
        if ship.is_hit(): # FIXME Will not know if hit is new or not !
            if self.world.target in ship.hit_with:
                reward += 10 #high positive reward. hitting target is good
                #print("Hit target, ending")
            else:
                reward -= 5 #high negative reward. hitting anything else than target is bad
            done = True
        else:   # general case, we're trying to reach target so being close should be rewarded
            reward -= 1/ self.MAX_STEPS
            #self.reward = - ((2* distance_t / norm_pos)  - 1) / self.self.MAX_STEPS # FIXME Macro instead of magic number
            #print(self.reward)
        
        # limits episode to self.MAX_STEPS
        if self.stepnumber >= self.MAX_STEPS:
            reward -= 5
            done = True

        return reward, done

    def step(self, action):
        ship = self.world.ship

        self._take_actions(action)

        self.world.step(self.fps)
        self.stepnumber += 1
        
        self.state = self._get_state()

        #if self.stepnumber == 1:
        #    print(len(self.state))
        #print(ship.body.linearVelocity)
        #print(np.sqrt(ship.body.linearVelocity[0]**2 + ship.body.linearVelocity[1]**2))
        #if not (self.stepnumber % self.FPS):
        #    print(self.state)

        # Normalized ship states
        #state += list(np.asarray(self.ship.body.GetLocalVector(self.ship.body.linearVelocity))/Ship.Vmax)
       
        
        #FIXME Separate function
        
        self.reward, done = self._get_reward_done()
        self.episode_reward += self.reward

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

class ShipNavRocksContinuousSteer(ShipNavRocks):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32) # steer right or left

    def _take_actions(self, action):
        if action is not None:
            self.world.ship.steer(action[0], self.fps)

class ShipNavRocksSteerAndThrustDiscrete(ShipNavRocks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.MultiDiscrete([3, 3])

    def _take_actions(self, action):
        if action is not None:
            super()._take_actions(self, action[0])
            if action[1] == 0:
                self.world.ship.thrust(-1, self.fps)
            elif action[1] == 1:
                self.world.ship.thrust(1, self.fps)


class ShipNavRocksLidar(ShipNavRocks):
    def _read_kwargs(self, **kwargs):
        n_rocks_default = 0
        self.n_rocks = kwargs.get('n_rocks', n_rocks_default)

        rock_scale_default = RockOnlyWorldLidar.ROCK_SCALE_DEFAULT
        self.rock_scale = kwargs.get('rock_scale', rock_scale_default)

        n_lidars_default = 10
        self.n_lidars = kwargs.get('n_lidars', n_lidars_default)
        
        fps_default = self.FPS
        self.fps = kwargs.get('self.FPS', fps_default)

        display_traj_default = False
        self.display_traj = kwargs.get('display_traj', display_traj_default)

        display_traj_T_default = 0.1
        self.display_traj_T = kwargs.get('display_traj_T', display_traj_T_default)

    def _build_world(self):
        return RockOnlyWorldLidar(self.n_rocks, self.n_lidars, self.rock_scale)

    def _get_obs_space(self):
        return spaces.Box(-1.0,1.0,shape=(4 +  self.n_lidars,), dtype=np.float32)
    
    def _get_state(self):
        ship = self.world.ship
        state = self._get_ship_state()

        state += [l.fraction for l in ship.lidars] 
        
        return np.array(state, dtype=np.float32)
            
