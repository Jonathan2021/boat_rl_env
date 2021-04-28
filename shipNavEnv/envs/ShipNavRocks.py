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
        self.viewer = None
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

        self.drawlist = [b.body for b in self.world.get_bodies()]
        
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
                print("Hit target, ending")
                done = True
            else:
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
                self.traj.append(COGpos)

        #print(state)
        if done: 
            print("Returning reward %d" % self.reward)
        return self.state, self.reward, done, {}

    def render(self, mode='human', close=False):
        #print([d.userData for d in self.drawlist])
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        ship = self.world.ship

        if self.viewer is None:

            self.viewer = rendering.Viewer(self.world.WIDTH, self.world.HEIGHT)
            
            water = rendering.FilledPolygon(((-10*self.world.WIDTH, -10*self.world.HEIGHT), (-10*self.world.WIDTH, 10*self.world.HEIGHT), (10*self.world.WIDTH, 10*self.world.HEIGHT), (10*self.world.WIDTH, -10*self.world.WIDTH)))
            self.water_color = rgb(126, 150, 233) #FIXME Why store it in self ? Check for other things like this
            water.set_color(*self.water_color)
            self.viewer.add_geom(water)
            
            self.shiptrans = rendering.Transform()
            self.thrustertrans = rendering.Transform()
            self.COGtrans = rendering.Transform()
            
            thruster = rendering.FilledPolygon(((-Ship.THRUSTER_WIDTH / 2, 0),
                                              (Ship.THRUSTER_WIDTH / 2, 0),
                                              (Ship.THRUSTER_WIDTH / 2, -Ship.THRUSTER_HEIGHT),
                                              (-Ship.THRUSTER_WIDTH / 2, -Ship.THRUSTER_HEIGHT)))
            
            thruster.add_attr(self.thrustertrans) # add thruster angle, assigned later
            thruster.add_attr(self.shiptrans) # add ship angle and ship position, assigned later
            thruster.set_color(*ship.body.color1)
            
            self.viewer.add_geom(thruster)
            
            COG = rendering.FilledPolygon(((-Ship.THRUSTER_WIDTH / 0.2, 0),
                                            (0, -Ship.THRUSTER_WIDTH/0.2),
                                              (Ship.THRUSTER_WIDTH / 0.2, 0),
                                              (0, Ship.THRUSTER_WIDTH/0.2)))
            COG.add_attr(self.COGtrans) # add COG position
            COG.add_attr(self.shiptrans) # add ship angle and ship position
            
            COG.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(COG)
            horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
            horizon.set_color(*ship.body.color1)
            horizon.add_attr(self.shiptrans) # add ship angle and ship position

            self.viewer.add_geom(horizon)

        
        #FIXME Feels pretty hacky, should check on that later
        # Adjusting window
        width_min = min(0, ship.body.position[0]-2*Ship.SHIP_HEIGHT)
        width_max = max(self.world.WIDTH, ship.body.position[0]+2*Ship.SHIP_HEIGHT)
        height_min = min(0, ship.body.position[1]-2*Ship.SHIP_HEIGHT)
        height_max = max(self.world.HEIGHT, ship.body.position[1]+2*Ship.SHIP_HEIGHT)
        ratio_w = (width_max-width_min)/self.world.WIDTH
        ratio_h = (height_max-height_min)/self.world.HEIGHT
        if ratio_w > ratio_h:
            height_min *= ratio_w/ratio_h
            height_max *= ratio_w/ratio_h
        else:
            width_min *= ratio_h/ratio_w
            width_max *= ratio_h/ratio_w
        
        self.viewer.set_bounds(width_min,width_max,height_min,height_max)
        

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData.is_hit() else (obj.color2 if f.body.userData.seen else obj.color3))).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color2 if f.body.userData.is_hit() else (obj.color3 if f.body.userData.seen else obj.color2)), filled=False, linewidth=2).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData.seen else obj.color3)).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color3 if f.body.userData.seen else obj.color1), filled=False, linewidth=2).add_attr(t)
                else:   
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                
        for j,dot in enumerate(self.traj):
            t = rendering.Transform(translation=dot)
            alpha = 1-(len(self.traj)-j)/len(self.traj)
            self.viewer.draw_circle(radius = 2, res=30, color = getColor(idx=0,alpha=alpha), filled=True).add_attr(t) 
            
        self.shiptrans.set_translation(*ship.body.position)
        self.shiptrans.set_rotation(ship.body.angle)
        self.thrustertrans.set_rotation(ship.thruster_angle)
        self.COGtrans.set_translation(*ship.body.localCenter)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
