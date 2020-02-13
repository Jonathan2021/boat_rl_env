#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:19:52 2020

@author: gfo
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)
import gym
from gym import spaces
from gym.utils import seeding

"""

The objective of this environment is control a ship to reach a target

STATE VARIABLES
The state consists of the following variables:
    - distance to target (ship's frame)
    - target bearing
    - angular velocity
    - gimbal angle
all state variables are roughly in the range [-1, 1]
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - no action

"""

FPS = 60

# THRUSTER
THRUSTER_MIN_THROTTLE = 0.4 # [%]
THRUSTER_MAX_ANGLE = 0.4    # [rad]
THRUSTER_MAX_FORCE = 3e4    # [N]


THRUSTER_HEIGHT = 20  # [m]
THRUSTER_WIDTH = 0.8 # [m]

# SHIP
SHIP_HEIGHT = 20    # [m]
SHIP_WIDTH = 5      # [m]

# SEA
SEA_H = 900    # [m]
SEA_W = 1600   # [m]

# ship model inspired by DP060
SHIP_MASS = 0.01*27e3 # [kg]
SHIP_INERTIA = 280e3 # [kg.m²]
Vmax = 100 # [m/s]
K_Nr = (THRUSTER_MAX_FORCE*SHIP_HEIGHT*math.sin(THRUSTER_MAX_ANGLE)/(2*2*math.pi)) # [N.m/(rad/s)]
K_Xu = THRUSTER_MAX_FORCE/Vmax # [N/(m/s)]
K_Yv = 10*K_Xu # [N/(m/s)]


# ROCK
ROCK_RADIUS = 20

n_Rocks = 25


class ShipNavigationWithObstaclesEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World(gravity=(0,0))
        self.ship = None
        self.target = None
        self.throttle = 0
        self.thruster_angle = 0.0

        high = np.ones(2, dtype=np.float32)
        low = -high

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.ship:
            return
        self.world.DestroyBody(self.ship)
        self.world.DestroyBody(self.target)
        for rock in self.rocks:
            self.world.DestroyBody(rock)
        self.ship = None

    def reset(self):
        print('reset env')
        self._destroy()
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.rudder_angle = 0.0
        self.stepnumber = 0
        self.rocks = []

        initial_x = np.random.uniform( 2*SHIP_HEIGHT, SEA_W-2*SHIP_HEIGHT)
        initial_y = np.random.uniform( 2*SHIP_HEIGHT, SEA_H-2*SHIP_HEIGHT)
        initial_heading = np.random.uniform(0, math.pi)
        
        targetX = np.random.uniform( 10*SHIP_HEIGHT, SEA_W-10*SHIP_HEIGHT)
        targetY = np.random.uniform( 10*SHIP_HEIGHT, SEA_H-10*SHIP_HEIGHT)
          
        self.target = self.world.CreateStaticBody(
                position = (targetX,targetY),
                angle = 0.0,
                fixtures = fixtureDef(
                        shape = circleShape(pos=(0,0),radius = ROCK_RADIUS),
                        categoryBits=0x0010,
                        maskBits=0x1111,
                        restitution=0.1))
        self.target.color1 = rgb(255,0,0)
        self.target.color2 = rgb(0,255,0)
        
        for i in range(n_Rocks):
            rock =self.world.CreateStaticBody(
                position=(np.random.uniform( 2*SHIP_HEIGHT, SEA_W-2*SHIP_HEIGHT), np.random.uniform( 2*SHIP_HEIGHT, SEA_H-2*SHIP_HEIGHT)),
                angle=np.random.uniform( 0, 2*math.pi),
                fixtures=fixtureDef(
                    shape = circleShape(pos=(0,0),radius = ROCK_RADIUS),
                categoryBits=0x0010,
                maskBits=0x1111,
                restitution=1.0))
            rock.color1 = rgb(83, 43, 9) # brown
            rock.color2 = rgb(41, 14, 9) # darker brown
            self.rocks.append(rock)
            
            
        
        self.ship = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=initial_heading,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-SHIP_WIDTH / 2, 0),
                                              (+SHIP_WIDTH / 2, 0),
                                              (SHIP_WIDTH / 2, +SHIP_HEIGHT),
                                              (0, +SHIP_HEIGHT*1.2),
                                              (-SHIP_WIDTH / 2, +SHIP_HEIGHT))),
                density=0.0,
                categoryBits=0x0010,
                maskBits=0x1111,
                restitution=0.0),
                linearDamping=0,
                angularDamping=0
        )

        self.ship.color1 = rgb(230, 230, 230)
        self.ship.linearVelocity = (0.0,0.0)
        self.ship.angularVelocity = 0
        
         
        newMassData = self.ship.massData
        newMassData.mass = SHIP_MASS
        newMassData.center = (0.0,SHIP_HEIGHT/2)
        newMassData.I = SHIP_INERTIA + SHIP_MASS*(newMassData.center[0]**2+newMassData.center[1]**2) # inertia is defined at origin location not localCenter
        self.ship.massData = newMassData
        
        
        self.drawlist = [self.ship, self.target] + self.rocks
        
       
        return self.step(2)[0]

    def step(self, action):

        if action == 0:
            self.thruster_angle += 0.01
        elif action == 1:
            self.thruster_angle -= 0.01

        self.thruster_angle = np.clip(self.thruster_angle, -THRUSTER_MAX_ANGLE, THRUSTER_MAX_ANGLE)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)

        # main engine force
        force_pos = (self.ship.position[0], self.ship.position[1])
        COGpos = self.ship.GetWorldPoint(self.ship.localCenter)
        force_thruster = (-np.sin(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE,
                  np.cos(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE )
        
        localVelocity = self.ship.GetLocalVector(self.ship.linearVelocity)

        force_damping_in_ship_frame = (-localVelocity[0] *K_Yv,-localVelocity[1] *K_Xu)
        
        force_damping = self.ship.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.ship.angle)* force_damping_in_ship_frame[0] -np.sin(self.ship.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.ship.angle)* force_damping_in_ship_frame[0] + np.cos(self.ship.angle) * force_damping_in_ship_frame[1] )
        force_total = tuple(map(lambda x, y: x + y, force_thruster, force_damping))

        torque_damping = -self.ship.angularVelocity *K_Nr

        self.ship.ApplyTorque(torque=torque_damping,wake=False)
        self.ship.ApplyForce(force=force_thruster, point=self.ship.position, wake=False)
        self.ship.ApplyForce(force=force_damping, point=COGpos, wake=False)
        
        self.world.Step(1.0 / FPS, 60, 60)
        
        pos = self.ship.position
        angle = self.ship.angle+np.pi/2
        vel_l = np.array(self.ship.linearVelocity)
        vel_a = self.ship.angularVelocity
            
    #- distance to target (ship's frame)
    #- target bearing
    #- angular velocity
    #- gimbal angle
        norm_pos = np.max((SEA_W,SEA_H))
        x_distance = (self.target.position[0] - pos.x)/norm_pos
        y_distance = (self.target.position[1] - pos.y)/norm_pos
        distance = np.linalg.norm((x_distance, y_distance))
        u = x_distance*np.cos(angle) + y_distance*np.sin(angle)
        v = -x_distance*np.sin(angle) + y_distance*np.cos(angle)
        bearing = np.arctan2(v,u)/(np.pi)
        #print("bearing = %0.3f u = %0.3f v = %0.3f" %(bearing,u,v))
        
        state = [
            distance,
            bearing,
            vel_a,
            (self.thruster_angle / THRUSTER_MAX_ANGLE),
            vel_l[0],
            vel_l[1],
            localVelocity[0],
            localVelocity[1]
        ]
        
        # # print state
        # if self.viewer is not None:
        #     print('\t'.join(["{:7.3}".format(s) for s in state]))

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        
        
        outside = (abs(pos.x - SEA_W*0.5) > SEA_W*0.49) or (abs(pos.y - SEA_H*0.5) > SEA_H*0.49)
        print('distance = {} \ttarget pos X = {}\tpos Y = {}'.format(distance,self.target.position[0] ,self.target.position[1] ))
        hit_target = (distance < (2*ROCK_RADIUS)/norm_pos)
        done = False
        
        reward = -1.0/FPS

        if outside:
            print('outside')
            self.game_over = True
            reward = -1 # high negative reward when outside of playground
        elif hit_target:
            print("target hit!!!",distance,(2*ROCK_RADIUS)/norm_pos)
            self.game_over = True
            reward = +1000  #high positive reward. hitting target is good
        else:   # general case, we're trying to reach target so being close should be rewarded
            reward = (1-distance**0.4) + (0.5-np.absolute(bearing)**0.4)
            
        if self.game_over:
            print("game over")
            done = True

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        return np.array(state[:]), reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(SEA_W, SEA_H)

            water = rendering.FilledPolygon(((0, 0), (0, SEA_H), (SEA_W, SEA_H), (SEA_W, 0)))
            self.water_color = rgb(126, 150, 233)
            water.set_color(*self.water_color)
            self.water_color_half_transparent = np.array((np.array(self.water_color) + rgb(255, 255, 255))) / 2
            self.viewer.add_geom(water)

            self.shiptrans = rendering.Transform()
            self.thrustertrans = rendering.Transform()
            self.COGtrans = rendering.Transform()
            
            thruster = rendering.FilledPolygon(((-THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT),
                                              (-THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT)))
            
            thruster.add_attr(self.thrustertrans) # add thruster angle
            thruster.add_attr(self.shiptrans) # add ship angle and ship position
            thruster.set_color(1.0, 1.0, 0.0)
            
            self.viewer.add_geom(thruster)
            
            COG = rendering.FilledPolygon(((-THRUSTER_WIDTH / 0.2, 0),
                                            (0, -THRUSTER_WIDTH/0.2),
                                              (THRUSTER_WIDTH / 0.2, 0),
                                              (0, THRUSTER_WIDTH/0.2)))
            COG.add_attr(self.COGtrans) # add COG position
            COG.add_attr(self.shiptrans) # add ship angle and ship position
            
            COG.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(COG)
            
            
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:   
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                

        self.shiptrans.set_translation(*self.ship.position)
        self.shiptrans.set_rotation(self.ship.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.COGtrans.set_translation(*self.ship.localCenter)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255