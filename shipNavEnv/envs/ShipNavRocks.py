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

"""
The objective of this environment is control a ship to reach a target

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

MAX_STEPS = 1000    # max steps for a simulation
FPS = 60            # simulation framerate

# THRUSTER
THRUSTER_MIN_THROTTLE = 0.4 # [%]
THRUSTER_MAX_ANGLE = 0.4    # [rad]
THRUSTER_MAX_FORCE = 3e4    # [N]

THRUSTER_HEIGHT = 20        # [m]
THRUSTER_WIDTH = 0.8        # [m]

# SHIP
SHIP_HEIGHT = 20            # [m]
SHIP_WIDTH = 5              # [m]

# SEA
SEA_H = 900                 # [m]
SEA_W = 1600                # [m]

# ship model
# dummy parameters for fast simulation
SHIP_MASS = 27e1            # [kg]
SHIP_INERTIA = 280e1        # [kg.m²]
Vmax = 300                  # [m/s]
Rmax = 1*np.pi              #[rad/s]
K_Nr = (THRUSTER_MAX_FORCE*SHIP_HEIGHT*math.sin(THRUSTER_MAX_ANGLE)/(2*Rmax)) # [N.m/(rad/s)]
K_Xu = THRUSTER_MAX_FORCE/Vmax # [N/(m/s)]
K_Yv = 10*K_Xu              # [N/(m/s)]


# ROCK
ROCK_RADIUS = 20            # [m]

# return (distance, bearing) tuple of target relatively to ship
def getDistanceBearing(ship,target):
    COGpos = ship.GetWorldPoint(ship.localCenter)
    x_distance = (target.position[0] - COGpos[0])
    y_distance = (target.position[1] - COGpos[1])
    localPos = ship.GetLocalVector((x_distance,y_distance))
    distance = np.linalg.norm(localPos)
    bearing = np.arctan2(localPos[0],localPos[1])
    return (distance, bearing)

# collision handler
class myContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)
    def BeginContact(self, contact):
        #print(' with '.join((contact.fixtureA.body.userData['name'],contact.fixtureB.body.userData['name'])))
        contact.fixtureA.body.userData['hit'] = True
        contact.fixtureA.body.userData['hit_with'] = contact.fixtureB.body.userData['name']
        contact.fixtureB.body.userData['hit'] = True
        contact.fixtureB.body.userData['hit_with'] = contact.fixtureA.body.userData['name']
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

# gym env class
class ShipNavRocks(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self,**kwargs):
        if 'n_rocks' in kwargs.keys():
            self.n_rocks = kwargs['n_rocks']
        else:
            self.n_rocks = 0
        if 'n_rocks_obs' in kwargs.keys():
            self.n_rocks_obs = kwargs['n_rocks_obs']
        else:
            self.n_rocks_obs = self.n_rocks
        if 'obs_radius' in kwargs.keys():
            self.obs_radius = kwargs['obs_radius']
        else:
            self.obs_radius = 200
        if 'FPS' in kwargs.keys():
            self.fps = kwargs['FPS']
            self.metadata['video.frames_per_second'] = self.fps
        else:
            self.fps = 60
        if 'display_traj' in kwargs.keys():
            self.display_traj = kwargs['display_traj']
        else:
            self.display_traj = False
        if 'display_traj_T' in kwargs.keys():
            self.display_traj_T = kwargs['display_traj_T']
        else:
            self.display_traj_T = 0.1
            
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0,0),
            contactListener=myContactListener())
        self.ship = None
        self.target = None
        self.rocks = []
        self.ships = []
        
        self.episode_number = 0
        self.stepnumber = 0
        self.game_over = False
        self.throttle = 0
        self.thruster_angle = 0.0
        self.state = []
        self.reward = 0
        self.episode_reward = 0
        self.drawlist = None
        
        self.observation_space = spaces.Box(-1.0,1.0,shape=(4 +2*self.n_rocks_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.traj = []
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.ship: return
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.target)
        self.target = None
        while self.rocks :
            self.world.DestroyBody(self.rocks.pop(0))

    def reset(self):
        self._destroy()
        self.game_over = False
        self.throttle = 0
        self.thruster_angle = 0.0
        self.stepnumber = 0
        self.episode_reward = 0
        self.rocks = []
        self.traj = []

        # create rock field randomly
        for i in range(self.n_rocks):
            radius = np.random.uniform( 0.5*ROCK_RADIUS,2*ROCK_RADIUS)
            rock =self.world.CreateStaticBody(
                position=(np.random.uniform( 2*SHIP_HEIGHT, SEA_W-2*SHIP_HEIGHT), np.random.uniform( 2*SHIP_HEIGHT, SEA_H-2*SHIP_HEIGHT)),
                angle=np.random.uniform( 0, 2*math.pi),
                fixtures=fixtureDef(
                    shape = circleShape(pos=(0,0),radius = radius),
                categoryBits=0x0010,
                maskBits=0x1111,
                restitution=1.0))
            rock.color1 = rgb(83, 43, 9) # brown
            rock.color2 = rgb(41, 14, 9) # darker brown
            rock.color3 = rgb(255, 255, 255) # seen
            rock.userData = {'id':i,
                             'name':'rock',
                             'hit':False,
                             'hit_with':'',
                             'distance_to_ship':1.0,
                             'bearing_from_ship':0.0,
                             'seen':False,
                             'in_range':False,
                             'radius':radius}
            
            self.rocks.append(rock)
        
        getDistToRockfield = lambda x,y: np.asarray([np.sqrt((rock.position.x - x)**2 + (rock.position.y - y)**2) for rock in self.rocks]).min() if len(self.rocks) > 0 else np.inf # infinite distance if there is no rock field
        
        # create target randomly, but not overlapping an existing rock
        initial_x, initial_y = np.random.uniform( [2*SHIP_HEIGHT ,2*SHIP_HEIGHT], [SEA_W-2*SHIP_HEIGHT,SEA_H-2*SHIP_HEIGHT])
        while(getDistToRockfield(initial_x, initial_y) < 3*ROCK_RADIUS):
            initial_x, initial_y = np.random.uniform( [2*SHIP_HEIGHT ,2*SHIP_HEIGHT], [SEA_W-2*SHIP_HEIGHT,SEA_H-2*SHIP_HEIGHT])

        initial_heading = np.random.uniform(0, math.pi)
        
        # create target randomly, but not overlapping an existing rock
        targetX, targetY = np.random.uniform( [10*SHIP_HEIGHT ,10*SHIP_HEIGHT], [SEA_W-10*SHIP_HEIGHT,SEA_H-10*SHIP_HEIGHT])

        while(getDistToRockfield(targetX,targetY) < 3*ROCK_RADIUS):
            targetX, targetY = np.random.uniform( [10*SHIP_HEIGHT ,10*SHIP_HEIGHT], [SEA_W-10*SHIP_HEIGHT,SEA_H-10*SHIP_HEIGHT])
        
          
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
        self.target.color3 = rgb(255, 255, 255) # seen
        self.target.userData = {'name':'target',
                                'hit':False,
                                'hit_with':'',
                                'seen':True,
                                'in_range':False}

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
        self.ship.userData = {'name':'ship',
                              'hit':False,
                              'hit_with':''}
        
         
        newMassData = self.ship.massData
        newMassData.mass = SHIP_MASS
        newMassData.center = (0.0,SHIP_HEIGHT/2)
        newMassData.I = SHIP_INERTIA + SHIP_MASS*(newMassData.center[0]**2+newMassData.center[1]**2) # inertia is defined at origin location not localCenter
        self.ship.massData = newMassData
        
        self.ships.append(self.ship)
        self.drawlist = self.ships + [self.target] + self.rocks
        
        return self.step(2)[0]

    def step(self, action):
        done = False
        state = []
        # implement action
        if action == 0:
            self.thruster_angle += 0.01*60/(self.fps)
        elif action == 1:
            self.thruster_angle -= 0.01*60/(self.fps)

        # thruster angle and throttle saturation
        self.thruster_angle = np.clip(self.thruster_angle, -THRUSTER_MAX_ANGLE, THRUSTER_MAX_ANGLE)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)

        # main engine force
        COGpos = self.ship.GetWorldPoint(self.ship.localCenter)
        force_thruster = (-np.sin(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE,
                  np.cos(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE )
        
        localVelocity = self.ship.GetLocalVector(self.ship.linearVelocity)

        force_damping_in_ship_frame = (-localVelocity[0] *K_Yv,-localVelocity[1] *K_Xu)
        
        force_damping = self.ship.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.ship.angle)* force_damping_in_ship_frame[0] -np.sin(self.ship.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.ship.angle)* force_damping_in_ship_frame[0] + np.cos(self.ship.angle) * force_damping_in_ship_frame[1] )
        
        torque_damping = -self.ship.angularVelocity *K_Nr

        self.ship.ApplyTorque(torque=torque_damping,wake=False)
        self.ship.ApplyForce(force=force_thruster, point=self.ship.position, wake=False)
        self.ship.ApplyForce(force=force_damping, point=COGpos, wake=False)
        
        # one step forward
        self.world.Step(1.0 / self.fps, 60, 60)

        # state construction
        norm_pos = np.max((SEA_W,SEA_H))
        distance_t, bearing_t = getDistanceBearing(self.ship,self.target)
        
        #state += list(np.asarray(self.ship.GetLocalVector(self.ship.linearVelocity))/Vmax)
        state.append(self.ship.angularVelocity/Rmax)
        state.append(self.thruster_angle / THRUSTER_MAX_ANGLE)
        state.append(distance_t/norm_pos)
        state.append(bearing_t/np.pi)
        
        for rock in self.rocks:
            distance, bearing = getDistanceBearing(self.ship,rock)
            distance = np.maximum(distance-rock.userData['radius'],0)
            rock.userData['distance_to_ship'] = distance/norm_pos
            rock.userData['bearing_from_ship'] = bearing/np.pi
            rock.userData['in_range'] = True if distance < self.obs_radius else False
        
        # sort rocks from closest to farthest
        self.rocks.sort(key=lambda x:x.userData['distance_to_ship'])
        
        # set 'seen' bool
        for i in range(self.n_rocks_obs):
            if self.rocks[i].userData['in_range']:
                self.rocks[i].userData['seen']=True 
                state.append(self.rocks[i].userData['distance_to_ship'])
                state.append(self.rocks[i].userData['bearing_from_ship'])
            else: #if closest rocks are outside horizon, fill observation with rocks infinitely far on the ship axis
                self.rocks[i].userData['seen']=False
                state.append(1)
                state.append(0)
        for rock in self.rocks[self.n_rocks_obs:]:
            rock.userData['seen']=False

        # REWARD -------------------------------------------------------------------------------------------------------
        self.reward = 0
        
        if self.ship.userData['hit']:
            if(self.ship.userData['hit_with']=='target'):
                self.reward = +10  #high positive reward. hitting target is good
            else:
                self.reward = -1 #high negative reward. hitting anything else than target is bad
            done = True
        else:   # general case, we're trying to reach target so being close should be rewarded
            self.reward = (distance_t/norm_pos)/1000
        
        # limits episode to MAX_STEPS
        if self.stepnumber >= MAX_STEPS:
            done = True

        self.episode_reward += self.reward

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1
        self.state = np.array(state, dtype=np.float32)
        
        #render trajectory
        if self.display_traj:
            if self.stepnumber % int(self.display_traj_T*self.fps) == 0:
                self.traj.append(COGpos)
        
        return self.state, self.reward, done, {}

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
            horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
            horizon.set_color(*rgb(200, 230, 255))
            horizon.add_attr(self.shiptrans) # add ship angle and ship position

            self.viewer.add_geom(horizon)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData['hit'] else (obj.color2 if f.body.userData['seen'] else obj.color3))).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color2 if f.body.userData['hit'] else (obj.color3 if f.body.userData['seen'] else obj.color2)), filled=False, linewidth=2).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData['in_range'] else obj.color3)).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color3 if f.body.userData['in_range'] else obj.color1), filled=False, linewidth=2).add_attr(t)
                else:   
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                
        for dot in self.traj:
            t = rendering.Transform(translation=dot)
            self.viewer.draw_circle(radius = 2, res=30, color = (0,50,0), filled=True).add_attr(t) 
            
        self.shiptrans.set_translation(*self.ship.position)
        self.shiptrans.set_rotation(self.ship.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.COGtrans.set_translation(*self.ship.localCenter)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
