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

# SEA
SEA_H = 900                 # [m]
SEA_W = 1600                # [m]

# ROCK
ROCK_RADIUS = 20            # [m]

# return (distance, bearing) tuple of target relatively to ship
def getDistanceBearing(ship,target):
    COGpos = ship.GetWorldPoint(ship.localCenter)
    x_distance = (target.position[0] - COGpos[0])
    y_distance = (target.position[1] - COGpos[1])
    localPos = ship.GetLocalVector((x_distance,y_distance))
    distance = np.linalg.norm(localPos)
    bearing = np.arctan2(localPos[0], localPos[1])
    return (distance, bearing)

# collision handler
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        contact.fixtureA.body.userData['hit'] = True
        contact.fixtureA.body.userData['hit_with'] = contact.fixtureB.body.userData['name']
        contact.fixtureB.body.userData['hit'] = True
        contact.fixtureB.body.userData['hit_with'] = contact.fixtureA.body.userData['name']
        #print('There was a contact!')
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

# gym env class
class ShipNavRocks(gym.Env):

    def __init__(self,**kwargs):
        
        # Configuration
        # FIXME: Should move kwargs access in some configure function, can keep it in dict form (with defaults) and then iterate on keys
        #FIXME: Defaults should be in var
        self._read_kwargs(**kwargs)
           
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0,0),
            contactListener=ContactDetector(self))
        self.ship = None
        self.target = None
        self.rocks = []
        self.ships = []
        
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
        self.n_rocks = kwargs.get('n_rocks',n_rocks_default)

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

    def _destroy(self):
        self.world.contactListener = None

        if self.ship: self.ship.destroy()
        if self.target: self.world.DestroyBody(self.target)
        while self.rocks :
            self.rocks.pop(0).destroy()

        self.ship = None
        self.target = None
        self.rocks = []
    
    def _create_map(self):
        for i in range(self.n_rocks):
            radius = np.random.uniform( 0.5*ROCK_RADIUS,2*ROCK_RADIUS)
            rock = Rock(self.world, np.random.uniform(0, SEA_W), np.random.uniform(0, SEA_H))
            self.rocks.append(rock)
        
        getDistToRockfield = lambda x,y: np.asarray([np.sqrt((rock.body.position.x - x)**2 + (rock.body.position.y - y)**2) for rock in self.rocks]).min() if len(self.rocks) > 0 else np.inf # infinite distance if there is no rock field
        

        # create boat position randomly, but not overlapping an existing rock
        initial_x, initial_y = np.random.uniform( [0 , 0], [SEA_W, SEA_H])
        while(getDistToRockfield(initial_x, initial_y) < 2*ROCK_RADIUS):
            initial_x, initial_y = np.random.uniform( [0 ,0], [SEA_W, SEA_H])

        initial_heading = np.random.uniform(0, 2*math.pi)
        
        # create target randomly, but not overlapping an existing rock
        targetX, targetY = np.random.uniform( [0 , 0], [SEA_W, SEA_H])

        #FIXME Repeats itself 
        while(getDistToRockfield(targetX,targetY) < 2*ROCK_RADIUS):
            targetX, targetY = np.random.uniform( [0, 0], [SEA_W, SEA_H])
        
          
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

        self.ship = Ship(self.world, initial_x, initial_y, initial_heading)

    def reset(self):
        self._destroy()

        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref


        self.stepnumber = 0
        self.episode_reward = 0
        self.rocks = []
        self.traj = []

        self._create_map()

        self.ship.reset() # Will be reseted in new World class reset

        self.ships.append(self.ship)
        self.drawlist = [s.body for s in self.ships] + [self.target] + [r.body for r in self.rocks]
        
        return self.step(2)[0] #FIXME Doesn't that mean we already do one time step ? Expected behavior ?

    def step(self, action):
        done = False
        state = []
        print('ACTION %d' % action)
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        # implement action
        # thruster angle and throttle saturation
        if action == 0:
            self.ship.steer(1)
        elif action == 1:
            self.ship.steer(-1)
        #else:
        #    print("Doing nothing !")

        self.ship.thrust(0)

        # main engine force
        COGpos = self.ship.body.GetWorldPoint(self.ship.body.localCenter)

        force_thruster = (-np.sin(self.ship.body.angle + self.ship.thruster_angle) * Ship.THRUSTER_MAX_FORCE,
                  np.cos(self.ship.body.angle + self.ship.thruster_angle) * Ship.THRUSTER_MAX_FORCE )
        
        localVelocity = self.ship.body.GetLocalVector(self.ship.body.linearVelocity)

        force_damping_in_ship_frame = (-localVelocity[0] *Ship.K_Yv,-localVelocity[1] *Ship.K_Xu)
        
        force_damping = self.ship.body.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.ship.body.angle)* force_damping_in_ship_frame[0] -np.sin(self.ship.body.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.ship.body.angle)* force_damping_in_ship_frame[0] + np.cos(self.ship.body.angle) * force_damping_in_ship_frame[1] )
        
        torque_damping = -self.ship.body.angularVelocity *Ship.K_Nr

        self.ship.body.ApplyTorque(torque=torque_damping,wake=False)
        self.ship.body.ApplyForce(force=force_thruster, point=self.ship.body.position, wake=False)
        self.ship.body.ApplyForce(force=force_damping, point=COGpos, wake=False)


        ### DEBUG ###
        #print('Step: %d \nShip: %s\nLocals: %s' % (self.stepnumber, self.ship, locals()))
        
        # one step forward
        velocityIterations = 8
        positionIterations = 3
        self.world.Step(1.0 / self.fps, velocityIterations, positionIterations)

        # state construction
        norm_pos = np.max((SEA_W, SEA_H))
        distance_t, bearing_t = getDistanceBearing(self.ship.body ,self.target)
        #print(bearing_t)
        #print(distance_t)
        
        # Normalized ship states
        #state += list(np.asarray(self.ship.body.GetLocalVector(self.ship.body.linearVelocity))/Ship.Vmax)
        state.append(self.ship.body.angularVelocity/Ship.Rmax)
        state.append(self.ship.thruster_angle / Ship.THRUSTER_MAX_ANGLE)
        standardized_dist = (2* distance_t / norm_pos)  - 1
        state.append(standardized_dist)
        state.append(bearing_t/np.pi)
        
        for rock in self.rocks:
            distance, bearing = getDistanceBearing(self.ship.body, rock.body)
            distance = np.maximum(distance-rock.body.userData['radius'],0) #FIXME Is this useful ? If ship collides with rock, the engine notifies us right? + We don't take into account the ships geometry.
            rock.body.userData['distance_to_ship'] = 2 * distance/norm_pos - 1
            rock.body.userData['bearing_from_ship'] = bearing/np.pi
            rock.body.userData['in_range'] = True if distance < self.obs_radius else False #FIXME From center of ship center to center of rock.body. Meaning it wouldn't see very large rocks
        
        # sort rocks from closest to farthest
        self.rocks.sort(key=lambda x:x.body.userData['distance_to_ship'])
        
        # set 'seen' bool
        #FIXME Could be done in previous loop (a bit more efficient)
        for i in range(self.n_rocks_obs):
            if self.rocks[i].body.userData['in_range']:
                self.rocks[i].body.userData['seen']=True 
                state.append(self.rocks[i].body.userData['distance_to_ship'])
                state.append(self.rocks[i].body.userData['bearing_from_ship'])
            else: #if closest rocks are outside horizon, fill observation with rocks infinitely far on the ship axis
                self.rocks[i].body.userData['seen']=False
                state.append(1) #FIXME Maybe don't include them in state instead of choosing arbitrary values
                state.append(0)
        for rock in self.rocks[self.n_rocks_obs:]:
            rock.body.userData['seen']=False

        #FIXME Separate function
        # REWARD -------------------------------------------------------------------------------------------------------
        self.reward = 0
        #print(distance_t)
        
        if self.ship.body.userData['hit']:
            if(self.ship.body.userData['hit_with']=='target'):
                self.reward = +10 #high positive reward. hitting target is good
            else:
                self.reward = -1 #high negative reward. hitting anything else than target is bad
            done = True
        else:   # general case, we're trying to reach target so being close should be rewarded
            pass
            #self.reward = - 1/ MAX_STEPS
            #self.reward = - ((2* distance_t / norm_pos)  - 1) / MAX_STEPS # FIXME Macro instead of magic number
            #print(self.reward)
        
        # limits episode to MAX_STEPS
        if self.stepnumber >= MAX_STEPS:
            self.reward = -1
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
            
            water = rendering.FilledPolygon(((-10*SEA_W, -10*SEA_H), (-10*SEA_W, 10*SEA_H), (10*SEA_W, 10*SEA_H), (10*SEA_W, -10*SEA_W)))
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
            thruster.set_color(*self.ship.body.color1)
            
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
            horizon.set_color(*self.ship.body.color1)
            horizon.add_attr(self.shiptrans) # add ship angle and ship position

            self.viewer.add_geom(horizon)

        
        #FIXME Feels pretty hacky, should check on that later
        # Adjusting window
        width_min = min(0,self.ship.body.position[0]-2*Ship.SHIP_HEIGHT)
        width_max = max(SEA_W,self.ship.body.position[0]+2*Ship.SHIP_HEIGHT)
        height_min = min(0,self.ship.body.position[1]-2*Ship.SHIP_HEIGHT)
        height_max = max(SEA_H,self.ship.body.position[1]+2*Ship.SHIP_HEIGHT)
        ratio_w = (width_max-width_min)/SEA_W
        ratio_h = (height_max-height_min)/SEA_H
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
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData['hit'] else (obj.color2 if f.body.userData['seen'] else obj.color3))).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color2 if f.body.userData['hit'] else (obj.color3 if f.body.userData['seen'] else obj.color2)), filled=False, linewidth=2).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color1 if f.body.userData['in_range'] else obj.color3)).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color= (obj.color3 if f.body.userData['in_range'] else obj.color1), filled=False, linewidth=2).add_attr(t)
                else:   
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                
        for j,dot in enumerate(self.traj):
            t = rendering.Transform(translation=dot)
            alpha = 1-(len(self.traj)-j)/len(self.traj)
            self.viewer.draw_circle(radius = 2, res=30, color = getColor(idx=0,alpha=alpha), filled=True).add_attr(t) 
            
        self.shiptrans.set_translation(*self.ship.body.position)
        self.shiptrans.set_rotation(self.ship.body.angle)
        self.thrustertrans.set_rotation(self.ship.thruster_angle)
        self.COGtrans.set_translation(*self.ship.body.localCenter)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
