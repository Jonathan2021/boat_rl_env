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
from shipNavEnv.envs.utils import getColor

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
        self.throttle = 0
        self.thruster_angle = 0.0
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

        if self.ship: self.world.DestroyBody(self.ship)
        if self.target: self.world.DestroyBody(self.target)
        while self.rocks :
            self.world.DestroyBody(self.rocks.pop(0))

        self.ship = None
        self.target = None
        self.rocks = []
    
    def _create_map(self):
        for i in range(self.n_rocks):
            radius = np.random.uniform( 0.5*ROCK_RADIUS,2*ROCK_RADIUS)
            rock =self.world.CreateStaticBody(
                position=(np.random.uniform(0, SEA_W), np.random.uniform(0, SEA_H)), # FIXME Should have something like: map.get_random_available_position()
                angle=np.random.uniform( 0, 2*math.pi), # FIXME Not really useful if circle shaped
                fixtures=fixtureDef(
                shape = circleShape(pos=(0,0),radius = radius),
                categoryBits=0x0010, # FIXME Move categories to MACRO
                maskBits=0x1111, # FIXME Same as above + it can collide with itself -> may cause problem when generating map ?
                restitution=1.0))
            rock.color1 = rgb(83, 43, 9) # brown
            rock.color2 = rgb(41, 14, 9) # darker brown
            rock.color3 = rgb(255, 255, 255) # seen
            rock.userData = {'id':i,
                             'name':'rock',
                             'hit':False,
                             'hit_with':'',
                             'distance_to_ship':1.0, # FIXME are the 4 lines, including this one, used in any way?
                             'bearing_from_ship':0.0,
                             'seen':False,
                             'in_range':False,
                             'radius':radius}
            
            self.rocks.append(rock)
        
        getDistToRockfield = lambda x,y: np.asarray([np.sqrt((rock.position.x - x)**2 + (rock.position.y - y)**2) for rock in self.rocks]).min() if len(self.rocks) > 0 else np.inf # infinite distance if there is no rock field
        

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
                categoryBits=0x0010, #FIXME Same category as rocks ?
                maskBits=0x1111,
                restitution=0.0),
            linearDamping=0,
            angularDamping=0
        )

        self.ship.color1 = getColor(idx=0)
        self.ship.linearVelocity = (0.0,0.0)
        self.ship.angularVelocity = 0
        self.ship.userData = {'name':'ship',
                              'hit':False,
                              'hit_with':''}
        


    def reset(self):
        self._destroy()

        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref


        self.throttle = 0
        self.thruster_angle = 0.0
        self.stepnumber = 0
        self.episode_reward = 0
        self.rocks = []
        self.traj = []

        self._create_map()

                 
        newMassData = self.ship.massData
        newMassData.mass = SHIP_MASS
        newMassData.center = (0.0,SHIP_HEIGHT/2) #FIXME Is this the correct center of mass ?
        newMassData.I = SHIP_INERTIA + SHIP_MASS*(newMassData.center[0]**2+newMassData.center[1]**2) # inertia is defined at origin location not localCenter
        self.ship.massData = newMassData
        
        self.ships.append(self.ship)
        self.drawlist = self.ships + [self.target] + self.rocks
        
        return self.step(2)[0] #FIXME Doesn't that mean we already do one time step ? Expected behavior ?

    def step(self, action):
        done = False
        state = []
        #print('ACTION %d' % action)
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        # implement action
        if action == 0:
            self.thruster_angle += 0.01*60/(self.fps) # FIXME: Put this inside a func + use a macro
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


        ### DEBUG ###
        #print('Step: %d \nShip: %s\nLocals: %s' % (self.stepnumber, self.ship, locals()))
        
        # one step forward
        velocityIterations = 8
        positionIterations = 3
        self.world.Step(1.0 / self.fps, velocityIterations, positionIterations)

        # state construction
        norm_pos = np.max((SEA_W, SEA_H))
        distance_t, bearing_t = getDistanceBearing(self.ship,self.target)
        #print(bearing_t)
        #print(distance_t)
        
        # Normalized ship states
        #state += list(np.asarray(self.ship.GetLocalVector(self.ship.linearVelocity))/Vmax)
        state.append(self.ship.angularVelocity/Rmax)
        state.append(self.thruster_angle / THRUSTER_MAX_ANGLE)
        standardized_dist = (2* distance_t / norm_pos)  - 1
        state.append(standardized_dist) #FIXME Not in [-1,1]
        state.append(bearing_t/np.pi)
        
        for rock in self.rocks:
            distance, bearing = getDistanceBearing(self.ship,rock)
            distance = np.maximum(distance-rock.userData['radius'],0) #FIXME Is this useful ? If ship collides with rock, the engine notifies us right? + We don't take into account the ships geometry.
            rock.userData['distance_to_ship'] = 2 * distance/norm_pos - 1
            rock.userData['bearing_from_ship'] = bearing/np.pi
            rock.userData['in_range'] = True if distance < self.obs_radius else False #FIXME From center of ship center to center of rock. Meaning it wouldn't see very large rocks
        
        # sort rocks from closest to farthest
        self.rocks.sort(key=lambda x:x.userData['distance_to_ship'])
        
        # set 'seen' bool
        #FIXME Could be done in previous loop (a bit more efficient)
        for i in range(self.n_rocks_obs):
            if self.rocks[i].userData['in_range']:
                self.rocks[i].userData['seen']=True 
                state.append(self.rocks[i].userData['distance_to_ship'])
                state.append(self.rocks[i].userData['bearing_from_ship'])
            else: #if closest rocks are outside horizon, fill observation with rocks infinitely far on the ship axis
                self.rocks[i].userData['seen']=False
                state.append(1) #FIXME Maybe don't include them in state instead of choosing arbitrary values
                state.append(0)
        for rock in self.rocks[self.n_rocks_obs:]:
            rock.userData['seen']=False

        #FIXME Separate function
        # REWARD -------------------------------------------------------------------------------------------------------
        self.reward = 0
        #print(distance_t)
        
        if self.ship.userData['hit']:
            if(self.ship.userData['hit_with']=='target'):
                self.reward = +10  #high positive reward. hitting target is good
            else:
                self.reward = -1 #high negative reward. hitting anything else than target is bad
            done = True
        else:   # general case, we're trying to reach target so being close should be rewarded
            self.reward = - ((2* distance_t / norm_pos)  - 1) / MAX_STEPS # FIXME Macro instead of magic number
            #print(self.reward)
        
        # limits episode to MAX_STEPS
        if self.stepnumber >= MAX_STEPS:
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

        print(state)
        
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
            
            thruster = rendering.FilledPolygon(((-THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT),
                                              (-THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT)))
            
            thruster.add_attr(self.thrustertrans) # add thruster angle, assigned later
            thruster.add_attr(self.shiptrans) # add ship angle and ship position, assigned later
            thruster.set_color(*self.ship.color1)
            
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
            horizon.set_color(*self.ship.color1)
            horizon.add_attr(self.shiptrans) # add ship angle and ship position

            self.viewer.add_geom(horizon)

        
        #FIXME Feels pretty hacky, should check on that later
        # Adjusting window
        width_min = min(0,self.ship.position[0]-2*SHIP_HEIGHT)
        width_max = max(SEA_W,self.ship.position[0]+2*SHIP_HEIGHT)
        height_min = min(0,self.ship.position[1]-2*SHIP_HEIGHT)
        height_max = max(SEA_H,self.ship.position[1]+2*SHIP_HEIGHT)
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
            
        self.shiptrans.set_translation(*self.ship.position)
        self.shiptrans.set_rotation(self.ship.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.COGtrans.set_translation(*self.ship.localCenter)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
