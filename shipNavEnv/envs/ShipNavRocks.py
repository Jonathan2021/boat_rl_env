#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:19:52 2020

@author: Gilles Foinet
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from shipNavEnv.Worlds import RockOnlyWorld, RockOnlyWorldLidar, World
from pyvirtualdisplay import Display
from gym.envs.classic_control import rendering
from Box2D import b2CircleShape
import os

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
    #RENDER
    MAIN_WIN_HEIGHT= 720
    MAIN_WIN_WIDTH = 1280
    SHIP_VIEW_HEIGHT = 400
    SHIP_VIEW_WIDTH = 400
    WIN_SHIFT_X = -250
    WIN_SHIFT_Y = 100
    SHIP_VIEW_STATE_HEIGHT = 96
    SHIP_VIEW_STATE_WIDTH = 96

    # RL
    FINAL_TRANS_REW = 1

    MAX_TIME = 100 # No more fuel at the end
    FPS = 30            # simulation framerate
    MAX_STEPS = MAX_TIME * FPS  # max steps for a simulation
    SHIP_STATE_LENGTH = 6
    WORLD_STATE_LENGTH = 1
    OBSTACLE_STATE_LENGTH = SHIP_VIEW_STATE_HEIGHT * SHIP_VIEW_STATE_WIDTH * 3
    SINGLE_OBSTACLE_LENGTH = 2
    
    #Naming
    SPACE_LIDAR = 'lidars'
    SPACE_WORLD = 'world'
    SPACE_OBS = 'obstacles'
    SPACE_SHIP = 'ship' # lol
    SPACE_SHIP_VIEW = 'ship view'


    def __init__(self,**kwargs):
        
        # Configuration
        # FIXME: Should move kwargs access in some configure function, can keep it in dict form (with defaults) and then iterate on keys
        #FIXME: Defaults should be in var
        self._read_kwargs(**kwargs)
        self._update_obstacle_state_length()
           
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
        self.prev_dist = None
        self.obstacles = []

        self.reward_hit = 0
        self.success_rew = 0
        self.time_rew = 0
        self.angle_rew = 0
        self.dist_rew = 0
        self.reward_max_time = 0
        
        # Observation are continuous in [-1, 1] 
        self.observation_space = self._get_obs_space()
        
        # Left or Right (or nothing)
        self.action_space = spaces.Discrete(3)


        # Rendering
        self.main_viewer = None
        self.ship_viewer = None
        self.ship_state_viewer = None
        self.render_display = ':1' # os.environ['DISPLAY'] <- Not thread safe
        if self.ship_view:
            #os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
            self.virt_disp_hidden =  Display(visible=False, size=(1000, 1000), manage_global_env=False)
            self.virt_disp_hidden.start()
       
        self.reset()

    def _build_world(self):
        return RockOnlyWorld(self.n_rocks, self.scale, {'obs_radius': self.obs_radius}, self.waypoints)

    def _get_space_dict(self):
        space_dict = dict()
        if self.SHIP_STATE_LENGTH:
            space_dict[self.SPACE_SHIP] = spaces.Box(-1.0, 1.0, shape=(self.SHIP_STATE_LENGTH,), dtype=np.float32)
        if self.WORLD_STATE_LENGTH:
            space_dict[self.SPACE_WORLD] = spaces.Box(-1.0, 1.0, shape=(self.WORLD_STATE_LENGTH,), dtype=np.float32)
        if self.get_obstacles:
            space_dict[self.SPACE_OBS] = spaces.Box(-1.0, 1.0, shape=(self.OBSTACLE_STATE_LENGTH,), dtype=np.float32)
        if self.ship_view:
            space_dict[self.SPACE_SHIP_VIEW] = spaces.Box(0, 255, shape=(3, self.SHIP_VIEW_STATE_WIDTH, self.SHIP_VIEW_STATE_HEIGHT), dtype=np.uint8)
        return space_dict

    def _get_obs_space(self):
        return spaces.Dict(self._get_space_dict())
    
    def _get_ship_kwargs(self):
        return {
                'obs_radius': self.obs_radius,
                }

    possible_kwargs = {
            'n_rocks': 40,
            'scale': RockOnlyWorld.ROCK_SCALE_DEFAULT,
            'ship_view': True,
            'n_obstacles_obs': 0,
            'obs_radius': 200,
            'waypoints': True,
            'fps': FPS,
            'display_traj': False,
            'display_traj_T': 0.1
            }

    def _read_kwargs(self, **kwargs):
        for key, default in self.possible_kwargs.items():
            setattr(self, key, kwargs.get(key, default))
        
        self.get_obstacles = self.n_obstacles_obs > 0 and self.SINGLE_OBSTACLE_LENGTH > 0

    def _update_obstacle_state_length(self):
        self.OBSTACLE_STATE_LENGTH = 0 
        #if self.ship_view:
        #    self.OBSTACLE_STATE_LENGTH += self.SHIP_VIEW_STATE_HEIGHT * self.SHIP_VIEW_STATE_WIDTH * 3
        self.OBSTACLE_STATE_LENGTH += self.SINGLE_OBSTACLE_LENGTH * self.n_obstacles_obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _adjust_times(self):
        self.QUICKEST_TIME_SHOULD_TAKE = self.world.get_ship_objective_dist() / self.world.ship.Vmax # + Calculate somehow how long it takes to turn to bearing 0 ? Even if NN could learn that by itself
        self.QUICKEST_TIME_SHOULD_TAKE_STEPS = self.QUICKEST_TIME_SHOULD_TAKE* self.fps
        #print("I should take %f" % self.QUICKEST_TIME_SHOULD_TAKE)

    def reset(self):
        if self.main_viewer:
            self.main_viewer.geoms = []
        if self.ship_viewer:
            self.ship_viewer.geoms = []
        if self.ship_state_viewer:
            self.ship_state_viewer.geoms = []

        self.world.reset()
        self.obstacles = []

        self._adjust_times()

        #print(self.QUICKEST_TIME_SHOULD_TAKE)

        self.stepnumber = 0
        self.episode_reward = 0
        self.original_dist = self.world.get_ship_target_dist()
        self.prev_dist = self.original_dist
        self.is_success = False

        self.reward_hit = 0
        self.success_rew = 0
        self.time_rew = 0
        self.angle_rew = 0
        self.dist_rew = 0
        self.reward_max_time = 0

        # Rendering
        self.view_state_rendered_once = False
        self.rendered_once = False

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
        #print([obstacle.seen for obstacle in obstacles])

        new_seen = []
        for obs in obstacles:
            if obs.seen and obs not in self.obstacles: # Obstacle is seen and isn't in previously observed obstacles
                new_seen.append(obs)
            
        new_obstacles = []
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
        velocity_x, velocity_y = ship.body.GetLocalVector(ship.body.linearVelocity)
        state.append(velocity_x / ship.VmaxX)
        state.append(velocity_y / ship.VmaxY)
        state.append(ship.body.angularVelocity/ship.Rmax)
        state.append(ship.thruster_angle / ship.THRUSTER_MAX_ANGLE)
        state.append(self.world.get_ship_objective_standard_dist()) # 
        state.append(self.world.get_ship_objective_standard_bearing())
        return state        

    def _get_world_state(self):
        return [2 * self.stepnumber / self.MAX_STEPS - 1] #, 2*self.QUICKEST_TIME_SHOULD_TAKE/ self.MAX_TIME - 1]
    
    def _get_obstacle_state(self):
        ship = self.world.ship
        obstacles = self._get_obstacles()

        state = []
        
        for i in range(self.n_obstacles_obs):
            if i < len(obstacles) and obstacles[i].seen:
                state.append(np.clip(2 * self.world.get_ship_dist(obstacles[i]) / ship.obs_radius - 1, -1, 1))
                state.append(np.clip(self.world.get_ship_standard_bearing(obstacles[i]), -1, 1))
            else:
                state.append(1)
                state.append(np.random.uniform(-1, 1))
        return state
    
    def _get_ship_view_state(self):
        if not self.ship_state_viewer:
            self.ship_state_viewer = rendering.Viewer(self.SHIP_VIEW_STATE_WIDTH, self.SHIP_VIEW_STATE_HEIGHT, display=self.virt_disp_hidden.new_display_var)
            #self.ship_state_viewer = InvisibleViewer(self.SHIP_VIEW_WIDTH, self.SHIP_VIEW_HEIGHT)
        self.world.render_ship_view(self.ship_state_viewer, not self.view_state_rendered_once)
        self.view_state_rendered_once = True
        pixels = self.ship_state_viewer.render(return_rgb_array = True)
        pixels = np.moveaxis(pixels, -1, 0)
        return pixels

    def _get_state(self):
        state = dict()
        state[self.SPACE_SHIP] = self._get_ship_state()
        state[self.SPACE_WORLD] = self._get_world_state()
        if self.get_obstacles:
            state[self.SPACE_OBS] = self._get_obstacle_state()
        if self.ship_view:
            state[self.SPACE_SHIP_VIEW] = self._get_ship_view_state()
        return state

    def _dist_reward(self):
        return self.world.delta_dist / (self.world.ship.Vmax / self.FPS) * 2 * abs(self._timestep_reward())  # Helps to make agent know it should go to objective (not redundant of timestep-wise neg reward because suppose agent knows it is too far to get there in time, at least it'll try to get closer) (normalized by max dist should be able to take in a step). Proportional to timestep reward because I want going quite straight to objective to yield positive reward (So that it doesn't do straight into a rock to end the suffering)
        #return self.world.delta_dist / self.original_dist * 100 # we're trying to reach target so being close should be rewarded

    def _timestep_reward(self):
        reward = - 1 / (self.MAX_STEPS) # Could be analogous to gas left, normalized to be -100 at the end. (ran out of gas)
        #if self.stepnumber > 2 * self.QUICKEST_TIME_SHOULD_TAKE_STEPS:
        #    reward -= 0.1
        return reward
    
    def _get_thruster_angle_reward(self):
        penalty = -abs(self.world.ship.thruster_angle / self.world.ship.THRUSTER_MAX_ANGLE) / self.MAX_TIME # Punish not going straight, such that if max angle for the whole episode, then gets -1 cumulative reward
        #if self.world.ship.thruster > 
        if self.world.ship.thruster_angle > 0:
            penalty /= 2 # Encourage dodging to starboard side (kinda hacky but couldn't think of another way)
        return penalty
        #return -abs(self.delta_thruster_angle) / self.world.ship.THRUSTER_MAX_ANGLE_STEP / self.MAX_TIME # Such as, if max step possible at a given fps for every step on max_steps -> then -1

    def _hit_reward(self):
        ship = self.world.ship
        done = False
        reward = 0
        if ship.is_hit() and not self.world.target in ship.hit_with: # FIXME Will not know if hit is new or not !
            reward -= self.FINAL_TRANS_REW #high negative reward. hitting anything else than target is bad
            #Removed the neg reward to experiment
            done = True
        return reward, done

    def _max_time_reward(self):
        reward = 0
        done = False
        # limits episode to self.MAX_STEPS
        if self.stepnumber >= self.MAX_STEPS:
            reward -= self.FINAL_TRANS_REW # same as rock
            done = True
        return reward, done
    
    def _get_reward_done(self):
        reward = 0
        done = False
        #dist_reward = self._dist_reward()
        #self.dist_rew += dist_reward
        #reward += dist_reward
        #print("Dist reward %.8f" % dist_reward)
        time_reward = self._timestep_reward()
        self.time_rew += time_reward
        reward += time_reward
        #print("Time reward %.8f" % time_reward)
        angle_rew = self._get_thruster_angle_reward()
        self.angle_rew += angle_rew
        #print("Angle rew %.8f" % angle_rew)
        reward += angle_rew
        reward_hit, done = self._hit_reward()
        self.reward_hit += reward_hit
        #print("Hit reward %.8f" % reward_hit)
        reward += reward_hit
        reward_max_time, done_max_time = self._max_time_reward()
        self.reward_max_time += reward_max_time
        #print("Max time reward %.8f" % reward_max_time)
        reward += reward_max_time
        done = done or done_max_time

        self.is_success = self.world.is_success()
        if self.is_success:
            self.success_rew = self.FINAL_TRANS_REW
            reward += self.success_rew
        done = done or self.is_success
        #print("Total reward %.8f" % reward)
        #if done:
            #if self.stepnumber < 3:
            #    print("Distance %s" % self.original_dist)
            #    self.render()
            #    import time
            #    time.sleep(15)
            #print("Hit rew %d" % self.reward_hit)
            #print("Success rew %d" % self.success_rew)
            #print("time_rew %f" % self.time_rew)
            #print("dist_rew %f" % self.dist_rew)
            #print("rew max time %d" % self.reward_max_time)
        return reward, done

    def step(self, action):
        prev_angle = self.world.ship.thruster_angle
        self._take_actions(action)
        addDotTraj = self.display_traj and (self.stepnumber % np.ceil(self.display_traj_T*self.fps) == 0)
        self.delta_thruster_angle = self.world.ship.thruster_angle - prev_angle
        self.world.step(self.fps, update_obstacles=self.get_obstacles, addDotTraj=addDotTraj)
        self.stepnumber += 1
        
        self.state = self._get_state()
        #for sens in self.world.ship.sensors:
        #    print(sens.userData['touching'])
        #    if type(sens.shape) is b2CircleShape:
        #        print(sens.shape.pos)
        #print(self.world.ship.bumper_state())#ignore=[self.world.target.body]))

        #if self.stepnumber == 10:
        #    for rock in self.world.rocks:
        #        print(rock.body.position)

        #if self.stepnumber == 1:
        #    print(self.state.keys())
        #    print(self.observation_space)
        #    print(self.OBSTACLE_STATE_LENGTH)

        #if self.stepnumber == 1:
        #    print(self.world.render_ship_view(mode='rgb_array'))
        #print(ship.body.linearVelocity)
        #print(ship.body.angularVelocity)
        #print(ship.body.GetLocalVector(ship.body.linearVelocity))
        #print(np.sqrt(ship.body.linearVelocity[0]**2 + ship.body.linearVelocity[1]**2))
        #if not (self.stepnumber % self.FPS):
        #print(self.state)
        #print(self.world.get_ship_target_bearing())

        # Normalized ship states
        #state += list(np.asarray(self.ship.body.GetLocalVector(self.ship.body.linearVelocity))/Ship.Vmax)
       
        
        #FIXME Separate function
        
        self.reward, done = self._get_reward_done()
        self.episode_reward += self.reward
            
       # print(state)
        # if done: 
        #     print("Returning reward %d" % self.reward)
        info = {"is_success": self.is_success};
        mainShipPosEast = np.double(self.world.ship.body.position[0])
        mainShipPosNorth = np.double(self.world.ship.body.position[1])
        mainShipHeading = (self.world.ship.body.angle + np.pi) % (2 * np.pi) - np.pi
        info['ship_0'] = (mainShipPosEast,mainShipPosNorth,mainShipHeading)
        
        for i,ship in enumerate(self.world.ships):
            shipPosEast = np.double(ship.body.position[0])
            shipPosNorth = np.double(ship.body.position[1])
            shipHeading = (ship.body.angle + np.pi) % (2 * np.pi) - np.pi
            info['ship_{0}'.format(i+1)] = (shipPosEast,shipPosNorth,shipHeading)
        return self.state, self.reward, done, info

    def render(self, mode='human', close=False):
        if close:
            if self.main_viewer is not None:
                self.main_viewer.close()
                self.main_viewer = None
            if self.ship_view and self.ship_viewer is not None:
                self.ship_viewer.close()
                self.ship_viewer = None
            return

        if not self.main_viewer:
            self.main_viewer = rendering.Viewer(self.MAIN_WIN_WIDTH, self.MAIN_WIN_HEIGHT, display=self.render_display)
            win_x, win_y = self.main_viewer.window.get_location()
            self.main_viewer.window.set_location(win_x + self.WIN_SHIFT_X, win_y + self.WIN_SHIFT_Y)

        if self.ship_view and not self.ship_viewer:
            self.ship_viewer = rendering.Viewer(self.SHIP_VIEW_WIDTH, self.SHIP_VIEW_HEIGHT, display=self.render_display)
            win_x, win_y = self.ship_viewer.window.get_location()
            self.ship_viewer.window.set_location(
                win_x + (self.MAIN_WIN_WIDTH + self.SHIP_VIEW_WIDTH)//2 + self.WIN_SHIFT_X,
                win_y - (self.MAIN_WIN_HEIGHT + self.SHIP_VIEW_HEIGHT)//2 + self.WIN_SHIFT_Y + 200)

        #self.main_viewer.set_bounds("b", "a", "d", "e")

        #print([d.userData for d in self.drawlist])
        all_close = True
        if self.ship_view:
            self.world.render_ship_view(self.ship_viewer, not self.rendered_once)
            all_close = self.ship_viewer.render(return_rgb_array = mode == 'rgb_array')

        self.world.render(self.main_viewer, not self.rendered_once)
        all_close = self.main_viewer.render(return_rgb_array = mode == 'rgb_array') and all_close

        self.rendered_once = True

        return all_close
        #else:
        #    return self.world.render_ship_view(mode, close)

    def close(self):
        if self.ship_view:
            if self.ship_state_viewer:
                self.ship_state_viewer.close()
            self.virt_disp_hidden.stop()
            if self.ship_viewer:
                self.ship_viewer.close()
        if self.main_viewer:
            self.main_viewer.close()

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

class ShipNavRocksSteerAndThrustContinuous(ShipNavRocks):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer right or left

    def _take_actions(self, action):
        if action is not None:
            self.world.ship.steer(action[0], self.fps)
            self.world.ship.thrust(action[1], self.fps)



class ShipNavRocksLidar(ShipNavRocks):
    SINGLE_OBSTACLE_LENGTH = 0 # Trick not to observe obstacles even if in kwargs
    possible_kwargs = ShipNavRocks.possible_kwargs.copy()
    possible_kwargs.update({'n_lidars': 15, 'obs_radius': 0, 'ship_view': False, 'n_obstacles_obs': 0})

    def _build_world(self):
        return RockOnlyWorldLidar(self.n_rocks, self.n_lidars, self.scale, self._get_ship_kwargs(), waypoint_support=self.waypoints)

        return spaces.Box(-1.0,1.0,shape=(self.SHIP_STATE_LENGTH + self.WORLD_STATE_LENGTH + self.n_lidars + self.OBSTACLE_STATE_LENGTH,), dtype=np.float32)

    def _get_lidar_state(self):
     return [2 * l.fraction - 1 for l in self.world.ship.lidars] 

    def _get_space_dict(self):
        space_dict = ShipNavRocks._get_space_dict(self)
        if self.n_lidars:
            space_dict[self.SPACE_LIDAR] = spaces.Box(-1.0, 1.0, shape=(self.n_lidars,), dtype=np.float32)
        return space_dict
    
    def _get_state(self):
        state = ShipNavRocks._get_state(self)
        state[self.SPACE_LIDAR] = self._get_lidar_state()
        return state
