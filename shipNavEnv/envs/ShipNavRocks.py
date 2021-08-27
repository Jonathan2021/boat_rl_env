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
    - throttle (if applicable)
    - no action
"""

# gym env class
class ShipNavRocks(gym.Env):
    """
    Env composed of a ship with radar only and rocks as obstacles.
    Also used as a base class for all other subsequent envs.
    """

    #RENDER MACROS
    # visible main window dimensions
    MAIN_WIN_HEIGHT= 720
    MAIN_WIN_WIDTH = 1280

    # visible radar dimensions
    SHIP_VIEW_HEIGHT = 400
    SHIP_VIEW_WIDTH = 400

    # Value used for translation on the screen
    WIN_SHIFT_X = -250
    WIN_SHIFT_Y = 100

    # This is not really rendered (rendered in virtual display).
    SHIP_VIEW_STATE_HEIGHT = 96
    SHIP_VIEW_STATE_WIDTH = 96

    # RL

    FINAL_TRANS_REW = 1 # Base reward for episode termination. You can scale it by a factor in certain scenarios.

    MAX_TIME = 100 # Time in seconds (Could correspond to no more fuel at the end)
    FPS = 30            # simulation framerate
    MAX_STEPS = MAX_TIME * FPS  # max steps for a simulation
    SHIP_STATE_LENGTH = 6 # Length of the state for infos linked to the ship (speed, angle to target etc.)
    WORLD_STATE_LENGTH = 1 # Ratio to episode end
    OBSTACLE_STATE_LENGTH = SHIP_VIEW_STATE_HEIGHT * SHIP_VIEW_STATE_WIDTH * 3 # Length of the radar state
    SINGLE_OBSTACLE_LENGTH = 2 # Length of a single obstacle (for Old radar, so that if it detects 3 obstacles the length of the state will be 3 * 2 = 6)
    
    # Naming of the different state parts
    SPACE_LIDAR = 'lidars'
    SPACE_WORLD = 'world'
    SPACE_OBS = 'obstacles'
    SPACE_SHIP = 'ship' # lol
    SPACE_SHIP_VIEW = 'ship view'


    def __init__(self,**kwargs):
        """ A bunch of config options to be set + initial logic common to all other worlds. """
        
        self._read_kwargs(**kwargs) # Parse options and store them
        self._update_obstacle_state_length() # Update state length according to the possibly new options
           
        self.seed()
        self.world = self._build_world() # Build a world, and what a wonderful world that is.
        
        # inital conditions
        # FIXME: Redundant with reset()
        self.stepnumber = 0
        self.state = []
        self.reward = 0
        self.episode_reward = 0
        self.prev_dist = None # Previous dist to target
        self.obstacles = []

        self.reward_hit = 0
        self.success_rew = 0
        self.time_rew = 0
        self.angle_rew = 0
        self.bumper_rew = 0
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
        self.render_display = ':1' # Display to draw visible renders. Arbitrary because os.environ['DISPLAY'] is not thread safe (will be replaced). Can probably just create a new os var and check if it exists and use it or set it to os.environ ('Display'] or whatever
        if self.ship_view:
            self.virt_disp_hidden =  Display(visible=False, size=(1000, 1000), manage_global_env=False) # Create a virtual display
            self.virt_disp_hidden.start() # Start it
       
        self.reset() # Do reset logic

    def _build_world(self):
        """ Build a world (called by init) """
        return RockOnlyWorld(self.n_rocks, self.scale, {'obs_radius': self.obs_radius}, self.waypoints)

    def _get_space_dict(self):
        """ Get the observation space dictionnary with each key corresponding to an OpenAI observation space"""
        space_dict = dict()
        if self.SHIP_STATE_LENGTH:
            space_dict[self.SPACE_SHIP] = spaces.Box(-1.0, 1.0, shape=(self.SHIP_STATE_LENGTH,), dtype=np.float32)
        if self.WORLD_STATE_LENGTH:
            space_dict[self.SPACE_WORLD] = spaces.Box(-1.0, 1.0, shape=(self.WORLD_STATE_LENGTH,), dtype=np.float32)
        if self.get_obstacles:
            space_dict[self.SPACE_OBS] = spaces.Box(-1.0, 1.0, shape=(self.OBSTACLE_STATE_LENGTH,), dtype=np.float32)
        if self.ship_view:
            space_dict[self.SPACE_SHIP_VIEW] = spaces.Box(0, 255, shape=(3, self.SHIP_VIEW_STATE_WIDTH, self.SHIP_VIEW_STATE_HEIGHT), dtype=np.uint8) # Value in 0, 255 + shape makes Stable Baselines3 detect it is an image.
        return space_dict

    def _get_obs_space(self):
        """ Make an OpenAI spaces.Dict from a dict of spaces """
        return spaces.Dict(self._get_space_dict())
    
    def _get_ship_kwargs(self):
        """ make kwargs dict with ship specifics """
        return {
                'obs_radius': self.obs_radius,
                }

    # Dict of default kwargs
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
        """ Read kwargs and replace default with the value """
        for key, default in self.possible_kwargs.items():
            setattr(self, key, kwargs.get(key, default))
        
        self.get_obstacles = self.n_obstacles_obs > 0 and self.SINGLE_OBSTACLE_LENGTH > 0 # Do we get_obstacles (Old radar) ?

    def _update_obstacle_state_length(self):
        """
        Adjust the state length for obstacles (Old radar)
        """
        self.OBSTACLE_STATE_LENGTH = 0 
        self.OBSTACLE_STATE_LENGTH += self.SINGLE_OBSTACLE_LENGTH * self.n_obstacles_obs

    def seed(self, seed=None):
        """ Set the seeding """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # TODO Not used anywhere now so either remove it or keep it since it can be an indicator
    def _adjust_times(self):
        """ Time estimates that can be useful """
        self.QUICKEST_TIME_SHOULD_TAKE = self.world.get_ship_objective_dist() / self.world.ship.Vmax # + Calculate somehow how long it takes to turn to bearing 0 ? Even if NN could learn that by itself
        self.QUICKEST_TIME_SHOULD_TAKE_STEPS = self.QUICKEST_TIME_SHOULD_TAKE* self.fps

    def reset(self):
        """ Reset the env to factory new but keeping the parameters """
        # Clear geoms (but keep the same viewer)
        if self.main_viewer:
            self.main_viewer.geoms = []
        if self.ship_viewer:
            self.ship_viewer.geoms = []
        if self.ship_state_viewer:
            self.ship_state_viewer.geoms = []

        # Call world reset and set all variables to original values
        self.world.reset()
        self.obstacles = []

        self._adjust_times() # Adjust times since the world is not the same

        self.stepnumber = 0
        self.episode_reward = 0
        self.original_dist = self.world.get_ship_target_dist()
        self.prev_dist = self.original_dist
        self.is_success = False

        self.reward_hit = 0
        self.success_rew = 0
        self.time_rew = 0
        self.angle_rew = 0
        self.bumper_rew = 0
        self.dist_rew = 0
        self.reward_max_time = 0

        # Rendering
        self.view_state_rendered_once = False
        self.rendered_once = False

        return self._get_state() # Return the 1st state without stepping

    def _take_actions(self, action):
        """ Apply the given actions. Here you can only steer """
        ship = self.world.ship
        
        
        if action is not None:
            if action == 0:
                ship.steer(1, self.fps) # Turn right
            elif action == 1:
                ship.steer(-1, self.fps) # Turn left

    def _get_obstacles(self):
        """ Get obstacle (wrapper) for old radar """
        return self._get_obstacles_conservative()

    def _get_obstacles_conservative(self):
        """ Conservative way of selecting obstacles in the n_obstacles limit such that they keep their place in the list if they are still seen """
        obstacles = self.world.get_obstacles()
        
        # sort rocks by seen and from closest to farthest
        obstacles.sort(key=lambda x: (0 if x.seen else 1, x.distance_to_ship))

        new_seen = []
        for obs in obstacles:
            if obs.seen and obs not in self.obstacles: # Obstacle is seen and isn't in previously observed obstacles
                new_seen.append(obs)
            
        new_obstacles = []
        for obs in self.obstacles: #Iterate on previously selected obstacles (during last step)
            if not obs.seen and new_seen: # If old obstacle seen is not seen anymore, and some new available
                new_obstacles.append(new_seen[0]) #Add closest one according to previous sort
                del new_seen[0]
            else:
                # If still valid or
                # if no new to replace, keep the old one (it's seen value should have been reset and will be handled in the function getting state)
                new_obstacles.append(obs)
            
        new_obstacles += new_seen # If still some new seen left, add them at the tail
        self.obstacles = new_obstacles

        for obs in self.obstacles[self.n_obstacles_obs:]: # see only limited amount of obstacles
            obs.seen = False # FIXME call unsee method ?

        self.obstacles = self.obstacles[:self.n_obstacles_obs] # keep track for next iteration of that limited amount only
        
        return self.obstacles

    def _get_ship_state(self):
        """ Ship related state, normalized to be in [-1, 1] """
        ship = self.world.ship

        state = []
        velocity_x, velocity_y = ship.body.GetLocalVector(ship.body.linearVelocity)
        state.append(velocity_x / ship.VmaxX) # x axis speed
        state.append(velocity_y / ship.VmaxY) # y axis speed
        state.append(ship.body.angularVelocity/ship.Rmax) # angular velocity
        state.append(ship.thruster_angle / ship.THRUSTER_MAX_ANGLE) # thruster angle
        state.append(self.world.get_ship_objective_standard_dist()) # Distance to target
        state.append(self.world.get_ship_objective_standard_bearing()) # Bearing to target
        return state        

    def _get_world_state(self):
        """ World related state, normalized to be in [-1, 1] """
        return [2 * self.stepnumber / self.MAX_STEPS - 1] # TODO Could represent fuel consumption, so could be moved to ship state
    
    def _get_obstacle_state(self):
        """ Obstacle state (old radar) construction """
        ship = self.world.ship
        obstacles = self._get_obstacles() # Get the selected obstacles

        state = []
        
        for i in range(self.n_obstacles_obs):
            if i < len(obstacles) and obstacles[i].seen: # If it was a selected obstacle and is seen
                state.append(np.clip(2 * self.world.get_ship_dist(obstacles[i]) / ship.obs_radius - 1, -1, 1)) # Add its distance
                state.append(np.clip(self.world.get_ship_standard_bearing(obstacles[i]), -1, 1)) # add its bearing
            else: # if not, fill in
                state.append(1) # 1 = far away
                state.append(np.random.uniform(-1, 1)) # Random bearing
        return state
    
    def _get_ship_view_state(self):
        """ New radar state (image) """
        if not self.ship_state_viewer: # Create the viewer if it doesn't already exist
            self.ship_state_viewer = rendering.Viewer(self.SHIP_VIEW_STATE_WIDTH, self.SHIP_VIEW_STATE_HEIGHT, display=self.virt_disp_hidden.new_display_var) # Set the display to the invisible virtual display
        self.world.render_ship_view(self.ship_state_viewer, not self.view_state_rendered_once, self.stepnumber % self.fps == 0) # Add the geoms
        self.view_state_rendered_once = True
        pixels = self.ship_state_viewer.render(return_rgb_array = True) # Draw image on virtual display
        pixels = np.moveaxis(pixels, -1, 0) # make image channel first (expected by Stable Baselines3
        return pixels

    def _get_state(self):
        """ Fill in space dict with the various state parts """
        state = dict()
        state[self.SPACE_SHIP] = self._get_ship_state()
        state[self.SPACE_WORLD] = self._get_world_state()
        if self.get_obstacles: # We want old radar ?
            state[self.SPACE_OBS] = self._get_obstacle_state()
        if self.ship_view: # We want new radar ?
            state[self.SPACE_SHIP_VIEW] = self._get_ship_view_state()
        return state

    # Not in use anymore because it can dissuade the agent from avoiding hard path by moving away from the objective
    def _dist_reward(self):
        """ Reward based on distance difference between 2 steps to target. Getting closer -> positive and getting further -> negative """
        return self.world.delta_dist / (self.world.ship.Vmax / self.FPS) * 2 * abs(self._timestep_reward())  # Helps to make agent know it should go to objective (not redundant of timestep-wise neg reward because suppose agent knows it is too far to get there in time, at least it'll try to get closer) (normalized by max dist should be able to take in a step). Proportional to timestep reward because I want going quite straight to objective to yield positive reward.


    def _timestep_reward(self):
        """ Small neg reward every timestep (can correspond to a fuel consumption penalty) so that the agent hurries on """
        reward = - 1 / (self.MAX_STEPS) # If episode goes to max time, then total sum is -1
        return reward
    
    def _get_thruster_angle_reward(self):
        """ Penalty on using the thruster. We penalize angle and not moving it because it is preferable (COLREGs say so)
        to move early and perhaps quite a bit but go straight afterwards instead of doing a big curve over time with a small angle of the thruster.""" 
        penalty = -abs(self.world.ship.thruster_angle / self.world.ship.THRUSTER_MAX_ANGLE) / self.MAX_STEPS # Punish not going straight, such that if max angle for the whole episode, then gets -1 cumulative reward
        if self.world.ship.thruster_angle > 0:
            penalty /= 2 # Encourage dodging to starboard side (kinda hacky but couldn't think of another way)
        return penalty

    def _hit_reward(self):
        """ Penalty for hitting something """
        ship = self.world.ship
        done = False
        reward = 0
        if ship.is_hit() and not self.world.target in ship.hit_with: # FIXME Now that target is a sensor, it shouldn't be in hit_with so you can probably remove this
            reward -= 5 * self.FINAL_TRANS_REW #high negative reward. Scaled by a factor compared to other final transition rewards because it seems like it tend to take risks of going close to ships if I didn't scale.
            done = True
        return reward, done

    def _max_time_reward(self):
        """ Penalty for not finishing before the max number of preselected timesteps """
        reward = 0
        done = False
        # limits episode to self.MAX_STEPS
        if self.stepnumber >= self.MAX_STEPS:
            reward -= self.FINAL_TRANS_REW
            done = True
        return reward, done

    def _bumper_reward(self):
        """ Reward for touching stuff with the bumper around the ship (this has vocation of encouragin choosing safe routes). """
        n_touches = len(self.world.ship.bumper_state(ignore=[self.world.target.body]))
        return np.sqrt(n_touches) * -1 / self.MAX_STEPS # kinda hacky but if touches something then >= timestep reward and if touches more then punishes more without penalizing too much envs satured in obstacles where it is impossible to not touch anything.
        # TODO there is most certainly a better way to achieve this objective.
    
    def _get_reward_done(self):
        """ Sum up all the different reward parts and check for episode termination """
        reward = 0
        done = False

        #dist_reward = self._dist_reward()
        #self.dist_rew += dist_reward

        time_reward = self._timestep_reward()
        self.time_rew += time_reward
        reward += time_reward

        angle_rew = self._get_thruster_angle_reward()
        self.angle_rew += angle_rew
        reward += angle_rew

        bumper_rew = self._bumper_reward()
        self.bumper_rew += bumper_rew
        reward += bumper_rew

        reward_hit, done = self._hit_reward()
        self.reward_hit += reward_hit
        reward += reward_hit

        reward_max_time, done_max_time = self._max_time_reward()
        self.reward_max_time += reward_max_time
        reward += reward_max_time

        done = done or done_max_time

        self.is_success = self.world.is_success()
        if self.is_success: # We reached target
            self.success_rew = self.FINAL_TRANS_REW
            reward += self.success_rew
        done = done or self.is_success

        return reward, done

    def step(self, action):
        """ Take a step by applying the given action and get the next state, reward associated with the transition, is episode done ? and additional info """
        self._take_actions(action) # Take action

        addDotTraj = self.display_traj and (self.stepnumber % np.ceil(self.display_traj_T*self.fps) == 0) # Do we add dot ?
        self.world.step(self.fps, update_obstacles=self.get_obstacles, addDotTraj=addDotTraj) # Step the world simulation
        self.stepnumber += 1
        
        self.state = self._get_state() # Get the state after stepping (St+1)
        
        self.reward, done = self._get_reward_done() # Get the transition reward
        self.episode_reward += self.reward
            
        # Record additional info
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
        """ Render the env visually """
        if close: # Close viewers
            if self.main_viewer is not None:
                self.main_viewer.close()
                self.main_viewer = None
            if self.ship_view and self.ship_viewer is not None:
                self.ship_viewer.close()
                self.ship_viewer = None
            return

        if not self.main_viewer: # If main visible viewer wasn't created, do it
            self.main_viewer = rendering.Viewer(self.MAIN_WIN_WIDTH, self.MAIN_WIN_HEIGHT, display=self.render_display)
            # Shift its position on screen
            win_x, win_y = self.main_viewer.window.get_location()
            self.main_viewer.window.set_location(win_x + self.WIN_SHIFT_X, win_y + self.WIN_SHIFT_Y)

        if self.ship_view and not self.ship_viewer: # We do want the radar image and visible window is not created yet
            self.ship_viewer = rendering.Viewer(self.SHIP_VIEW_WIDTH, self.SHIP_VIEW_HEIGHT, display=self.render_display)
            win_x, win_y = self.ship_viewer.window.get_location()
            self.ship_viewer.window.set_location(
                win_x + (self.MAIN_WIN_WIDTH + self.SHIP_VIEW_WIDTH)//2 + self.WIN_SHIFT_X,
                win_y - (self.MAIN_WIN_HEIGHT + self.SHIP_VIEW_HEIGHT)//2 + self.WIN_SHIFT_Y + 200)

        all_close = True
        if self.ship_view: # Render radar image
            self.world.render_ship_view(self.ship_viewer, not self.rendered_once, False)
            all_close = self.ship_viewer.render(return_rgb_array = mode == 'rgb_array')

        # Render main window
        self.world.render(self.main_viewer, not self.rendered_once)
        all_close = self.main_viewer.render(return_rgb_array = mode == 'rgb_array') and all_close

        self.rendered_once = True

        return all_close # Were the windows all closed ?

    def close(self):
        """ Close all windows, including hidden ones """
        if self.ship_view:
            if self.ship_state_viewer:
                self.ship_state_viewer.close()
            self.virt_disp_hidden.stop() # Stop the virtual display
            if self.ship_viewer:
                self.ship_viewer.close()
        if self.main_viewer:
            self.main_viewer.close()

# Never trained on yet
class ShipNavRocksContinuousSteer(ShipNavRocks):
    """ Env with rocks only but steering is continuous instead of discrete """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32) # Change action space to expect continuous values

    def _take_actions(self, action):
        if action is not None:
            self.world.ship.steer(action[0], self.fps) # Steer the requested amount

# Never used yet
class ShipNavRocksSteerAndThrustDiscrete(ShipNavRocks):
    """ Env with rocks only but can steer and throttle with discrete steps """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.MultiDiscrete([3, 3]) # steer left, right, nothing + throttle less, more, nothing

    def _take_actions(self, action):
        if action is not None:
            super()._take_actions(self, action[0]) # steer

            # Throttle
            if action[1] == 0:
                self.world.ship.thrust(-1, self.fps)
            elif action[1] == 1:
                self.world.ship.thrust(1, self.fps)

# Never used yet
class ShipNavRocksSteerAndThrustContinuous(ShipNavRocks):
    """ Env with rocks only and steer / throttle continuous """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer right or left

    def _take_actions(self, action):
        if action is not None:
            self.world.ship.steer(action[0], self.fps)
            self.world.ship.thrust(action[1], self.fps)



class ShipNavRocksLidar(ShipNavRocks):
    """ Env with rocks only and discrete steer only but with lasers around ship """
    SINGLE_OBSTACLE_LENGTH = 0 # Trick not to observe obstacles even if in kwargs
    possible_kwargs = ShipNavRocks.possible_kwargs.copy()
    possible_kwargs.update({'n_lidars': 15, 'obs_radius': 0, 'ship_view': False, 'n_obstacles_obs': 0})

    def _build_world(self):
        return RockOnlyWorldLidar(self.n_rocks, self.n_lidars, self.scale, self._get_ship_kwargs(), waypoint_support=self.waypoints)

    def _get_lidar_state(self):
        """ Lidar state which corresponds to the fraction of the original length scaled and shifted to be in [-1, 1] """
        return [2 * l.fraction - 1 for l in self.world.ship.lidars] 

    def _get_space_dict(self):
        """ Space dict with lidars added """
        space_dict = ShipNavRocks._get_space_dict(self)
        if self.n_lidars:
            space_dict[self.SPACE_LIDAR] = spaces.Box(-1.0, 1.0, shape=(self.n_lidars,), dtype=np.float32)
        return space_dict
    
    def _get_state(self):
        """ Getting state parts with lidars in addition """
        state = ShipNavRocks._get_state(self)
        state[self.SPACE_LIDAR] = self._get_lidar_state()
        return state
