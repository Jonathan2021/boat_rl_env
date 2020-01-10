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

START_HEIGHT = 100.0
START_SPEED = 1.0

# THRUSTER
THRUSTER_MIN_THROTTLE = 0.4 # [%]
THRUSTER_MAX_ANGLE = 0.4    # [rad]
THRUSTER_MAX_FORCE = 1e5    # [N]


THRUSTER_HEIGHT = 2  # [m]
THRUSTER_WIDTH = 0.8 # [m]

# SHIP
SHIP_HEIGHT = 20    # [m]
SHIP_WIDTH = 2      # [m]

# VIEWPORT
VIEWPORT_H = 1800    # [pixels]
VIEWPORT_W = 3200   # [pixels]

SCALE = 1 # [m/pixel]

# ROCK
ROCK_RADIUS = 20

#LIDAR
LIDAR_RANGE   = 200
LIDAR_NB_RAY = 8;


class ShipNavigationLidarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        print('init2')
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World(gravity=(0,0))
        self.thruster = None
        self.ship = None
        self.target = None
        self.lidar = None
        self.throttle = 0
        self.thruster_angle = 0.0
        
        high = np.ones(2+LIDAR_NB_RAY, dtype=np.float32)
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
        self.ship = None

    def reset(self):
        print('reset env')
        self._destroy()
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.rudder_angle = 0.0
        self.stepnumber = 0

        initial_x = np.random.uniform( 2*SHIP_HEIGHT, VIEWPORT_W-2*SHIP_HEIGHT)
        initial_y = np.random.uniform( 2*SHIP_HEIGHT, VIEWPORT_H-2*SHIP_HEIGHT)
        initial_heading = np.random.uniform(0, math.pi)
        
        targetX = np.random.uniform( 10*SHIP_HEIGHT, VIEWPORT_W-10*SHIP_HEIGHT)
        targetY = np.random.uniform( 10*SHIP_HEIGHT, VIEWPORT_H-10*SHIP_HEIGHT)
        
        
        self.target = self.world.CreateStaticBody(
                position = (targetX,targetY),
                angle = 0.0,
                fixtures = fixtureDef(
                        shape = polygonShape(vertices = ((ROCK_RADIUS,0),
                                                    (0,ROCK_RADIUS),
                                                    (-ROCK_RADIUS,0),
                                                    (0,-ROCK_RADIUS)))                                        
                                        ))
        self.target.color1 = rgb(255,0,0)
        
        self.ship = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=initial_heading,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-SHIP_WIDTH / 2, 0),
                                             (+SHIP_WIDTH / 2, 0),
                                             (SHIP_WIDTH / 2, +SHIP_HEIGHT),
                                             (0, +SHIP_HEIGHT*1.2),
                                             (-SHIP_WIDTH / 2, +SHIP_HEIGHT))),
                density=1.0,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0),
                linearDamping=5,
                angularDamping=20,
        )

        # self.lander.angularDamping = 0.9

        self.ship.color1 = rgb(230, 230, 230)
        self.ship.linearVelocity = (0.0,0.0)
        self.ship.angularVelocity = 0

        self.drawlist = [self.ship,self.target]
        
        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(LIDAR_NB_RAY)]
        
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
        force = (-np.sin(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE,
                 np.cos(self.ship.angle + self.thruster_angle) * THRUSTER_MAX_FORCE )
        self.ship.ApplyForce(force=force, point=force_pos, wake=False)
        self.world.Step(1.0 / FPS, 60, 60)
        
        pos = self.ship.position
        angle = self.ship.angle+np.pi/2
        vel_l = np.array(self.ship.linearVelocity)
        vel_a = self.ship.angularVelocity

        #LIDAR measurements
        for i in range(LIDAR_NB_RAY):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(2*i*np.pi/(LIDAR_NB_RAY) + angle)*LIDAR_RANGE,
                pos[1] - math.cos(2*i*np.pi/(LIDAR_NB_RAY) + angle)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
            
    #- distance to target (ship's frame)
    #- target bearing
    #- angular velocity
    #- gimbal angle
        norm_pos = np.max((VIEWPORT_W,VIEWPORT_H))
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
            (self.thruster_angle / THRUSTER_MAX_ANGLE)
        ]
        state += [l.fraction for l in self.lidar]
        # # print state
        # if self.viewer is not None:
        #     print('\t'.join(["{:7.3}".format(s) for s in state]))

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        
        
        outside = (abs(pos.x - VIEWPORT_W*0.5) > VIEWPORT_W*0.49) or (abs(pos.y - VIEWPORT_H*0.5) > VIEWPORT_H*0.49)

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

        return np.array(state[2:]), reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            
            #self.viewer.set_bounds(0, 10*W, 0, 10*H)

            water = rendering.FilledPolygon(((0, 0), (0, VIEWPORT_H), (VIEWPORT_W, VIEWPORT_H), (VIEWPORT_W, 0)))
            self.water_color = rgb(126, 150, 233)
            water.set_color(*self.water_color)
            self.water_color_half_transparent = np.array((np.array(self.water_color) + rgb(255, 255, 255))) / 2
            self.viewer.add_geom(water)

            self.shiptrans = rendering.Transform()

            thruster = rendering.FilledPolygon(((-THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, 0),
                                              (THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT),
                                              (-THRUSTER_WIDTH / 2, -THRUSTER_HEIGHT)))
            self.thrustertrans = rendering.Transform()
            thruster.add_attr(self.thrustertrans)
            thruster.add_attr(self.shiptrans)
            thruster.set_color(.4, .4, .4)
            self.viewer.add_geom(thruster)
            
        #LIDAR RENDER
        for lid in self.lidar:
            self.viewer.draw_polyline( [lid.p1, lid.p2], color=(1,0,0), linewidth=1 )
            
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)

        self.shiptrans.set_translation(*self.ship.position)
        self.shiptrans.set_rotation(self.ship.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255