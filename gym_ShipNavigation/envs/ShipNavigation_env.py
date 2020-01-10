import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)
import gym
from gym import spaces
from gym.utils import seeding

"""

The objective of this environment is to land a rocket on a ship.

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
    
Continuous control inputs are:
    - gimbal (left/right)

"""

CONTINUOUS = False
FPS = 60
SCALE_S = 0.15  # Temporal Scaling, lower is faster - adjust forces appropriately
INITIAL_RANDOM = 0.01  # Random scaling of initial velocity, higher is more difficult

START_HEIGHT = 100.0
START_SPEED = 1.0

# ROCKET
MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 11.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.8
ENGINE_WIDTH = ENGINE_HEIGHT * 0.3
THRUSTER_HEIGHT = ROCKET_HEIGHT * 1.

# SHIP
SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 40

# VIEWPORT
VIEWPORT_H = 900
VIEWPORT_W = 1600
H = 7. * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H

# ROCK
ROCK_RADIUS = 0.02*W

#LIDAR
LIDAR_RANGE   = 160/SCALE_S
LIDAR_NB_RAY = 16;

MEAN = np.array([-0.034, -0.15, -0.016, 0.0024, 0.0024, 0.137,
                 - 0.02, -0.01, -0.8, 0.002])
VAR = np.sqrt(np.array([0.08, 0.33, 0.0073, 0.0023, 0.0023, 0.8,
                        0.085, 0.0088, 0.063, 0.076]))



class ShipNavigationEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World(gravity=(0,0))
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.targetX = 0.0
        self.targetY = 0.0
        self.target = None
        
        high = np.ones(2+LIDAR_NB_RAY, dtype=np.float32)
        low = -high

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if CONTINUOUS:
            self.action_space = spaces.Box(-1.0, +1.0, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.lander:
            return
        self.world.DestroyBody(self.lander)
        self.world.DestroyBody(self.target)     
        self.lander = None

    def reset(self):
        self._destroy()
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.gimbal = 0.0
        self.stepnumber = 0
        self.lidar_render = 0

        initial_x = np.random.uniform( 2*ROCKET_HEIGHT, W-2*ROCKET_HEIGHT)
        initial_y = np.random.uniform( 2*ROCKET_HEIGHT, H-2*ROCKET_HEIGHT)
        
        self.targetX = np.random.uniform( 10*ROCKET_HEIGHT, W-10*ROCKET_HEIGHT)
        self.targetY = np.random.uniform( 10*ROCKET_HEIGHT, H-10*ROCKET_HEIGHT)
        self.target = self.world.CreateStaticBody(
                position = (self.targetX,self.targetY),
                angle = 0.0,
                fixtures = fixtureDef(
                        shape = polygonShape(vertices = ((ROCK_RADIUS,0),
                                                    (0,ROCK_RADIUS),
                                                    (-ROCK_RADIUS,0),
                                                    (0,-ROCK_RADIUS)))                                        
                                        ))
        self.target.color1 = rgb(255,0,0)
        
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-ROCKET_WIDTH / 2, 0),
                                             (+ROCKET_WIDTH / 2, 0),
                                             (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                                             (0, +ROCKET_HEIGHT*1.2),
                                             (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT))),
                density=1.0,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0),
                linearDamping=5,
                angularDamping=20,
        )

        # self.lander.angularDamping = 0.9

        self.lander.color1 = rgb(230, 230, 230)

        self.lander.linearVelocity = (
            -self.np_random.uniform(0, INITIAL_RANDOM) * START_SPEED * (initial_x - W / 2) / W,
            -START_SPEED)

        self.lander.angularVelocity = (1 + INITIAL_RANDOM) * np.random.uniform(-1, 1)

        self.drawlist = [self.lander,self.target]
        
        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(LIDAR_NB_RAY)]
        
        
        if CONTINUOUS:
            return self.step([0, 0, 0])[0]
        else:
            return self.step(2)[0]

    def step(self, action):

        self.force_dir = 0

        if CONTINUOUS:
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.15 / FPS
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0.1

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-np.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * (self.power),
                 np.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * (self.power))
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)
        self.world.Step(1.0 / FPS, 60, 60)
        
        pos = self.lander.position
        angle = self.lander.angle+np.pi/2
        vel_l = np.array(self.lander.linearVelocity)
        vel_a = self.lander.angularVelocity

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
        norm_pos = np.max((W,H))
        x_distance = (self.targetX - pos.x)/norm_pos
        y_distance = (self.targetY - pos.y)/norm_pos
        distance = np.linalg.norm((x_distance, y_distance))
        u = x_distance*np.cos(angle) + y_distance*np.sin(angle)
        v = -x_distance*np.sin(angle) + y_distance*np.cos(angle)
        bearing = np.arctan2(v,u)/(np.pi)
        #print("bearing = %0.3f u = %0.3f v = %0.3f" %(bearing,u,v))
        
        state = [
            distance,
            bearing,
            vel_a,
            (self.gimbal / GIMBAL_THRESHOLD)
        ]
        state += [l.fraction for l in self.lidar]
        # # print state
        # if self.viewer is not None:
        #     print('\t'.join(["{:7.3}".format(s) for s in state]))

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        
        
        outside = (abs(pos.x - W*0.5) > W*0.49) or (abs(pos.y - H*0.5) > H*0.49)

        hit_target = (distance < (2*ROCK_RADIUS)/norm_pos)
        done = False
        
        reward = -1.0/FPS

        if outside:
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
            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon(((0, 0), (0, H), (W, H), (W, 0)))
            self.sky_color = rgb(126, 150, 233)
            sky.set_color(*self.sky_color)
            self.sky_color_half_transparent = np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            engine = rendering.FilledPolygon(((-ENGINE_WIDTH / 2, 0),
                                              (ENGINE_WIDTH / 2, 0),
                                              (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                                              (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT)))
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(.4, .4, .4)
            self.viewer.add_geom(engine)
            
        #LIDAR RENDER
        #self.lidar_render = (self.lidar_render+1) % 100
        #i = self.lidar_render
        #if i < 2*len(self.lidar):
        #    l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
        #    self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )
        for lid in self.lidar:
            self.viewer.draw_polyline( [lid.p1, lid.p2], color=(1,0,0), linewidth=1 )
            
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255