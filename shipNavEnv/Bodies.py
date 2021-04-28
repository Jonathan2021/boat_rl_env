from Box2D.b2 import fixtureDef, polygonShape, circleShape
import numpy as np
import math
import abc
from shipNavEnv.utils import getColor, rgb
from enum import Enum
from gym.envs.classic_control import rendering

class BodyType(Enum):
    BODY = 0,
    SHIP = 1,
    ROCK = 2,
    TARGET = 3

class Body:
    def __init__(self, world, *args, **kwargs):
        self.world = world
        self.body = None
        self._build(*args, **kwargs)
        self.type = BodyType.BODY
        self.hit_with = []
        self.reset()

    @abc.abstractmethod
    def _build(self, **kwargs):
        pass

    def clear_hit(self):
        self.hit_with = []
    
    def unhit(self, body):
        self.hit_with.remove(body)

    def is_hit(self):
        return len(self.hit_with) > 0

    def destroy(self):
        self.world.DestroyBody(self.body)

    @abc.abstractmethod
    def render(self, viewer):
        pass

class Obstacle(Body):
    DEFAULT_DIST = 1
    DEFAULT_BEARING = 0
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False

    def reset(self):
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False

class RoundObstacle(Obstacle):
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)

    def render(self, viewer):
        t = self.body.transform
        viewer.draw_circle(self.radius, color = (self.body.color1 if self.is_hit() else (self.body.color2 if self.seen else self.body.color3))).add_attr(t)
        #viewer.draw_circle(f.shape.radius, color= (obj.color2 if f.body.userData.is_hit() else (obj.color3 if f.body.userData.seen else obj.color2)), filled=False, linewidth=2).add_attr(t)
        #viewer.draw_circle(f.shape.radius, color= (obj.color1 if f.body.userData.seen else obj.color3)).add_attr(t)
        #        viewer.draw_circle(f.shape.radius, color= (obj.color3 if f.body.userData.seen else obj.color1), filled=False, linewidth=2).add_attr(t)


class Ship(Body):
    # THRUSTER
    THRUSTER_MIN_THROTTLE = 0.4 # [%]
    THRUSTER_MAX_ANGLE = 0.4    # [rad]
    THRUSTER_MAX_FORCE = 3e4    # [N]
    THURSTER_MAX_DIFF = 0.1     # ???
    THRUSTER_MAX_ANGLE_STEP = 0.01 
    THRUSTER_MAX_THROTTLE_STEP = 0.01

    THRUSTER_HEIGHT = 20        # [m]
    THRUSTER_WIDTH = 0.8        # [m]

    # SHIP
    SHIP_HEIGHT = 20            # [m]
    SHIP_WIDTH = 5              # [m]

    # dummy parameters for fast simulation
    SHIP_MASS = 27e1            # [kg]
    SHIP_INERTIA = 280e1        # [kg.m²]
    Vmax = 300                  # [m/s]
    Rmax = 1*np.pi              #[rad/s]
    K_Nr = (THRUSTER_MAX_FORCE*SHIP_HEIGHT*math.sin(THRUSTER_MAX_ANGLE)/(2*Rmax)) # [N.m/(rad/s)]
    K_Xu = THRUSTER_MAX_FORCE/Vmax # [N/(m/s)]
    K_Yv = 10*K_Xu              # [N/(m/s)]


    def __init__(self, world, init_angle, init_x, init_y, obs_radius, **kwargs):
        super().__init__(world, init_angle, init_x, init_y, **kwargs)
        self.throttle = 0
        self.thruster_angle = 0
        self.type = BodyType.SHIP
        self.obs_radius = obs_radius
        self.rendered_once = False

    def _build(self, init_angle, init_x, init_y, **kwargs):
        self.body = self.world.CreateDynamicBody(
                position=(init_x, init_y),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-Ship.SHIP_WIDTH / 2, 0),
                        (+Ship.SHIP_WIDTH / 2, 0),
                        (Ship.SHIP_WIDTH / 2, +Ship.SHIP_HEIGHT),
                        (0, +Ship.SHIP_HEIGHT*1.2),
                        (-Ship.SHIP_WIDTH / 2, +Ship.SHIP_HEIGHT))),
                    density=0.0,
                    categoryBits=0x0010, #FIXME Same category as rocks ?
                    maskBits=0x1111,
                    restitution=0.0),
                linearDamping=0,
                angularDamping=0
                )

        self.body.color1 = getColor(idx=0)
        self.body.linearVelocity = (0.0,0.0)
        self.body.angularVelocity = 0
        self.body.userData = self

        newMassData = self.body.massData
        newMassData.mass = Ship.SHIP_MASS
        newMassData.center = (0.0, Ship.SHIP_HEIGHT/2) #FIXME Is this the correct center of mass ?
        newMassData.I = Ship.SHIP_INERTIA + Ship.SHIP_MASS*(newMassData.center[0]**2+newMassData.center[1]**2) # inertia is defined at origin location not localCenter
        self.body.massData = newMassData

    def can_see(self, obstacle: Obstacle):
        return obstacle.distance_to_ship < self.obs_radius
        

    def thrust(self, throttle, fps=60):
        throttle = np.clip(throttle, -1, 1)
        throttle = throttle * Ship.THRUSTER_MAX_THROTTLE_STEP * 60 / fps

        self.throttle = np.clip(self.throttle + throttle, Ship.THRUSTER_MIN_THROTTLE, 1)

    def steer(self, steer, fps=60):
        steer = np.clip(steer, -1, 1)
        steer = steer * Ship.THRUSTER_MAX_ANGLE_STEP * 60 / fps

        self.thruster_angle = np.clip(self.thruster_angle + steer, -Ship.THRUSTER_MAX_ANGLE, Ship.THRUSTER_MAX_ANGLE)

    def reset(self):
        self.throttle = 0
        self.thruster_angle = 0

    def render(self, viewer):
        if not self.rendered_once:
            self.shiptrans = rendering.Transform()
            self.thrustertrans = rendering.Transform()
            self.COGtrans = rendering.Transform()
            
            thruster = rendering.FilledPolygon((
                (-self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT),
                (-self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT)))
            
            thruster.name = "thruster"
            
            thruster.add_attr(self.thrustertrans) # add thruster angle, assigned later
            thruster.add_attr(self.shiptrans) # add ship angle and ship position, assigned later
            thruster.set_color(*self.body.color1)
            
            viewer.add_geom(thruster)
            
            COG = rendering.FilledPolygon((
                (-Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, -Ship.THRUSTER_WIDTH/0.2),
                (Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, Ship.THRUSTER_WIDTH/0.2)))
            COG.add_attr(self.COGtrans) # add COG position
            COG.add_attr(self.shiptrans) # add ship angle and ship position

            COG.name = "COG"
            
            COG.set_color(0.0, 0.0, 0.0)
            viewer.add_geom(COG)
            horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
            horizon.set_color(*self.body.color1)
            horizon.add_attr(self.shiptrans) # add ship angle and ship position

        viewer.add_geom(horizon)

        trans = self.body.transform
        for f in self.body.fixtures:
            path = [trans * v for v in f.shape.vertices]
            viewer.draw_polygon(path, color=self.body.color1)

        self.shiptrans.set_translation(*self.body.position)
        self.shiptrans.set_rotation(self.body.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.COGtrans.set_translation(*self.body.localCenter)


class Rock(RoundObstacle):
    RADIUS = 20
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
        self.type = BodyType.ROCK

    def _build(self, x, y, **kwargs):
        radius = np.random.uniform(0.5*Rock.RADIUS,2*Rock.RADIUS)

        self.body = self.world.CreateStaticBody(
            position=(x, y), # FIXME Should have something like: map.get_random_available_position()
            angle=np.random.uniform( 0, 2*math.pi), # FIXME Not really useful if circle shaped
            fixtures=fixtureDef(
            shape = circleShape(pos=(0,0),radius = radius),
            categoryBits=0x0010, # FIXME Move categories to MACRO
            maskBits=0x1111, # FIXME Same as above + it can collide with itself -> may cause problem when generating map ?
            restitution=1.0))
        self.body.color1 = rgb(83, 43, 9) # brown
        self.body.color2 = rgb(41, 14, 9) # darker brown
        self.body.color3 = rgb(255, 255, 255) # seen

        self.radius = radius

        self.body.userData = self

class Target(RoundObstacle):
    RADIUS = 20
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
        self.type = BodyType.TARGET

    def _build(self, x, y, **kwargs):
        self.body =  self.world.CreateStaticBody(
            position = (x, y),
            angle = 0.0,
            fixtures = fixtureDef(
            shape = circleShape(pos=(0,0), radius = Target.RADIUS),
            categoryBits=0x0010,
            maskBits=0x1111,
            restitution=0.1))
        self.body.color1 = rgb(255,0,0)
        self.body.color2 = rgb(0,255,0)
        self.body.color3 = rgb(255, 255, 255) # seen
        self.body.userData = self
        self.radius = self.RADIUS
