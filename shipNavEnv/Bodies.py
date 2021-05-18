from Box2D.b2 import fixtureDef, polygonShape, circleShape
import numpy as np
import math
import abc
from shipNavEnv.utils import getColor, rgb
from enum import Enum
from gym.envs.classic_control import rendering
from shipNavEnv.Callbacks import LidarCallback

class BodyType(Enum):
    BODY = 0,
    SHIP = 1,
    ROCK = 2,
    TARGET = 3

class Body:
    def __init__(self, world, *args, **kwargs):
        self.world = world
        self.body = None
        self.hit_with = []
        self.type = BodyType.BODY
        self.rendered_once = False
        self._build(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def _build(self, **kwargs):
        pass

    def clean(self):
        self.hit_with = []
        self.rendered_once = False

    def reset(self):
        self.destroy()
        self.body = None
        self._build(*self.args, **self.kwargs)

    def clear_hit(self):
        self.hit_with = []
    
    def unhit(self, body):
        self.hit_with.remove(body)

    def is_hit(self):
        return len(self.hit_with) > 0

    def destroy(self):
        self.world.DestroyBody(self.body)
        self.clean()

    @abc.abstractmethod
    def render(self, viewer):
        pass

    def step(self, fps):
        pass

    def get_color(self):
        return self.body.color


class Obstacle(Body):
    DEFAULT_DIST = 1
    DEFAULT_BEARING = 0
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False

    def clean(self):
        super().clean()
        self.unsee()

    def unsee(self):
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False

    def get_color(self):
        return self.body.color1 if self.is_hit() else self.body.color2 if self.seen else self.body.color3

class RoundObstacle(Obstacle):
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)

    def render(self, viewer):
        trans = self.body.transform
        for f in self.body.fixtures:
            t = rendering.Transform(translation=trans * f.shape.pos)
            viewer.draw_circle(self.radius, color = self.get_color()).add_attr(t)
            viewer.draw_circle(f.shape.radius, color= (self.body.color2 if self.is_hit() else (self.body.color3 if self.seen else self.body.color2)), filled=False, linewidth=2).add_attr(t)


class Ship(Body):
    # THRUSTER
    THRUSTER_MIN_THROTTLE = 0.4 # [%]
    THRUSTER_MAX_ANGLE = 0.4    # [rad]
    THRUSTER_MAX_FORCE = 3e4    # [N]
    THURSTER_MAX_DIFF = 0.1     # ???
    THRUSTER_MAX_ANGLE_STEP = 0.60 
    THRUSTER_MAX_THROTTLE_STEP = 0.60

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
        Body.__init__(self, world, init_angle, init_x, init_y, **kwargs)
        self.throttle = 0
        self.thruster_angle = 0
        self.type = BodyType.SHIP
        self.obs_radius = obs_radius

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

        self.body.color = getColor(idx=0)
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
        throttle = throttle * Ship.THRUSTER_MAX_THROTTLE_STEP / fps
        self.throttle = np.clip(self.throttle + throttle, Ship.THRUSTER_MIN_THROTTLE, 1)

    def steer(self, steer, fps=60):
        steer = np.clip(steer, -1, 1)
        steer = steer * Ship.THRUSTER_MAX_ANGLE_STEP / fps

        self.thruster_angle = np.clip(self.thruster_angle + steer, -Ship.THRUSTER_MAX_ANGLE, Ship.THRUSTER_MAX_ANGLE)

    def clean(self):
        super().clean()
        self.throttle = 0
        self.thruster_angle = 0

    def render(self, viewer):
        color = self.get_color()
        if not self.rendered_once:
            self.shiptrans = rendering.Transform()
            self.thrustertrans = rendering.Transform()
            self.COGtrans = rendering.Transform()
            
            thruster = rendering.FilledPolygon((
                (-self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT),
                (-self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT)))
            
            thruster.add_attr(self.thrustertrans) # add thruster angle, assigned later
            thruster.add_attr(self.shiptrans) # add ship angle and ship position, assigned later
            thruster.set_color(*color)
            
            viewer.add_geom(thruster)
            
            COG = rendering.FilledPolygon((
                (-Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, -Ship.THRUSTER_WIDTH/0.2),
                (Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, Ship.THRUSTER_WIDTH/0.2)))
            COG.add_attr(self.COGtrans) # add COG position
            COG.add_attr(self.shiptrans) # add ship angle and ship position

            COG.set_color(0.0, 0.0, 0.0)
            viewer.add_geom(COG)
            if self.obs_radius:
                horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
                horizon.set_color(*color)
                horizon.add_attr(self.shiptrans) # add ship angle and ship position

                viewer.add_geom(horizon)

        trans = self.body.transform
        for f in self.body.fixtures:
            path = [trans * v for v in f.shape.vertices]
            viewer.draw_polygon(path, color=color)

        self.shiptrans.set_translation(*self.body.position)
        self.shiptrans.set_rotation(self.body.angle)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.COGtrans.set_translation(*self.body.localCenter)
        
        self.rendered_once = True
    
    def update(self):
        pass

    def step(self, fps):
        COGpos = self.body.GetWorldPoint(self.body.localCenter)

        force_thruster = (-np.sin(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE,
                  np.cos(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE )
        
        localVelocity = self.body.GetLocalVector(self.body.linearVelocity)

        force_damping_in_ship_frame = (-localVelocity[0] * Ship.K_Yv,-localVelocity[1] *Ship.K_Xu)
        
        force_damping = self.body.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.body.angle)* force_damping_in_ship_frame[0] -np.sin(self.body.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.body.angle)* force_damping_in_ship_frame[0] + np.cos(self.body.angle) * force_damping_in_ship_frame[1] )
        
        torque_damping = -self.body.angularVelocity *Ship.K_Nr

        self.body.ApplyTorque(torque=torque_damping,wake=False)
        self.body.ApplyForce(force=force_thruster, point=self.body.position, wake=False)
        self.body.ApplyForce(force=force_damping, point=COGpos, wake=False)

class ShipLidar(Ship):
    def __init__(self, world, init_angle, init_x, init_y, nb_lidars, lidar_range, **kwargs):
        Ship.__init__(self, world, init_angle, init_x, init_y, 0, **kwargs)
        self.nb_lidars = nb_lidars
        self.lidar_range = lidar_range
        self.lidars = [LidarCallback() for _ in range(self.nb_lidars)]
        self.update()

    def _update_lidars(self):
        pos = self.body.position
        for i, lidar in enumerate(self.lidars):
            lidar.fraction = 1.0
            lidar.p1 = pos
            lidar.p2 = (
                    pos[0] + math.sin(2 * math.pi * i / self.nb_lidars) * self.lidar_range,
                    pos[1] - math.cos(2 * math.pi / self.nb_lidars) * self.lidar_range)
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
    
    def update(self):
        self._update_lidars()

    def render(self, viewer):
        Ship.render(self, viewer)
        for lidar in self.lidars:
            viewer.draw_polyline( [lidar.p1, lidar.p2], color=rgb(255, 0, 0), linewidth=1)

class ShipObstacle(Ship, Obstacle):
    def __init__(self, world, init_angle, init_x, init_y, **kwargs):
        #super(Obstacle).__init__(world, init_x, init_y)
        Obstacle.clean(self)
        Ship.__init__(self, world, init_angle, init_x, init_y, 0, **kwargs)

    def _build(self, init_angle, init_x, init_y, **kwargs):
        Ship._build(self, init_angle, init_x, init_y, **kwargs)
        self.body.color1 = rgb(83, 43, 9) # brown
        self.body.color2 = rgb(41, 14, 9) # darker brown
        self.body.color3 = rgb(255, 255, 255) # seen
        self.doSleep = True

    def render(self, viewer):
        Ship.render(self, viewer)

    def take_random_actions(self, fps):
        steer = np.random.uniform(-1, 1)
        thrust = np.random.uniform(-1, 1)
        self.steer(steer, fps)
        self.thrust(thrust, fps)
    
    def step(self, fps):
        self.take_random_actions(fps)
        Ship.step(self, fps)

    def get_color(self):
        return self.body.color1 if self.is_hit() else self.body.color2 if self.seen else self.body.color3
    
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
