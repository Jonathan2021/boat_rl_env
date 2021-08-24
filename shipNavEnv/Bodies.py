from Box2D.b2 import fixtureDef, polygonShape, circleShape
import numpy as np
import math
import abc
from shipNavEnv.utils import getColor, rgb, make_half_circle
from enum import Enum
from gym.envs.classic_control import rendering
from shipNavEnv.Callbacks import LidarCallback
from Box2D import b2PolygonShape, b2FixtureDef, b2ChainShape, b2EdgeShape, b2CircleShape

class BodyType(Enum):
    BODY = 0,
    SHIP = 1,
    ROCK = 2,
    TARGET = 3

class Body:
    def __init__(self, world,*args, **kwargs):
        self.world = world
        self.body = None
        self.hit_with = []
        self.type = BodyType.BODY
        self._build(*args, **kwargs)
        self.args = args
        self.ship_view_trans = rendering.Transform()
        self.kwargs = kwargs

    @abc.abstractmethod
    def _build(self, **kwargs):
        pass

    def clean(self):
        self.ship_view_trans = rendering.Transform()
        self.hit_with = []

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
    def render(self, viewer, first_time=True, ship_view=None):
        pass

    def step(self, fps):
        pass

    def get_color(self):
        return self.body.color

    def get_color_ship_view(self):
        v_x, v_y = self.world.userData.ship.body.GetLocalVector(self.body.linearVelocity)
        col= (np.clip((v_x / Ship.Vmax + 1) / 2, 0, 1),
            np.clip((v_y / Ship.Vmax + 1) / 2, 0, 1),
            np.clip((self.body.angularVelocity / Ship.Rmax + 1) / 2, 0, 1))
        return col


class Obstacle(Body):
    DEFAULT_DIST = 1
    DEFAULT_BEARING = 0
    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs)
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
        return self.body.color2 if self.seen else self.body.color1 if self.is_hit() else self.body.color1


class RoundObstacle(Obstacle):
    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs)

    def render(self, viewer, first_time=True, ship_view=None):
        trans = self.body.transform
        if first_time:
            for f in self.body.fixtures:
                circle = rendering.make_circle(self.radius)
                circle.set_color(*self.get_color())
                viewer.add_geom(circle)
                circle.userData = self
                if ship_view:
                    t = self.ship_view_trans
                else:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    countour = rendering.make_circle(f.shape.radius, filled=False)
                    countour.add_attr(t)
                    countour.set_color(*self.get_color())
                    viewer.add_geom(countour)
                circle.add_attr(t)

class Ship(Body):
    # THRUSTER
    THRUSTER_MAX_ANGLE = 0.4    # [rad]
    THRUSTER_MAX_FORCE = 3e4    # [N]
    THRUSTER_MIN_THROTTLE = 0.4 # [%]
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
    Vmax = 20                  # [m/s]
    Rmax = 1*np.pi / 4             #[rad/s]
    K_Nr = (THRUSTER_MAX_FORCE*SHIP_HEIGHT*math.sin(THRUSTER_MAX_ANGLE)/(2*Rmax)) # [N.m/(rad/s)]
    K_Xu = THRUSTER_MAX_FORCE/Vmax # [N/(m/s)]
    SCALE_K_Yv = 10
    K_Yv = SCALE_K_Yv * K_Xu              # [N/(m/s)]
    VmaxY = Vmax
    VmaxX = Vmax / SCALE_K_Yv

    def __init__(self, world, init_angle, position, obs_radius,display_Traj = False, **kwargs):
        Body.__init__(self, world, init_angle, position, **kwargs)
        self.throttle = 1
        self.thruster_angle = 0
        self.type = BodyType.SHIP
        self.obs_radius = obs_radius
        self.trajPos = [position]
        self.trajDots = []
        
        #Rendering transform
        self.shiptrans = rendering.Transform()
        self.thrustertrans = rendering.Transform()


    def _build(self, init_angle, position, **kwargs):
        self.body = self.world.CreateDynamicBody(
                position=position,
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-self.SHIP_WIDTH / 2, -self.SHIP_HEIGHT/2), 
                        (+self.SHIP_WIDTH / 2, -self.SHIP_HEIGHT/2),
                        (self.SHIP_WIDTH / 2, +self.SHIP_HEIGHT/2),
                        (0, +self.SHIP_HEIGHT*0.6),
                        (-self.SHIP_WIDTH / 2, +self.SHIP_HEIGHT/2))),
                    density=0.0,
                    categoryBits=0x0010, #FIXME Same category as rocks ?
                    maskBits=0x1111,
                    restitution=0.0),
                linearDamping=0,
                angularDamping=0
                )
        # bumper
        self.L1 = 1.6 * self.SHIP_HEIGHT
        self.L2 = 6.4 * self.SHIP_HEIGHT

        base_circle = b2FixtureDef(shape=b2CircleShape(pos=(0,0), radius=self.L1), isSensor=True)
        upper_circle = b2FixtureDef(shape=b2CircleShape(pos=(0, self.L2 - self.L1), radius=self.L1), isSensor=True)
        rectangle = b2FixtureDef(shape=b2PolygonShape(vertices=(
            (-self.L1, 0),
            (self.L1, 0),
            (self.L1, self.L2-self.L1),
            (-self.L1, self.L2-self.L1))), isSensor=True)
        
        self.sensors = []

        base_circle = self.body.CreateFixture(base_circle, density=0)
        base_circle.userData = {'init_angle_render': math.pi, 'touching': []}
        self.sensors.append(base_circle)

        upper_circle = self.body.CreateFixture(upper_circle, density=0)
        upper_circle.userData = {'init_angle_render': 0, 'touching': []}
        self.sensors.append(upper_circle)

        rectangle = self.body.CreateFixture(rectangle, density=0)
        rectangle.userData = {'lines_render': [[(-self.L1, self.L2-self.L1), (-self.L1, 0)], [(self.L1, 0), (self.L1, self.L2-self.L1)]], 'touching': []}
        self.sensors.append(rectangle)

        self.MAX_LENGTH = np.sqrt(max(
                self.SHIP_WIDTH ** 2 + self.SHIP_HEIGHT ** 2,
                (self.SHIP_WIDTH / 2) ** 2 + (self.SHIP_HEIGHT * 1.1) ** 2))
                

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

    def bumper_state(self, ignore=[]):
        touches = set()
        for sens in self.sensors:
            touches.update(sens.userData['touching'])
        for body in ignore:
            if body in touches:
                touches.remove(body)
        return touches

    def thrust(self, inc_throttle, fps=30):
        #print(self.body.GetLocalVector(self.body.linearVelocity))
        inc_throttle = np.clip(inc_throttle, -1, 1)
        inc_throttle = inc_throttle * Ship.THRUSTER_MAX_THROTTLE_STEP / fps
        self.throttle = np.clip(self.throttle + inc_throttle, self.THRUSTER_MIN_THROTTLE, 1)


    def steer(self, steer, fps=30):
        steer = np.clip(steer, -1, 1)
        steer = steer * Ship.THRUSTER_MAX_ANGLE_STEP / fps
        self.thruster_angle = np.clip(self.thruster_angle + steer, -Ship.THRUSTER_MAX_ANGLE, Ship.THRUSTER_MAX_ANGLE)

    def clean(self):
        super().clean()
        self.throttle = 1
        self.thruster_angle = 0

    def add_geoms(self, viewer, ship_view):
        color = self.get_color()
        if not ship_view:
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
            COG.add_attr(self.shiptrans) # add ship angle and ship position
            
            COG.set_color(0, 0, 0)

            viewer.add_geom(COG)

        
            if self.obs_radius:
                horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
                horizon.set_color(*color)
                horizon.add_attr(self.shiptrans) # add ship angle and ship position

                viewer.add_geom(horizon)

            #trans = self.body.transform
        shapes = []
        for f in self.body.fixtures:
            isSensor = f in self.sensors
            if type(f.shape) is polygonShape:
                if not isSensor:
                    shape = rendering.FilledPolygon(f.shape.vertices)
                    shape.set_color(*color)
                    shapes.append(shape)
                else:
                    for line in f.userData['lines_render']:
                        shape = rendering.PolyLine(line, False)
                        shape.set_color(*(133, 114, 216))
                        shapes.append(shape)
            elif type(f.shape) is circleShape:
                if isSensor:
                    shape = make_half_circle(radius=f.shape.radius, init_angle=f.userData['init_angle_render'], filled=False) 
                    shape.set_color(*(133, 114, 216))
                else:
                    shape = rendering.make_circle(f.shape.radius)
                    shape.set_color(*color)
                shape.add_attr(rendering.Transform(translation=f.shape.pos))
                shapes.append(shape)
            
        for shape in shapes:
            if ship_view:
                shape.add_attr(self.ship_view_trans)
            else:
                shape.add_attr(self.shiptrans)
            shape.userData = self
            viewer.add_geom(shape)


    def render(self, viewer, first_time=True, ship_view=None):
        if first_time:
                self.add_geoms(viewer, ship_view)
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.thrustertrans.set_translation(0, -self.SHIP_HEIGHT / 2)
        self.shiptrans.set_translation(*self.body.position)
        self.shiptrans.set_rotation(self.body.angle)
        
    
    def update(self, addTraj=False):
        if addTraj:
            self.trajPos.append(self.body.position)

    def step(self, fps):
        COGpos = self.body.GetWorldPoint(self.body.localCenter)

        force_thruster = (-np.sin(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE * self.throttle,
                  np.cos(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE * self.throttle)
        
        localVelocity = self.body.GetLocalVector(self.body.linearVelocity)
        #print("local velocity Vx = {Vx} Vy = {Vy}".format(Vx = localVelocity[1], Vy = localVelocity[0]))
        force_damping_in_ship_frame = (-localVelocity[0] * Ship.K_Yv,-localVelocity[1] *Ship.K_Xu)
        
        force_damping = self.body.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.body.angle)* force_damping_in_ship_frame[0] -np.sin(self.body.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.body.angle)* force_damping_in_ship_frame[0] + np.cos(self.body.angle) * force_damping_in_ship_frame[1] )
        #print("force_damping Fx = {Fx} Fy = {Fy} Ftot = {Ftot} N".format(Fx = force_damping[0], Fy = force_damping[1], Ftot = np.sqrt(force_damping[0]**2+force_damping[1]**2)))
        torque_damping = -self.body.angularVelocity *Ship.K_Nr

        self.body.ApplyTorque(torque=torque_damping,wake=False)
        self.body.ApplyForce(force=force_thruster, point=self.body.position, wake=False)
        self.body.ApplyForce(force=force_damping, point=COGpos, wake=False)

class ShipLidar(Ship):
    def __init__(self, world, init_angle, position, nb_lidars, lidar_range, **kwargs):
        Ship.__init__(self, world, init_angle, position, **kwargs)
        self.nb_lidars = nb_lidars
        self.lidar_range = lidar_range
        self.lidars = [LidarCallback(dont_report_type = [BodyType.TARGET], dont_report_object=[self]) for _ in range(self.nb_lidars)]
        self.update()

    def _update_lidars(self):
        pos = self.body.position
        angle = self.body.angle + np.pi/2
        nb_lib_after_basics = self.nb_lidars - 3
        nb_back = nb_lib_after_basics // 3
        nb_front_left = (nb_lib_after_basics - nb_back) // 2
        nb_front_right = nb_lib_after_basics - nb_back - nb_front_left

        for i, lidar in enumerate(self.lidars):
            lidar.fraction = 1.0
            lidar.p1 = pos
            if i == 0:
                lidar.p2 = (
                    pos[0] + math.sin(angle + math.pi / 2) * self.lidar_range,
                    pos[1] - math.cos(angle + math.pi / 2) * self.lidar_range)
            elif i == 1:
                lidar.p2 = (
                    pos[0] + math.sin(angle) * self.lidar_range,
                    pos[1] - math.cos(angle) * self.lidar_range)
            elif i == 2:
                lidar.p2 = (
                    pos[0] + math.sin(angle + math.pi) * self.lidar_range,
                    pos[1] - math.cos(angle + math.pi) * self.lidar_range)
            elif i < nb_back + 3:
                j = i - 3
                lidar.p2 = (
                    pos[0] + math.sin((-math.pi * (j + 1)) / (nb_back + 1) + angle) * self.lidar_range,
                    pos[1] - math.cos((-math.pi * (j + 1)) / (nb_back + 1) + angle) * self.lidar_range)
            elif i < 3 + nb_back + nb_front_left:
                j = i - 3 - nb_back
                lidar.p2 = (
                    pos[0] + math.sin((math.pi / 2 * (j + 1)) / (nb_front_left + 1) + angle + math.pi / 2) * self.lidar_range,
                    pos[1] - math.cos((math.pi / 2 * (j + 1)) / (nb_front_left + 1) + angle + math.pi / 2) * self.lidar_range)
            else:
                j = i - 3 - nb_back - nb_front_left
                lidar.p2 = (
                    pos[0] + math.sin((math.pi / 2 * (j + 1)) / (nb_front_right + 1) + angle) * self.lidar_range,
                    pos[1] - math.cos((math.pi / 2 * (j + 1)) / (nb_front_right + 1) + angle) * self.lidar_range)
                    
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
    
    def update(self, addTraj = False):
        self._update_lidars()
        Ship.update(self, addTraj)
        
    def add_geoms(self, viewer, ship_view):
        Ship.add_geoms(self, viewer, ship_view)
            
    def render(self, viewer, first_time=True, ship_view=None):
        Ship.render(self, viewer, first_time, ship_view)
        if not ship_view:
            for lidar in self.lidars:
                viewer.draw_polyline( [lidar.p1, lidar.p2], color=rgb(255, 0, 0), linewidth=1)
            for pos in self.trajPos:
                dot = rendering.make_circle(radius=2, res=30, filled=True)
                dotpos = rendering.Transform()
                dotpos.set_translation(*pos)
                dot.add_attr(dotpos)
                viewer.add_geom(dot)
                self.trajDots.append(dot)
            self.trajPos = []
            shipColor = np.array(self.get_color())
            startColor = shipColor*0.5 +0.5*np.array([1,1,1])
            for i,dot in enumerate(self.trajDots):
                c = i/len(self.trajDots)
                dotColor = tuple((1-c)*startColor +c*shipColor)
                dot.set_color(*dotColor)

class ShipObstacle(Ship, Obstacle):
    def __init__(self, world, init_angle, position, **kwargs):
        #super(Obstacle).__init__(world, init_x, init_y)
        Obstacle.clean(self)
        Ship.__init__(self, world, init_angle, position, 0, **kwargs)
        self.target_world_direction = self.body.GetWorldVector((0,1))

    def _build(self, *args, **kwargs):
        Ship._build(self, *args, **kwargs)
        self.body.color1 = rgb(255, 255, 255) # white
        self.body.color2 = rgb(41, 14, 9) # darker brown
        self.body.color3 = rgb(83, 43, 9) # brown
        self.doSleep = True

    def render(self, viewer, first_time=True, ship_view=None):
        Ship.render(self, viewer, first_time, ship_view)
        if not ship_view:
            target_dir_p2 = [ax * 20 + self.body.position[i] for i, ax in enumerate(self.target_world_direction)]
            viewer.draw_polyline( [self.body.position, target_dir_p2], color=rgb(255, 0, 0), linewidth=1)

    def take_actions(self, fps):
        look_ahead_lidar = LidarCallback(dont_report_type = [BodyType.TARGET], dont_report_object=[self])
        look_ahead_lidar.p1 = self.body.position
        look_ahead_lidar.p2 = self.body.GetWorldPoint((0, self.L2))
        self.world.RayCast(look_ahead_lidar, look_ahead_lidar.p1, look_ahead_lidar.p2) # can still bump on side if barely going towards target.
        if look_ahead_lidar.fraction < 1: # If something in front
            #print("Something in front ! Gotta turn right !")
            self.steer(1, fps) # steer to starboard as much as possible
            return

        touches = self.bumper_state(ignore=[self.world.userData.target.body])
        bearings = [np.arctan2(*reversed(self.body.GetLocalPoint(body.position))) for body in touches]
        for bearing in bearings:
            if bearing <= math.pi / 2 and bearing >= -math.pi / 6: # Detected something on front starboard side
                #print("Something on my right ! Gotta slow down and turn right")
                #print("Bearing to it %f" % bearing)
                self.thrust(-1, fps) # slow down
                self.steer(1, fps) # go starboard
                return

        angle_to_target_dir = np.arctan2(*reversed(self.body.GetLocalVector(self.target_world_direction))) - math.pi/2
        #print("Angle to target is %f" % angle_to_target_dir)
        self.thrust(1, fps) # accelerate
        if abs(angle_to_target_dir) > math.pi / 6:
            #print("Gotta rectify")
            look_to_target_lidar = LidarCallback(dont_report_type = [BodyType.TARGET], dont_report_object=[self])
            look_to_target_lidar.p1 = self.body.position
            look_to_target_lidar.p2 = self.body.GetWorldPoint((np.cos(angle_to_target_dir) * self.L2, np.sin(angle_to_target_dir) * self.L2))
            self.world.RayCast(look_to_target_lidar, look_to_target_lidar.p1, look_to_target_lidar.p2)
            if look_to_target_lidar.fraction == 1: # Nothing where we want to go
                #print("Nothing in my pass mouahahahaha")
                if angle_to_target_dir < 0 and angle_to_target_dir > -math.pi: # and self.thruster_angle >= 0:
                    #print("Guess I'll steer right")
                    self.steer(1, fps) # Steer port
                else: # and self.thruster_angle <= 0:
                    #print("Ok let's steer left!")
                    self.steer(-1, fps) # Steer starboard
        else:
            #print("Straightening up of %f" % (-self.thruster_angle / (self.THRUSTER_MAX_ANGLE_STEP / fps)))
            self.steer(-self.thruster_angle / (self.THRUSTER_MAX_ANGLE_STEP / fps), fps) #straighten up


           
    
    def step(self, fps):
        self.take_actions(fps)
        #self.thruster_angle = 0
        #self.throttle = 0
        Ship.step(self, fps)

    def unsee(self):
        Obstacle.unsee(self)
        self.bearing_to_ship = Obstacle.DEFAULT_BEARING 
    
    
    def get_color(self):
        return self.body.color2 if self.seen else self.body.color1 if self.is_hit() else self.body.color1

class ShipObstacleRand(ShipObstacle):
    def __init__(self, world, init_angle, position, **kwargs):
        ShipObstacle.__init__(self, world, init_angle, position, **kwargs)
        self.body.color1 = rgb(60,185,119)

    def take_actions(self, fps):
        steer = np.random.uniform(-1, 1)
        thrust = np.random.uniform(-1, 1)

        self.steer(steer, fps)
        self.thrust(thrust, fps)
    
class Rock(RoundObstacle):
    RADIUS = 20
    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs)
        self.type = BodyType.ROCK

    def _build(self, position, **kwargs):
        radius = np.random.uniform(0.5*Rock.RADIUS,2*Rock.RADIUS)

        self.body = self.world.CreateStaticBody(
            position=position, # FIXME Should have something like: map.get_random_available_position()
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
    RADIUS = 30
    def __init__(self, world, position, random_radius=True, **kwargs):
        self.random_radius = random_radius
        self.radius = self.RADIUS
        super().__init__(world, position, **kwargs)
        self.type = BodyType.TARGET

    def _build(self, position, **kwargs):
        if self.random_radius:
            self.radius = np.random.uniform(0.5*self.RADIUS,2*self.RADIUS)
        else:
            self.radius = self.RADIUS

        self.body =  self.world.CreateStaticBody(
            position = position,
            angle = 0.0,
            fixtures = fixtureDef(
            shape = circleShape(pos=(0,0), radius = self.radius),
            categoryBits=0x0010,
            maskBits=0x1111,
            restitution=0.1, isSensor=True, userData={'touching': []}))
        self.body.userData = self
    
    def render(self, viewer, first_time=True, ship_view=None):
        trans = self.body.transform
        for f in self.body.fixtures:
            t = rendering.Transform(translation=trans * f.shape.pos)
            viewer.draw_circle(f.shape.radius, color= rgb(15,15,15), filled=False, linewidth=2).add_attr(t)
