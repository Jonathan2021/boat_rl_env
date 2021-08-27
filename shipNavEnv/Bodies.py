from Box2D.b2 import fixtureDef, polygonShape, circleShape
import numpy as np
import math
import abc
from shipNavEnv.utils import getColor, rgb, make_half_circle
from enum import Enum
from gym.envs.classic_control import rendering
from shipNavEnv.Callbacks import LidarCallback, PlaceOccupied
from Box2D import b2PolygonShape, b2FixtureDef, b2ChainShape, b2EdgeShape, b2CircleShape


# FIXME Maybe add a different body type for "enemy" ships (would help in other parts)
class BodyType(Enum):
    BODY = 0,
    SHIP = 1,
    ROCK = 2,
    TARGET = 3


class Body:
    """
    Base class for all bodies in the world.
    Contains minimal init, common logic and abstract methods
    """
    def __init__(self, world,*args, **kwargs):
        self.world = world # FIXME This is the Box2D world directly, when we should use our own world class and API from it
        self.body = None # This will be the Box2D body of this object
        self.hit_with = [] # What is currently hitting this body
        self.type = BodyType.BODY # Type of body
        self._build(*args, **kwargs) # Build the body (with Box2D)
        self.args = args # store the arguments that may be useful
        self.ship_view_trans = rendering.Transform() # Transformation for the ship view (radar). Geoms in the gym.envs.classic_control.rendering will have this transform (reference) as an attribute, so we access it from here at each timestep to update it.
        self.kwargs = kwargs # store the dict arg for latter use

    
    @abc.abstractmethod
    def _build(self, **kwargs):
        """
        Abstract class for building the body. Called by __init__ at object creation.
        """
        pass

    def clean(self):
        """
        'Clean' the object to factory new without destroying it (removing body etc)
        """
        self.ship_view_trans = rendering.Transform()
        self.hit_with = []

    def reset(self):
        """
        Destroy and rebuild the object from scratch.
        """
        self.destroy()
        self._build(*self.args, **self.kwargs)

    def clear_hit(self):
        """
        Remove hits
        """
        self.hit_with = []
    
    def unhit(self, body):
        """
        Remove a collision with a specific body from the hit list
        """
        self.hit_with.remove(body)

    def is_hit(self):
        """ Check if has a collision registered """
        return len(self.hit_with) > 0

    def destroy(self):
        """
        Destroy the body and clean attributes
        """
        self.world.DestroyBody(self.body)
        self.body = None
        self.clean()

    # FIXME: render isn't really a suitable name since it doesn't really render anything but just populates the viewer with geoms...
    @abc.abstractmethod
    def render(self, viewer, first_time=True, ship_view=None):
        """ Abstract method for rendering the object graphically """
        pass

    def step(self, fps):
        """ Logic at each timestep """
        pass

    def get_color(self):
        """ Get the color (useful for render) """
        return self.body.color

    def get_color_ship_view(self):
        """
        Get the color in the ship view (radar) of the body.
        Information is encoded in the color channels (supposedly obtained by AIS or other means).
        """

        v_x, v_y = self.world.userData.ship.body.GetLocalVector(self.body.linearVelocity) # FIXME Implementation specific, ship should be passed as a parameter probably by World or other
        col= (np.clip((v_x / Ship.Vmax + 1) / 2, 0, 1), # Red is speed in the 
            np.clip((v_y / Ship.Vmax + 1) / 2, 0, 1),
            np.clip((self.body.angularVelocity / Ship.Rmax + 1) / 2, 0, 1))
        return col

    # TODO Should add methods to avoid direct access to variables by other objects (such as get_body() etc.)


class Obstacle(Body):
    """
    Generic obstacle base class.
    Builds on top of Body with logic specific to obstacles.
    """

    # FIXME These defaults were used for the old radar system, which is still in the code but not of much use.
    DEFAULT_DIST = 1 # Normalized between -1 and 1, so just means far away
    DEFAULT_BEARING = 0 # FIXME Should just be a random number in -1, 1

    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs) # Call Body init (building object etc)

        # What is the distance / bearing if undetected
        # FIXME the following are necessary for the deprecated radar system (should it be removed remove them)
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False # Has this obstacle been seen.

    def clean(self):
        super().clean()
        self.unsee()

    def unsee(self):
        """ Obstacle is no longer visible by agent. Values are back to default. """
        self.distance_to_ship = self.DEFAULT_DIST
        self.bearing_from_ship = self.DEFAULT_BEARING
        self.seen = False

    def get_color(self):
        # Color changes depending on the state
        return self.body.color2 if self.seen else self.body.color1 if self.is_hit() else self.body.color1


class RoundObstacle(Obstacle):
    """
    Round obstacle (which are just 'target' (used to be a solid obstacle) and rocks.
    """
    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs)

    def render(self, viewer, first_time=True, ship_view=None):
        trans = self.body.transform
        if first_time: # If first time, we need to add geoms to the viewer
            for f in self.body.fixtures:
                # FIXME Not fault proof if some fixtures are not circular in the future
                circle = rendering.make_circle(self.radius)
                circle.set_color(*self.get_color())
                viewer.add_geom(circle)
                circle.userData = self
                
                # Add translation to object depending on the view
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
    """
    Ship base class.
    Contains the ship logic, default values etc.
    """

    # THRUSTER
    THRUSTER_MAX_ANGLE = 0.4    # [rad]
    THRUSTER_MAX_FORCE = 3e4    # [N]
    THRUSTER_MIN_THROTTLE = 0.4 # [%]
    THRUSTER_MAX_ANGLE_STEP = 0.60 # [rad]
    THRUSTER_MAX_THROTTLE_STEP = 0.60 # [%]

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
    VmaxY = Vmax                # [m/s]
    VmaxX = Vmax / SCALE_K_Yv # [m/s]
    
    # bumper dimensions
    L1 = 1.6 * SHIP_HEIGHT
    L2 = 6.4 * SHIP_HEIGHT

    def __init__(self, world, init_angle, position, obs_radius,display_Traj = False, **kwargs):
        Body.__init__(self, world, init_angle, position, **kwargs)
        self.throttle = 1 # Full force by default (TODO: Make it random for robustness?)
        self.thruster_angle = 0 # Angle of thruster is 0 (straight) TODO same as above
        self.type = BodyType.SHIP
        self.obs_radius = obs_radius # Radius used for old and new radar
        self.trajPos = [position] # position history
        self.trajDots = []
        self.dont_render_fixture = [] # list of fixtures not to render FIXME should probably move this up to the Body class
        
        # Rendering transform
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

        # Building bumper fixtures around ship (which are a collection of sensors)
        base_circle = b2FixtureDef(shape=b2CircleShape(pos=(0,0), radius=self.L1), isSensor=True) # Lower part
        upper_circle = b2FixtureDef(shape=b2CircleShape(pos=(0, self.L2 - self.L1), radius=self.L1), isSensor=True) # Upper part
        rectangle = b2FixtureDef(shape=b2PolygonShape(vertices=(
            (-self.L1, 0),
            (self.L1, 0),
            (self.L1, self.L2-self.L1),
            (-self.L1, self.L2-self.L1))), isSensor=True) # Middle part
        
        self.sensors = []
        
        base_circle = self.body.CreateFixture(base_circle, density=0)
        base_circle.userData = {'init_angle_render': math.pi, 'touching_hard': [], 'touching_sensor': []}
        self.sensors.append(base_circle)

        upper_circle = self.body.CreateFixture(upper_circle, density=0)
        upper_circle.userData = {'init_angle_render': 0, 'touching_hard': [], 'touching_sensor': []}
        self.sensors.append(upper_circle)

        rectangle = self.body.CreateFixture(rectangle, density=0)
        rectangle.userData = {'lines_render': [[(-self.L1, self.L2-self.L1), (-self.L1, 0)], [(self.L1, 0), (self.L1, self.L2-self.L1)]], 'touching_hard': [], 'touching_sensor': []}
        self.sensors.append(rectangle)

        # MAX LENGTH of the ship 
        self.MAX_LENGTH = np.sqrt(max(
                self.SHIP_WIDTH ** 2 + self.SHIP_HEIGHT ** 2,
                (self.SHIP_WIDTH / 2) ** 2 + (self.SHIP_HEIGHT * 1.1) ** 2))
                

        
        self.body.color = getColor(idx=0)
        self.body.linearVelocity = (0.0,0.0) # TODO Maybe add a random initial velocity in [0, Vmax] at the start for robustness
        self.body.angularVelocity = 0 # TODO same as above
        self.body.userData = self

        # FIXME Defining ship mass and center of gravity manually. Use Box2D's automatic calculation instead by setting the correct densities for each fixtures.
        newMassData = self.body.massData
        newMassData.mass = Ship.SHIP_MASS
        newMassData.center = (0.0, Ship.SHIP_HEIGHT/2) #FIXME Is this the correct correct center of mass for a ship ?
        newMassData.I = Ship.SHIP_INERTIA + Ship.SHIP_MASS*(newMassData.center[0]**2+newMassData.center[1]**2) # inertia is defined at origin location not localCenter
        self.body.massData = newMassData

    def can_see(self, obstacle: Obstacle):
        """ Is the obstacle in the observation radius ? """
        return obstacle.distance_to_ship < self.obs_radius

    def bumper_state(self, hard=True, sensor=True, ignore=[]):
        """ Get a set of Box2D bodies touching the bumper sensors """
        # FIXME again it uses Box2D objects directly when it should perhaps interact with user defined class instead
        touches = set()
        for sens in self.sensors:
            if hard: # Get hard objects
                touches.update(sens.userData['touching_hard'])
            if sensor: # Get other sensors
                touches.update(sens.userData['touching_sensor'])
        for body in ignore:
            if body in touches:
                touches.remove(body) # Remove bodies we want to ignore
        return touches

    def thrust(self, inc_throttle, fps=30):
        """ Increment the throttle (increment is a scalar in [-1, 1]) at a given fps """
        inc_throttle = np.clip(inc_throttle, -1, 1)
        inc_throttle = inc_throttle * Ship.THRUSTER_MAX_THROTTLE_STEP / fps
        self.throttle = np.clip(self.throttle + inc_throttle, self.THRUSTER_MIN_THROTTLE, 1) # Throttle is a % so can't go over 1 and below the minimum %


    def steer(self, steer, fps=30):
        """ Increment the thurster angle (increment is a scalar in [-1, 1]) at a given fps """
        steer = np.clip(steer, -1, 1)
        steer = steer * Ship.THRUSTER_MAX_ANGLE_STEP / fps
        self.thruster_angle = np.clip(self.thruster_angle + steer, -Ship.THRUSTER_MAX_ANGLE, Ship.THRUSTER_MAX_ANGLE)

    def clean(self):
        super().clean()
        self.throttle = 1 # FIXME Shouldn't be using magic numbers like this
        self.thruster_angle = 0
        self.dont_render_fixture = []

    def add_geoms(self, viewer, ship_view):
        """ Add geoms to the viewer depending on if it's the radar or not """
        color = self.get_color()

        # If it is the global view
        if not ship_view:

            # Create thruster geom + add translations
            thruster = rendering.FilledPolygon((
                (-self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, 0),
                (self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT),
                (-self.THRUSTER_WIDTH / 2, -self.THRUSTER_HEIGHT)))
        
            thruster.add_attr(self.thrustertrans) # add thruster angle, assigned later
            thruster.add_attr(self.shiptrans) # add ship angle and ship position, assigned later
            thruster.set_color(*color)
        
            viewer.add_geom(thruster)
        
            # Center of gravity geom
            COG = rendering.FilledPolygon((
                (-Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, -Ship.THRUSTER_WIDTH/0.2),
                (Ship.THRUSTER_WIDTH / 0.2, 0),
                (0, Ship.THRUSTER_WIDTH/0.2)))
            COG.add_attr(self.shiptrans) # add ship angle and ship position
            
            COG.set_color(0, 0, 0)

            viewer.add_geom(COG)

            
            # Draw the observation radius
            if self.obs_radius:
                horizon = rendering.make_circle(radius=self.obs_radius, res=60, filled=False)
                horizon.set_color(*color)
                horizon.add_attr(self.shiptrans) # add ship angle and ship position

                viewer.add_geom(horizon)

        # Draw the ship itself
        # FIXME Shouldn't draw bumper perhaps, or at least just an average sized bumper and make the actual one random around this mean -> Since seeing the bumper leaks info on ship obstacle behavior (see _take_action in ShipObstacle class)
        shapes = []
        for f in self.body.fixtures:
            if f in self.dont_render_fixture:
                continue # Ignore the fixture
            isSensor = f in self.sensors # Is it part of the bumper ?
            if type(f.shape) is polygonShape:
                if not isSensor: # Polygon that is not a sensor
                    shape = rendering.FilledPolygon(f.shape.vertices)
                    shape.set_color(*color)
                    shapes.append(shape)
                else:
                    for line in f.userData['lines_render']: # Render each line (to avoid having the width lines in the middle of the bumper
                        shape = rendering.PolyLine(line, False)
                        shape.set_color(*(133, 114, 216))
                        shapes.append(shape)
            elif type(f.shape) is circleShape:
                if isSensor:
                    shape = make_half_circle(radius=f.shape.radius, init_angle=f.userData['init_angle_render'], filled=False) # Draw only half of the circle
                    shape.set_color(*(133, 114, 216))
                else:
                    shape = rendering.make_circle(f.shape.radius)
                    shape.set_color(*color)
                shape.add_attr(rendering.Transform(translation=f.shape.pos))
                shapes.append(shape)
            
        # Add the correct transformation depending on the view + add geom to the view
        for shape in shapes:
            if ship_view:
                shape.add_attr(self.ship_view_trans)
            else:
                shape.add_attr(self.shiptrans)
            shape.userData = self
            viewer.add_geom(shape)


    def render(self, viewer, first_time=True, ship_view=None):
        if first_time:
                self.add_geoms(viewer, ship_view) # Add the geoms if it is the first time for this viewer

        # Update transforms
        self.thrustertrans.set_rotation(self.thruster_angle)
        self.thrustertrans.set_translation(0, -self.SHIP_HEIGHT / 2)
        self.shiptrans.set_translation(*self.body.position)
        self.shiptrans.set_rotation(self.body.angle)
        
    
    def update(self, addTraj=False):
        """ Called at each step to update whatever is needed """
        if addTraj:
            self.trajPos.append(self.body.position) # Add current position to the history

    def step(self, fps):
        COGpos = self.body.GetWorldPoint(self.body.localCenter) # Center of gravity pos

        force_thruster = (-np.sin(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE * self.throttle,
                  np.cos(self.body.angle + self.thruster_angle) * self.THRUSTER_MAX_FORCE * self.throttle) # Get the thruster force from the throttle % and world thruster angles
        
        localVelocity = self.body.GetLocalVector(self.body.linearVelocity)
        force_damping_in_ship_frame = (-localVelocity[0] * Ship.K_Yv,-localVelocity[1] *Ship.K_Xu) # Local damping
        
        force_damping = self.body.GetWorldVector(force_damping_in_ship_frame) # Translate local damping to global
        force_damping = (np.cos(self.body.angle)* force_damping_in_ship_frame[0] -np.sin(self.body.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.body.angle)* force_damping_in_ship_frame[0] + np.cos(self.body.angle) * force_damping_in_ship_frame[1] ) # Adjust damping to body angle
        torque_damping = -self.body.angularVelocity *Ship.K_Nr

        # Apply all forces and torque
        self.body.ApplyTorque(torque=torque_damping,wake=False)
        self.body.ApplyForce(force=force_thruster, point=self.body.position, wake=False)
        self.body.ApplyForce(force=force_damping, point=COGpos, wake=False)


# FIXME Should this really be a separate inherited class ?
class ShipLidar(Ship):
    """
    Ship with lidars.
    Extends the ship base class.
    """
    def __init__(self, world, init_angle, position, nb_lidars, lidar_range, **kwargs):
        Ship.__init__(self, world, init_angle, position, **kwargs)
        self.nb_lidars = nb_lidars
        self.lidar_range = lidar_range
        self.lidars = [LidarCallback(dont_report_type = [BodyType.TARGET], dont_report_object=[self]) for _ in range(self.nb_lidars)]
        self.update()

    def _update_lidars(self):
        """ Update lidar start and finishing points and cast the ray with Box2D """
        pos = self.body.position
        angle = self.body.angle + np.pi/2 # Just to get thing straigther in my head (with the angle now being to the ship y axis = height axis)
        nb_lib_after_basics = self.nb_lidars - 3 # number of lidars left after using the 3 basics (front and sides)
        nb_back = nb_lib_after_basics // 3 # Number to place place facing to the lower half
        nb_front_left = (nb_lib_after_basics - nb_back) // 2 #  number in upper left quarter
        nb_front_right = nb_lib_after_basics - nb_back - nb_front_left # number in upper right quarter
        # FIXME Either add lidars if upper left and upper right are not the same or make the upper right the one with more since with colregs we usually turn to starboard side (right) meaning we are more likely to bump into something on the right

        # TODO Clean up all these cos and sin and angle transformation, they are confusing
        for i, lidar in enumerate(self.lidars): #
            lidar.fraction = 1.0
            lidar.p1 = pos
            if i == 0:
                # front
                lidar.p2 = (
                    pos[0] + math.sin(angle + math.pi / 2) * self.lidar_range,
                    pos[1] - math.cos(angle + math.pi / 2) * self.lidar_range)
            elif i == 1:
                # right
                lidar.p2 = (
                    pos[0] + math.sin(angle) * self.lidar_range,
                    pos[1] - math.cos(angle) * self.lidar_range)
            elif i == 2:
                # left
                lidar.p2 = (
                    pos[0] + math.sin(angle + math.pi) * self.lidar_range,
                    pos[1] - math.cos(angle + math.pi) * self.lidar_range)
            # Fill the back
            elif i < nb_back + 3:
                j = i - 3
                lidar.p2 = (
                    pos[0] + math.sin((-math.pi * (j + 1)) / (nb_back + 1) + angle) * self.lidar_range,
                    pos[1] - math.cos((-math.pi * (j + 1)) / (nb_back + 1) + angle) * self.lidar_range)
            # Fill the front left
            elif i < 3 + nb_back + nb_front_left:
                j = i - 3 - nb_back
                lidar.p2 = (
                    pos[0] + math.sin((math.pi / 2 * (j + 1)) / (nb_front_left + 1) + angle + math.pi / 2) * self.lidar_range,
                    pos[1] - math.cos((math.pi / 2 * (j + 1)) / (nb_front_left + 1) + angle + math.pi / 2) * self.lidar_range)
            # Fill the front right
            else:
                j = i - 3 - nb_back - nb_front_left
                lidar.p2 = (
                    pos[0] + math.sin((math.pi / 2 * (j + 1)) / (nb_front_right + 1) + angle) * self.lidar_range,
                    pos[1] - math.cos((math.pi / 2 * (j + 1)) / (nb_front_right + 1) + angle) * self.lidar_range)
                    
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
    
    def update(self, addTraj = False):
        self._update_lidars()
        Ship.update(self, addTraj)
        
    def render(self, viewer, first_time=True, ship_view=None):
        Ship.render(self, viewer, first_time, ship_view)
        if not ship_view:
            # Draw lidars
            for lidar in self.lidars:
                viewer.draw_polyline( [lidar.p1, lidar.p2], color=rgb(255, 0, 0), linewidth=1)
            # Draw trajectory
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
    """ Ship obstacles that inherit both from Ship and Obstacle classes """

    # Bumper dimensions (reduced compared to our agent ship)
    L1 = 0.8 * Ship.SHIP_HEIGHT
    L2 = 3.2 * Ship.SHIP_HEIGHT

    def __init__(self, world, init_angle, position, **kwargs):
        #super(Obstacle).__init__(world, init_x, init_y)
        Obstacle.clean(self)
        Ship.__init__(self, world, init_angle, position, 0, **kwargs)
        self.target_world_direction = self.body.GetWorldVector((0,1)) # Global angle it wants to go in
        self.fixture_in_front = None # Fixture to know if something is in front

    def _build(self, *args, **kwargs):
        Ship._build(self, *args, **kwargs)
        self.body.color1 = rgb(255, 255, 255) # white
        self.body.color2 = rgb(41, 14, 9) # darker brown
        self.body.color3 = rgb(83, 43, 9) # brown

    def render(self, viewer, first_time=True, ship_view=None):
        Ship.render(self, viewer, first_time, ship_view)
        if not ship_view:
            # Draw the angle it wants to follow
            target_dir_p2 = [ax * 20 + self.body.position[i] for i, ax in enumerate(self.target_world_direction)]
            viewer.draw_polyline( [self.body.position, target_dir_p2], color=rgb(255, 0, 0), linewidth=1)

    def take_actions(self, fps):
        """ Handwritten AI logic to kinda obey COLREGS """

        if not self.fixture_in_front:
            # Create a sensor in front of ship to detect incomming obstacle
            # TODO Maybe make it symmetric to thruster angle (x axis as symmetry axis) to look in front in the direction where you are going instead of in front where the ship is going
            fixture_in_front_def = b2FixtureDef(shape=b2PolygonShape(vertices=(
                (-self.SHIP_WIDTH, -self.SHIP_HEIGHT / 2),
                (+self.SHIP_WIDTH, -self.SHIP_HEIGHT / 2),
                (+self.SHIP_WIDTH, self.L2),
                (-self.SHIP_WIDTH, self.L2))), isSensor=True, userData={'touching_hard': [], 'touching_sensor': []})
            self.fixture_in_front = self.body.CreateFixture(fixture_in_front_def, density=0)
            self.dont_render_fixture.append(self.fixture_in_front)
        touch_front =  set(self.fixture_in_front.userData['touching_hard'])

        if any(x.userData.type != BodyType.TARGET and x != self for x in touch_front): # If there is something in front other tha n itself or the target
            self.thrust(-1, fps) # slow down
            self.steer(1, fps) # steer to starboard as much as possible
            return

        touches = self.bumper_state(ignore=[self.world.userData.target.body]) # Get all touches including sensors
        bearings = [np.arctan2(*reversed(self.body.GetLocalPoint(body.position))) for body in touches]
        for bearing in bearings:
            if bearing <= math.pi / 2 and bearing >= -math.pi / 6: # Detected something on front starboard side
                self.steer(1, fps) # go starboard
                return

        angle_to_target_dir = np.arctan2(*reversed(self.body.GetLocalVector(self.target_world_direction))) - math.pi/2
        self.thrust(1, fps) # accelerate
        if abs(angle_to_target_dir) > math.pi / 6: # We are deviating from our original angle

            # Make a ray cast in the direction you want to go
            look_to_target_lidar = LidarCallback(dont_report_type = [BodyType.TARGET], dont_report_object=[self])
            look_to_target_lidar.p1 = self.body.position
            look_to_target_lidar.p2 = self.body.GetWorldPoint((np.cos(angle_to_target_dir + math.pi/2) * self.L2, np.sin(angle_to_target_dir + math.pi/2) * self.L2)) 
            self.world.RayCast(look_to_target_lidar, look_to_target_lidar.p1, look_to_target_lidar.p2)
            # TODO Maybe use a sensor instead

            if look_to_target_lidar.fraction == 1: # Nothing where we want to go
                if angle_to_target_dir < 0 and angle_to_target_dir > -math.pi: 
                    self.steer(1, fps) # Steer port
                else:
                    self.steer(-1, fps) # Steer starboard
            else: # Something is in the way
                if angle_to_target_dir > 0 and self.thruster_angle < 0: # upper-left corner and turning in this direction (>0 since we did - pi/2 and positive angle goes to pi max)
                    self.steer(1, fps) # Turn in the other direction
                elif angle_to_target_dir > (-math.pi/2) and angle_to_target_dir < 0 and self.thruster_angle > 0: # upper-right and turning in this direction
                    self.steer(-1, fps)


        else:
            self.steer(-self.thruster_angle / (self.THRUSTER_MAX_ANGLE_STEP / fps), fps) #straighten up


           
    
    def step(self, fps):
        self.take_actions(fps)
        Ship.step(self, fps)

    def unsee(self):
        Obstacle.unsee(self)
        self.bearing_to_ship = Obstacle.DEFAULT_BEARING # Different from bearing from ship
    
    
    def get_color(self):
        return self.body.color2 if self.seen else self.body.color1 if self.is_hit() else self.body.color1

class ShipObstacleRand(ShipObstacle):
    """ Ship obstacle with random behavior """
    def __init__(self, world, init_angle, position, **kwargs):
        ShipObstacle.__init__(self, world, init_angle, position, **kwargs)
        self.body.color1 = rgb(60,185,119)

    def take_actions(self, fps):
        steer = np.random.uniform(-1, 1)
        thrust = np.random.uniform(-1, 1)

        self.steer(steer, fps)
        self.thrust(thrust, fps)
    
class Rock(RoundObstacle):
    """ Circular shaped static obstacle """
    RADIUS = 20
    def __init__(self, world, position, **kwargs):
        super().__init__(world, position, **kwargs)
        self.type = BodyType.ROCK

    def _build(self, position, **kwargs):
        radius = np.random.uniform(0.5*Rock.RADIUS,2*Rock.RADIUS)

        self.body = self.world.CreateStaticBody(
            position=position,
            fixtures=fixtureDef(
            shape = circleShape(pos=(0,0),radius = radius),
            categoryBits=0x0010, # FIXME No use ?
            maskBits=0x1111, # FIXME No use ?
            restitution=1.0))
        self.body.color1 = rgb(83, 43, 9) # brown
        self.body.color2 = rgb(41, 14, 9) # darker brown
        self.body.color3 = rgb(255, 255, 255) # seen

        self.radius = radius

        self.body.userData = self

# FIXME Target shouldn't be a box2D object. Should just be coordinates with a radius (would avoid collisions, sensors etc handling exceptions)
class Target(RoundObstacle):
    """ Target (box2D sensor) which is a circular zone our agent has to enter to successfully end the episode """
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
            restitution=0.1, isSensor=True, userData={'touching_hard': [], 'touching_sensor': []}))
        self.body.userData = self
    
    def render(self, viewer, first_time=True, ship_view=None):
        trans = self.body.transform
        for f in self.body.fixtures:
            t = rendering.Transform(translation=trans * f.shape.pos)
            viewer.draw_circle(f.shape.radius, color= rgb(15,15,15), filled=False, linewidth=2).add_attr(t)
