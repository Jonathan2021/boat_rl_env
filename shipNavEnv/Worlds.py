import Box2D
from Box2D import b2PolygonShape, b2FixtureDef, b2ChainShape, b2EdgeShape, b2CircleShape
from shipNavEnv.Bodies import Ship, Rock, Target, Body, ShipObstacle, ShipObstacleRand, ShipLidar, BodyType
from shipNavEnv.Callbacks import ContactDetector, PlaceOccupied, CheckObstacleRayCallback
from shipNavEnv.utils import rgb, calc_angle_two_points, get_path_dist
from shipNavEnv.grid_logic.Grid import GridAdapter
import numpy as np
import random
import math
import copy
from gym.envs.classic_control import rendering

class World:
    """ World base class. Contains common logic and variables for all worlds """

    # World specifics
    GRAVITY = (0,0)
    HEIGHT = 450
    WIDTH = 800
    DIAGONAL = np.sqrt(HEIGHT ** 2 + WIDTH ** 2)
    WAYPOINT_RADIUS = Ship.SHIP_HEIGHT # How large should the waypoints be when we use pathfinding ?

    def __init__(self, ship_kwargs=None, scale = 1, waypoint_support=True):
        """ Init all variables, instanciate objects, populate world """
        self.ship_kwargs = ship_kwargs
        self.listener = ContactDetector() # Callback associated with the world that will be called on every contact
        self.world = Box2D.b2World(contactListener = self.listener, gravity = World.GRAVITY, userData=self) # Instanciate box2D world

        # Factory new variables (to be filled in as we populate or step the world)
        self.ships =  []
        self.target = None
        self.rocks = []
        self.ship = None
        self.scale = scale # How much do we overflow from the original dims

        # Populate the world
        self.populate()

        self.n_obstacles = len(self.get_obstacles()) # How many obstacles do we have ?

        self.waypoint_support = waypoint_support # Do we want waypoints ?
        
        if self.waypoint_support:
            # Waypoint specific variables
            self.path = []
            self.grid = None
            self.waypoints = []

            self.do_all_grid_stuff() # Do all grid related stuff (construction, pathfinding etc.)

    def make_grid(self, scale = 1):
        """ Build the matrix representing the world with rocks (ships aren't drawn) -> We can expect that terrain is known and mapped so it is not abnormal to draw the world with its rocks """
        self.grid = GridAdapter(*self.get_bounds(self.scale), margin=self.ship.MAX_LENGTH / 2) # Initiate the grid with wanted compression and margins
        self.grid.add_rocks(self.rocks) # Add all rocks

    def get_ship_target_path(self):
        """ Get the path from the ship to the target in the grid using pathfinding algorithms """
        return self.grid.find_path(self.ship.body.position, self.target.body.position)

    def squeeze_path(self, margin=1):
        """ Transform the path from adjacent cells to key waypoints.
        A waypoint is kept when it is the last cell from the path visible from the previous waypoint (starting with ship), and so on """
        if not self.path: # We don't have a pth
            return
        new_path = [self.path[0]] # Add the first point (ship to the start of the waypoint list)
        index = 0
        while index < len(self.path) - 1: # Iterate on path points (except target)
            added = False
            for nextp in range(len(self.path) - 1, index, -1): # Go backwards from the end of the path to the current point we are considering
                p1 = self.path[index] # current point (waypoint)
                p2 = self.path[nextp] # possible next waypoint
                angle = calc_angle_two_points(p1, p2) # The angle between the 2 points (world frame)

                # dx and dy are used to get points a bit left and a bit right of ship (proportional to the margin we want)
                dy = math.sin(angle + math.pi / 2) * margin 
                dx = math.cos(angle + math.pi / 2) * margin
                pairs = [(p1, p2), ((p1[0] + dx, p1[1] + dy), (p2[0] + dx, p2[1] + dy)), ((p1[0] - dx, p1[1] - dy), (p2[0] - dx, p2[1] - dy))] # pair of coordinates for ray casting
                # TODO use sensor instead of ray casting (because a small object could possible get in between rays and makes the whole logic simpler (but can keep the above points for making the sensor fixture))
                is_ok = True
                for (test1, test2) in pairs:
                    callback = CheckObstacleRayCallback(dont_report=[BodyType.TARGET, BodyType.SHIP]) # We only care about rocks so we don't report the target or ships
                    self.world.RayCast(callback, test1, test2)
                    if  callback.hit_obstacle:
                        is_ok = False # There is something in front so p2 can't be a valid waypoint
                        break
                if is_ok or nextp - 1 == index: # It's a valid point (there is a straight path between it and previous waypoint) or it is just the next point in list anyways
                    new_path.append(p2) # add the waypoint
                    index = nextp # mark it as the current waypoint
                    break
        
        self.path = new_path # Replace the old path with the new path

    def do_all_grid_stuff(self):
        """ Concatenates all the grid logic """
        self.make_grid() # Make the grid
        self.path = self.get_ship_target_path() # Find a path
        self.expected_dist = 0
        if self.path:
            # Add ship and target position to path extremities (necessary for the waypoint algorithm)
            self.path[0] = tuple(self.ship.body.position)
            self.path[-1] = tuple(self.target.body.position)
            self.squeeze_path(margin=self.ship.MAX_LENGTH / 2) # Keep waypoints only
        self.waypoints = self.path[1:-1] # stripping of target and ship from the path
        self.estimate_dist() # Estimate the distance based on path
    
    def estimate_dist(self):
        """ An estimate of the distance the ship will have to go through to reach the target """
        if self.waypoint_support and self.path: # There are waypoints or their is a direct connection to the end
            self.dist_estimate = get_path_dist(self.path) # Get the path distance
        else:
            self.dist_estimate = 1.5 * self.get_ship_target_dist() # There is no path (even direct, meaning the pathfinding didn't find a way (possible even if there is because we compress and add margins so rocks appear bigger)) so we arbitrarly scale the direct path by a factor (1.5 here) <- ugly

    def _build_ship(self, angle, position=(0,0)):
        """ Build a ship (here basic ship with no lasers """
        return Ship(self.world, angle, position, **self.ship_kwargs if self.ship_kwargs else dict())

    def _add_obstacles(self):
        """ Abstract method to add obstacles """
        pass
    
    def populate(self):
        """ Populate the world logic (add obstacles and build the ship """
        self._add_obstacles()
        angle = self.get_random_angle()
        self.ship = self._build_ship(angle)

        # Look ahead to avoid bad initialisations (with rock in front)
        mass = self.ship.body.massData # We keep massData in a variable because by adding a fixture then removing it we mess it up and the ship doesn't move anymore
        ship_height = self.ship.SHIP_HEIGHT
        ship_width = self.ship.SHIP_WIDTH
        look_ahead_distance = 4 * ship_height # How far ahead do we look
        got_free_space = False
        for i in range(5, 0, -1): # We try 5 times to find a place for the ship to fit, each time reducing the free area ahead
            actual_dist = look_ahead_distance * i / 5 # Actual look ahead distance
            look_ahead_def = b2FixtureDef(shape=b2PolygonShape(vertices=(
                (-ship_width / 2 - actual_dist / 2 , -ship_height / 2 - actual_dist /3),
            (+ship_width / 2 + actual_dist / 2, -ship_height /2 - actual_dist / 3),
            (+ship_width / 2 + actual_dist / 2, +ship_height / 2 + actual_dist),
            (-ship_width / 2 - actual_dist / 2, +ship_height / 2 + actual_dist),
            ))) # Sort of triangle in front of ship
            look_ahead_fixture = self.ship.body.CreateFixture(look_ahead_def, density = 0)
            got_free_space =  self.get_random_free_space(self.ship) # Try and find a free space
            self.ship.body.DestroyFixture(look_ahead_fixture) # FIXME Probably should create and delete the fixture outside of the loop ?
            if got_free_space:
                break
        
        self.ship.body.massData = mass # Recalculates mass when destroying a fixture but since we calculated our own, put it back (or body won't move)

        self.target = Target(self.world, (self.WIDTH, self.HEIGHT)) # Build the target
        self.get_random_free_space(self.target, ignore_type=[BodyType.SHIP], dont_ignore=[self.ship]) # Place it somewhere with no rocks but not where we are


    def reset(self):
        """ Reset the world to factory new (remove obstacles and ship and rebuild them in random place. Do pathfinding if wanted """
        for body in self.get_bodies():
            body.destroy()
        
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []
        viewer = None
        self.populate()
        self.n_obstacles = len(self.get_obstacles())

        if self.waypoint_support:
            self.path = []
            self.grid = None
            self.waypoints = []
            self.do_all_grid_stuff()

        self.do_all_grid_stuff()

    def destroy(self):
        """ Destroy the world (this is not climate change) """
        for body in self.get_bodies():
            body.destroy()
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []

        if self.waypoint_support:
            self.path = []
            self.grid = None
            self.waypoints = []

    def get_random_pos(self, scale=1):
        """ Get a random position in the scale (1 being the actual word boundaries) """
        left, bottom, right, top = self.get_bounds(scale)
        return np.random.uniform( [left, bottom], [right, top])

    def get_bounds(self, scale=1):
        """ Get the world bounds from a given scale """
        extension = (scale - 1) / 2
        return (0 - extension * self.WIDTH, 0 + - extension * self.HEIGHT, self.WIDTH * (1 + extension), self.HEIGHT * (1 + extension))

    
    def get_random_angle(self):
        """ Do I really need to document this ? """
        return np.random.uniform(0, 2 * math.pi)

    # TODO Move from AABB query (which is a rectangle necessarily meaning that our triangle is in reality querying a rectangular area) to a sensor fixture because AABB also have a high margin around the actual fixture it seems
    def get_random_free_space(self, body : Body, trial = 0, limit = 200, ignore_type=[], dont_ignore=[]):
        if trial == limit:
            return False # We didn't find a place FIXME (maybe take something outside world border?)
        body_ = body.body
        query = PlaceOccupied(ignore=[body], ignore_type=ignore_type, dont_ignore=dont_ignore) # Query object
        position = self.get_random_pos()
        body_.position = position
        for fixture in body_.fixtures: # FIXME Should just use a sensor instead of AABB query
            aabb = fixture.GetAABB(0)
            self.world.QueryAABB(query, aabb) # Query the given area
            if query.fixture: # Did we find a fixture, meaning the area is already occupied ? FIXME A bit ugly
                return self.get_random_free_space(body, trial +1, limit, ignore_type=ignore_type, dont_ignore=dont_ignore) # Try again with trial +1
        return True # It worked, we found a position


    def get_bodies(self):
        """ Get all the Body object in the map """
        return ([self.ship] if self.ship else []) + ([self.target] if self.target else []) + self.get_obstacles()

    def get_obstacles(self, rocks=True, ships=True):
        """ Get the obstacles """
        return (self.rocks if rocks else []) + (self.ships if ships else [])

    def get_next_objective(self):
        """ Get the next objective, which can be a waypoint or the targer itself """
        if self.waypoint_support and self.waypoints:
            return self.waypoints[0]
        return self.target

    # FIXME Name isn't appropriate since it doesn't return a distance
    def _get_pos_dist(self, x1, x2):
        """ Get the vector between 2 points """
        x1, y1 = x1.body.position if isinstance(x1, Body) else x1
        x2, y2 = x2.body.position if isinstance(x2, Body) else x2
        return (x2 - x1, y2 - y1) 


    # FIXME again returns a vector not a distance -> ill named
    def _get_local_ship_pos_dist(self, x):
        """ Get the local vector from ship to x """
        return self.ship.body.GetLocalVector(self._get_pos_dist(self.ship, x))

    def get_ship_dist(self, x, use_waypoints=False):
        """ Get the distance from ship to x. If use_waypoints then go through the waypoints and get the total distance """
        if not self.waypoint_support or not use_waypoints or not self.waypoints:
            dist = np.linalg.norm(self._get_local_ship_pos_dist(x))
        else:
            dist = get_path_dist([self.ship.body.position] + self.waypoints)
            dist += np.linalg.norm(self._get_pos_dist(self.waypoints[-1], x))
        dist -= (x.radius if hasattr(x, 'radius') else 0)
        dist = max(0, dist)
        return dist


    def get_ship_target_dist(self, use_waypoints=True):
        """ Get the distance between the ship and the target """
        return self.get_ship_dist(self.target, use_waypoints=use_waypoints)
    
    def get_ship_objective_dist(self):
        """ Get distance between the ship and the next objective (waypoint or target) """
        return self.get_ship_dist(self.get_next_objective(), use_waypoints=False)

    def get_ship_standard_dist(self, x, use_waypoints=False):
        """ Get the standardized distance to x from ship in [-1, 1] and esperance of 0 """
        return 2 * self.get_ship_dist(x, use_waypoints=use_waypoints) / self.DIAGONAL - 1

    def get_ship_target_standard_dist(self, use_waypoints=True):
        """ Standardized distance from ship to target """
        return self.get_ship_standard_dist(self.target, use_waypoints)

    def get_ship_objective_standard_dist(self):
        """ Standardized distance from ship to next objective (waypoint or target) """
        return self.get_ship_standard_dist(self.get_next_objective(), use_waypoints=False)

    def get_ship_bearing(self, x):
        """ Bearing from ship to x """
        localPos = self._get_local_ship_pos_dist(x)
        return np.arctan2(localPos[0], localPos[1])

    def get_ship_target_bearing(self):
        """ Bearing from ship to target """
        return self.get_ship_bearing(self.target)

    def get_ship_objective_bearing(self):
        """ Bearing from ship to objective """
        return self.get_ship_bearing(self.get_next_objective())

    def get_ship_standard_bearing(self, x):
        """ Standardized bearing to x """
        return self.get_ship_bearing(x) / np.pi

    def get_ship_target_standard_bearing(self):
        """ Standardized bearing to target """
        return self.get_ship_standard_bearing(self.target)

    def get_ship_objective_standard_bearing(self):
        """ Standardized bearing to objective """
        return self.get_ship_standard_bearing(self.get_next_objective())

    def update_obstacle_data(self, rocks=True, ships=True):
        """ Update obstacle data: was it seen ?, the distance to ship and the bearing. Useful for old radar """
        for obstacle in self.get_obstacles(rocks, ships):
                distance = self.get_ship_dist(obstacle)

                # Update distance and bearings
                bearing = self.get_ship_bearing(obstacle)
                obstacle.distance_to_ship = distance
                obstacle.bearing_from_ship = bearing
                if obstacle.type == BodyType.SHIP:
                    obs_angle = obstacle.body.angle #+ math.pi/2 # the +math.pi/2 is to use y axis of ship instead of x for readability
                    obstacle.bearing_to_ship = np.arctan2(*reversed(self.ship.body.GetLocalVector((math.cos(obs_angle), math.sin(obs_angle))))) # The bearing from target ship to my agent ship which is not the same as the other way around + pi
                obstacle.seen = self.ship.can_see(obstacle) # Can I see it ? (is it in obs radius ?)
                if not obstacle.seen: # If not, unsee it (reset distance and bearing to defaults)
                    obstacle.unsee()

    def update_waypoints(self):
        """ Update waypoints by removing the next to be reached if that is the case """
        if self.waypoints:
            way_x, way_y = self.waypoints[0]
            ship_x, ship_y = self.ship.body.position
            if np.linalg.norm((way_x - ship_x, way_y - ship_y)) < self.WAYPOINT_RADIUS:
                self.waypoints = self.waypoints[1:]
                    

    def step(self, fps, update_obstacles=True,addDotTraj=False):
        """ Step the World by updating everything and stepping the simulation """
        for body in self.get_bodies():
            body.step(fps) # Step every body (so we kinda step everybody)
        
        # one step forward

        # These are the defaults recommended by Box2D if I remember correctly
        velocityIterations = 8 # How precise do we want our velocity calculations ? (the higher the more precise)
        positionIterations = 3 # How precise do we want our positions

        # Step the world and store distance difference
        prev_dist = self.get_ship_objective_dist()
        self.world.Step(1.0 / fps, velocityIterations, positionIterations)
        self.delta_dist = prev_dist - self.get_ship_objective_dist()

        self.ship.update(addDotTraj) # Update ship
        if update_obstacles:
            self.update_obstacle_data() # Update obstacles (if we are using an old radar)
        if self.waypoint_support:
            self.update_waypoints() # update waypoints if they are supported
        
    def is_success(self):
        """ Did we succeed in getting to the final target """
        return self.get_ship_target_dist() <= 0 # <= 0 because we substract radius. So being inside give smaller than radius - radius which is < 0

    # TODO I commented this here but the initative was to add noise to states (to ship radar here)
    #perturbartion_defaults = {
    #        'proba_rock_not_appearing': 0,
    #        'proba_ship_not_appearing': 0,
    #        'proba_ship_no_ais': 0,
    #        'proba_render_not_working': 0,
    #        'std_incorrect_position': 0,0
    #        }

    def render_ship_view(self, viewer, first_time, draw_traj, perturbation_dict=None):
        """ Add geoms to the ship radar (image) """
        DEBORDER = 3 # How much do we overflow for the black background

        ship = self.ship
        radius = ship.obs_radius

        MAX_TRAIL_LENGTH = 5 # How long is the trail behind obstacle ships

        if first_time: # First time rendering
            background = rendering.FilledPolygon((
                (-DEBORDER * self.WIDTH, -DEBORDER * self.HEIGHT),
                (-DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER*self.WIDTH, -DEBORDER*self.HEIGHT)))

            background.set_color(0,0,0) # black background
            viewer.add_geom(background)
            self.dots = {id(obstacle): [] for obstacle in self.get_obstacles(rocks=False)} # Trail dict (keys being the obstacle object id)
        
            
        for obstacle in self.get_obstacles():
            obstacle.render(viewer, first_time=first_time, ship_view=True) # Add obstacle geoms
            # Update the transformation (position translation and angle rotation) to be in ship agent fps perspective
            trans = obstacle.ship_view_trans
            trans.set_translation(*self.ship.body.GetLocalPoint(obstacle.body.position)) # Translate
            angle = obstacle.body.angle
            ship_angle=self.ship.body.angle
            local_angle = np.arctan2(*reversed(self.ship.body.GetLocalVector((math.cos(angle), math.sin(angle))))) 
            trans.set_rotation(local_angle) # Rotate
            if draw_traj and obstacle.type == BodyType.SHIP: # We want to add a dot to trail
                dot = rendering.make_circle(obstacle.SHIP_WIDTH / 2)
                dot.transform = rendering.Transform()
                dot.position = copy.copy(obstacle.body.position)
                dot.set_color(*obstacle.get_color_ship_view()) # Will keep this color (so same as ship when created but won't change as ship changes color)
                dot.add_attr(dot.transform)
                self.dots[id(obstacle)].append(dot)
                self.dots[id(obstacle)] = self.dots[id(obstacle)][-MAX_TRAIL_LENGTH:] # Only keep the required trail length

        for trail in self.dots.values():
            for dot in trail:
                dot.transform.set_translation(*self.ship.body.GetLocalPoint(dot.position)) # Update dot position to be in local frame of the agent
                viewer.add_onetime(dot)

        for geom in viewer.geoms + viewer.onetime_geoms: 
            if hasattr(geom, 'userData'):
                geom.set_color(*geom.userData.get_color_ship_view()) # Update the color of the ships and rocks (dots don't have userData so they aren't updated

        # Set view corner coordinates
        x, y = self.ship.body.position
        left = - radius
        right = radius
        top = radius
        bottom = -radius


        viewer.set_bounds(left, right, bottom, top)


    def render(self, viewer, first_time):
        """ Render the main window """
        DEBORDER = 3
        cyan = rgb(126, 150, 233)

        ship = self.ship

        if first_time:

            # Add water
            water = rendering.FilledPolygon((
                (-DEBORDER * self.WIDTH, -DEBORDER * self.HEIGHT),
                (-DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER*self.WIDTH, -DEBORDER*self.HEIGHT)))

            water.set_color(*cyan)
            viewer.add_geom(water)

            if self.waypoint_support: # Add line between waypoints
                path = rendering.PolyLine(self.path, False)
                path.set_linewidth(5)
                path.set_color(0, 0, 255)
                viewer.add_geom(path)

        for body in self.get_bodies():
            body.render(viewer, first_time) # Add body geoms

        for geom in viewer.geoms + viewer.onetime_geoms: 
            if hasattr(geom, 'userData'):
                geom.set_color(*geom.userData.get_color()) # Set the color of the geoms
       
        # Add waypoint circles
        if self.waypoint_support:
            for i, waypoint in enumerate(self.waypoints):
                t = rendering.Transform(translation = waypoint)
                if i == 0:
                    viewer.draw_circle(self.WAYPOINT_RADIUS, color=(0,255,0), filled=False, linewidth=3).add_attr(t)
                else:
                    viewer.draw_circle(self.WAYPOINT_RADIUS, color=(0,0,255), filled=False, linewidth=3).add_attr(t)


                    
        #FIXME Feels pretty hacky, should check on that later
        # Adjusting window size to get bigger when we go out of frame
        width_min = min(0, ship.body.position[0]-2*Ship.SHIP_HEIGHT)
        width_max = max(self.WIDTH, ship.body.position[0]+2*Ship.SHIP_HEIGHT)
        height_min = min(0, ship.body.position[1]-2*Ship.SHIP_HEIGHT)
        height_max = max(self.HEIGHT, ship.body.position[1]+2*Ship.SHIP_HEIGHT)
        ratio_w = (width_max-width_min)/self.WIDTH
        ratio_h = (height_max-height_min)/self.HEIGHT
        if ratio_w > ratio_h:
            height_min *= ratio_w/ratio_h
            height_max *= ratio_w/ratio_h
        else:
            width_min *= ratio_h/ratio_w
            width_max *= ratio_h/ratio_w
        
        viewer.set_bounds(width_min,width_max,height_min,height_max)


class RockOnlyWorld(World):
    """ World with rocks only (no lidar for agent) """
    ROCK_SCALE_DEFAULT = 2 # How much do we overflow with rocks. We want to overflow so that agent doesn't try to go around the field where no rocks are.

    def init_specific(self, rock_scale, n_rocks):
        """ Save the original number of rocks and scale """
        self.n_rocks = n_rocks
        self.rock_scale = rock_scale

    def __init__(self, n_rocks, rock_scale = ROCK_SCALE_DEFAULT, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorld.init_specific(self, rock_scale, n_rocks)
        super().__init__(ship_kwargs, self.rock_scale, waypoint_support)

    def _add_obstacles(self):
        """ Add obstacles (number is random uniform centered around original number) """
        nb_rocks = np.random.randint(self.n_rocks * 0.5, self.n_rocks * 1.5) if self.n_rocks else 0
        for i in range(nb_rocks):
            # Get a random pos and create the rock object at that pos
            pos = self.get_random_pos(scale = self.rock_scale)
            # FIXME probably should get rock radius here and pass it to constructor
            rock = Rock(self.world, pos)
            self.rocks.append(rock)

class RockOnlyWorldLidar(RockOnlyWorld):
    """ World with rocks only but Ship is equipped with lidars """
    def init_specific(self, n_lidars,):
        """ Save the number of lidars """
        self.n_lidars = n_lidars

    def __init__(self, n_rocks, n_lidars, rock_scale = RockOnlyWorld.ROCK_SCALE_DEFAULT, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        RockOnlyWorld.__init__(self, n_rocks, rock_scale, ship_kwargs, waypoint_support)

    def _build_ship(self, angle, position=(0,0)):
        """ Build the Ship with lasers """
        return ShipLidar(self.world, angle, position, self.n_lidars, 150, **self.ship_kwargs if self.ship_kwargs else dict())


class ShipsOnlyWorld(World):
    """ World composed of Ships only (no lidar for agent) """
    SCALE = 1.5

    def init_specific(self, n_ships, ship_scale):
        """ Save the original number of ships and scale """
        self.n_ships = n_ships
        self.ship_scale = ship_scale

    def __init__(self, n_ships, scale = SCALE, ship_kwargs=None, waypoint_support=False):
        ShipsOnlyWorld.init_specific(self, n_ships, scale)
        super().__init__(ship_kwargs, scale, waypoint_support=waypoint_support)

    def _add_obstacles(self):
        """ Add the ships """
        epsilon_random = 0.05 # Probability of building a ship with random behavior
        n_ships = np.random.randint(self.n_ships * 0.5, self.n_ships * 1.5) if self.n_ships else 0
        for i in range(n_ships):
            pos = self.get_random_pos(scale=self.ship_scale)
            angle = self.get_random_angle()
            if random.random() < epsilon_random:
                ship = ShipObstacleRand(self.world, angle, pos)
            else:
                ship = ShipObstacle(self.world, angle, pos)
            self.ships.append(ship)

class ShipsOnlyWorldLidar(ShipsOnlyWorld):
    """ World composed of ships only and agent has lasers """
    SCALE = ShipsOnlyWorld.SCALE

    def __init__(self, n_ships, n_lidars, scale = SCALE, ship_kwargs=None, waypoint_support=False):
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        super().__init__(n_ships, scale, ship_kwargs, waypoint_support)
  
    def _build_ship(self, angle, position=(0,0)):
        return ShipLidar(self.world, angle, position, self.n_lidars, 150, **self.ship_kwargs if self.ship_kwargs else dict())

class ShipsAndRocksWorld(ShipsOnlyWorldLidar):
    """ World with ships and rocks as obstacles + agent has lasers """
    SCALE_SHIP = ShipsOnlyWorldLidar.SCALE
    SCALE_ROCK = RockOnlyWorld.ROCK_SCALE_DEFAULT

    def __init__(self, n_ships, n_rocks, n_lidars, ship_scale = SCALE_SHIP, rock_scale = SCALE_ROCK, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorld.init_specific(self, rock_scale, n_rocks)
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        ShipsOnlyWorld.init_specific(self, n_ships, ship_scale)
        World.__init__(self, ship_kwargs, scale=max(self.rock_scale, self.ship_scale), waypoint_support=waypoint_support)

    def _add_obstacles(self):
        """ Add rocks and ships """
        ShipsOnlyWorld._add_obstacles(self)
        RockOnlyWorld._add_obstacles(self)
        

    def populate(self):
        World.populate(self)
        

# TODO Maybe this can be a map with stuff agent has never seen (used for robustness) or with ships with different speed etc.
class ImpossibleMap(World):
    def __init__(self):
        super().__init__()
