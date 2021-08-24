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
    GRAVITY = (0,0)
    HEIGHT = 450
    WIDTH = 800
    DIAGONAL = np.sqrt(HEIGHT ** 2 + WIDTH ** 2)
    WAYPOINT_RADIUS = Ship.SHIP_HEIGHT

    def __init__(self, ship_kwargs=None, scale = 1, waypoint_support=True):
        self.ship_kwargs = ship_kwargs
        self.listener = ContactDetector()
        self.world = Box2D.b2World(contactListener = self.listener, gravity = World.GRAVITY, userData=self)
        #self.world.userData = self
        self.ships =  []
        self.target = None
        self.rocks = []
        self.ship = None
        self.scale = scale
        self.populate()
        self.n_obstacles = len(self.get_obstacles())

        self.waypoint_support = waypoint_support
        
        if self.waypoint_support:
            self.path = []
            self.grid = None
            self.waypoints = []
            self.do_all_grid_stuff()
        #print(self.ship.body.position)
        #print(self.target.body.position)
        #print(self.rocks[0].body.position)

    def make_grid(self, scale = 1):
        self.grid = GridAdapter(*self.get_bounds(self.scale), margin=self.ship.MAX_LENGTH / 2)
        self.grid.add_rocks(self.rocks)

    def get_ship_target_path(self):
        return self.grid.find_path(self.ship.body.position, self.target.body.position)

    def squeeze_path(self, margin=1):
        if not self.path:
            return
        new_path = [self.path[0]]
        index = 0
        while index < len(self.path) - 1:
            added = False
            for nextp in range(len(self.path) - 1, index, -1):
                p1 = self.path[index]
                p2 = self.path[nextp]
                angle = calc_angle_two_points(p1, p2)
                dy = math.sin(angle + math.pi / 2) * margin
                dx = math.cos(angle + math.pi / 2) * margin
                pairs = [(p1, p2), ((p1[0] + dx, p1[1] + dy), (p2[0] + dx, p2[1] + dy)), ((p1[0] - dx, p1[1] - dy), (p2[0] - dx, p2[1] - dy))]
                is_ok = True
                for (test1, test2) in pairs:
                    callback = CheckObstacleRayCallback(dont_report=[BodyType.TARGET, BodyType.SHIP]) #Had to add BodyType ship because it hit itself sometimes but probably a good idea because we will have ship at somepoint
                    self.world.RayCast(callback, test1, test2)
                    if  callback.hit_obstacle:
                        is_ok = False
                        break
                if is_ok or nextp - 1 == index:
                    new_path.append(p2)
                    index = nextp
                    break
        
        self.path = new_path

    def do_all_grid_stuff(self):
        self.make_grid()
        self.path = self.get_ship_target_path()
        self.expected_dist = 0
        if self.path:
            self.path[0] = tuple(self.ship.body.position)
            self.path[-1] = tuple(self.target.body.position)
            self.squeeze_path(margin=self.ship.MAX_LENGTH / 2)
        self.waypoints = self.path[1:-1] # stripping of target and ship
        self.estimate_dist()
    
    def estimate_dist(self):
        if self.waypoint_support and self.path:
            self.dist_estimate = get_path_dist(self.path)
        else:
            self.dist_estimate = 1.5 * self.get_ship_target_dist()

    def _build_ship(self, angle, position=(0,0)):
        return Ship(self.world, angle, position, **self.ship_kwargs if self.ship_kwargs else dict())

    def _add_obstacles(self):
        pass
    
    def populate(self):
        self._add_obstacles()
        angle = self.get_random_angle()
        self.ship = self._build_ship(angle)

        # Look ahead to avoid bad initialisations (with rock in front)
        mass = self.ship.body.massData
        ship_height = self.ship.SHIP_HEIGHT
        ship_width = self.ship.SHIP_WIDTH
        #look_ahead_angle = 45 * np.pi / 180
        #look_ahead_distance = 4 * ship_height / np.sin(look_ahead_angle)
        look_ahead_distance = 4 * ship_height
        #diagonal = np.sqrt(self.HEIGHT ** 2 + self.WIDTH ** 2)
        got_free_space = False
        for i in range(5, 0, -1):
            actual_dist = look_ahead_distance * i / 5
            look_ahead_def = b2FixtureDef(shape=b2PolygonShape(vertices=(
                (-ship_width / 2 - actual_dist / 2 , -ship_height / 2 - actual_dist /3),
            (+ship_width / 2 + actual_dist / 2, -ship_height /2 - actual_dist / 3),
            (+ship_width / 2 + actual_dist / 2, +ship_height / 2 + actual_dist),
            (-ship_width / 2 - actual_dist / 2, +ship_height / 2 + actual_dist),
            )))
            look_ahead_fixture = self.ship.body.CreateFixture(look_ahead_def, density = 0)
            got_free_space =  self.get_random_free_space(self.ship)
            self.ship.body.DestroyFixture(look_ahead_fixture)
            if got_free_space:
                break
        #if not got_free_space:
        #    print("No free space") # Maybe add a counter or something
        
        self.ship.body.massData = mass # Recalculates mass when destroying a fixture but since we calculated our own, put it back (or body won't move)

        self.target = Target(self.world, (self.WIDTH, self.HEIGHT))
        self.get_random_free_space(self.target, ignore_type=[BodyType.SHIP], dont_ignore=[self.ship])


    def reset(self):
        for body in self.get_bodies():
            body.destroy()
        
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []
        viewer = None
        #viewer = None
        #viewer_state = None
        self.populate()
        self.n_obstacles = len(self.get_obstacles())

        if self.waypoint_support:
            self.path = []
            self.grid = None
            self.waypoints = []
            self.do_all_grid_stuff()

        self.do_all_grid_stuff()

    def destroy(self):
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
            self.do_all_grid_stuff()

    def get_random_pos(self, scale=1):
        left, bottom, right, top = self.get_bounds(scale)
        return np.random.uniform( [left, bottom], [right, top])

    def get_bounds(self, scale=1):
        extension = (scale - 1) / 2
        return (0 - extension * self.WIDTH, 0 + - extension * self.HEIGHT, self.WIDTH * (1 + extension), self.HEIGHT * (1 + extension))

    
    def get_random_angle(self):
        return np.random.uniform(0, 2 * math.pi)

    # FIXME According to Box2D doc Caution: Do not create a body at the origin and then move it. If you create several bodies at the origin, then performance will suffer.
    # Fix idea -> use functools partial to create body with position and return it (or destroy it)
    def get_random_free_space(self, body : Body, trial = 0, limit = 200, ignore_type=[], dont_ignore=[]):
        if trial == limit:
            #if body.type == BodyType.TARGET:
            #    print("Limit Trial")
            return False # FIXME (maybe take something outside world border
        body_ = body.body
        query = PlaceOccupied(ignore=[body], ignore_type=ignore_type, dont_ignore=dont_ignore)
        position = self.get_random_pos()
        body_.position = position
        for fixture in body_.fixtures:
            aabb = fixture.GetAABB(0)
            self.world.QueryAABB(query, aabb)
            if query.fixture: #FIXME A bit ugly
                return self.get_random_free_space(body, trial +1, limit, ignore_type=ignore_type, dont_ignore=dont_ignore)
        return True        


    def get_bodies(self):
        return ([self.ship] if self.ship else []) + ([self.target] if self.target else []) + self.get_obstacles()

    def get_obstacles(self, rocks=True, ships=True):
        return (self.rocks if rocks else []) + (self.ships if ships else [])

    def get_next_objective(self):
        if self.waypoint_support and self.waypoints:
            return self.waypoints[0]
        return self.target

    def _get_pos_dist(self, x1, x2):
        x1, y1 = x1.body.position if isinstance(x1, Body) else x1
        x2, y2 = x2.body.position if isinstance(x2, Body) else x2
        return (x2 - x1, y2 - y1) 


    def _get_local_ship_pos_dist(self, x):
        return self.ship.body.GetLocalVector(self._get_pos_dist(self.ship, x))

    def get_ship_dist(self, x, use_waypoints=False):
        if not self.waypoint_support or not use_waypoints or not self.waypoints:
            dist = np.linalg.norm(self._get_local_ship_pos_dist(x))
        else:
            dist = get_path_dist([self.ship.body.position] + self.waypoints)
            dist += np.linalg.norm(self._get_pos_dist(self.waypoints[-1], x))
        dist -= (x.radius if hasattr(x, 'radius') else 0)
        dist = max(0, dist)
        return dist


    def get_ship_target_dist(self, use_waypoints=True):
        return self.get_ship_dist(self.target, use_waypoints=use_waypoints)
    
    def get_ship_objective_dist(self):
        return self.get_ship_dist(self.get_next_objective(), use_waypoints=False)

    def get_ship_standard_dist(self, x, use_waypoints=False):
        return 2 * self.get_ship_dist(x, use_waypoints=use_waypoints) / self.DIAGONAL - 1

    def get_ship_target_standard_dist(self, use_waypoints=True):
        return self.get_ship_standard_dist(self.target, use_waypoints)

    def get_ship_objective_standard_dist(self):
        return self.get_ship_standard_dist(self.get_next_objective(), use_waypoints=False)

    def get_ship_bearing(self, x):
        localPos = self._get_local_ship_pos_dist(x)
        return np.arctan2(localPos[0], localPos[1])

    def get_ship_target_bearing(self):
        return self.get_ship_bearing(self.target)

    def get_ship_objective_bearing(self):
        return self.get_ship_bearing(self.get_next_objective())

    def get_ship_standard_bearing(self, x):
        return self.get_ship_bearing(x) / np.pi

    def get_ship_target_standard_bearing(self):
        return self.get_ship_standard_bearing(self.target)

    def get_ship_objective_standard_bearing(self):
        return self.get_ship_standard_bearing(self.get_next_objective())

    def update_obstacle_data(self, rocks=True, ships=True):
        for obstacle in self.get_obstacles(rocks, ships):
                distance = self.get_ship_dist(obstacle)
                bearing = self.get_ship_bearing(obstacle)
                obstacle.distance_to_ship = distance
                obstacle.bearing_from_ship = bearing
                if obstacle.type == BodyType.SHIP:
                    obs_angle = obstacle.body.angle #+ math.pi/2 # the +math.pi/2 is to use y axis of ship instead of x for readability
                    obstacle.bearing_to_ship = np.arctan2(*reversed(self.ship.body.GetLocalVector((math.cos(obs_angle), math.sin(obs_angle)))))
                obstacle.seen = self.ship.can_see(obstacle)
                if not obstacle.seen:
                    obstacle.unsee()
    def update_waypoints(self):
        if self.waypoints:
            way_x, way_y = self.waypoints[0]
            ship_x, ship_y = self.ship.body.position
            if np.linalg.norm((way_x - ship_x, way_y - ship_y)) < self.WAYPOINT_RADIUS:
                #print("Changed waypoint")
                self.waypoints = self.waypoints[1:]
                    

    def step(self, fps, update_obstacles=True,addDotTraj=False):
        for body in self.get_bodies():
            body.step(fps)
        ### DEBUG ###
        #print('Step: %d \nShip: %s\nLocals: %s' % (self.stepnumber, self.ship, locals()))
        
        # one step forward
        velocityIterations = 8
        positionIterations = 3
        prev_dist = self.get_ship_objective_dist()
        self.world.Step(1.0 / fps, velocityIterations, positionIterations)
        self.delta_dist = prev_dist - self.get_ship_objective_dist()

        self.ship.update(addDotTraj)
        if update_obstacles:
            self.update_obstacle_data()
        if self.waypoint_support:
            self.update_waypoints()
        
        #print(self.get_ship_target_path())
    def is_success(self):
        return self.get_ship_target_dist() <= 0 # since - radius, that means center is inside

    #perturbartion_defaults = {
    #        'proba_rock_not_appearing': 0,
    #        'proba_ship_not_appearing': 0,
    #        'proba_ship_no_ais': 0,
    #        'proba_render_not_working': 0,
    #        'std_incorrect_position': 0,0
    #        }
    def render_ship_view(self, viewer, first_time, draw_traj, perturbation_dict=None):
        DEBORDER = 3

    #    if not perturbation_dict:
    #        perturbation_dict = self.perturbartion_defaults.copy()

        ship = self.ship
        radius = ship.obs_radius

        MAX_TRAIL_LENGTH = 5

        if first_time:
            background = rendering.FilledPolygon((
                (-DEBORDER * self.WIDTH, -DEBORDER * self.HEIGHT),
                (-DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER*self.WIDTH, -DEBORDER*self.HEIGHT)))

            background.set_color(0,0,0)
            viewer.add_geom(background)
            self.dots = {id(obstacle): [] for obstacle in self.get_obstacles(rocks=False)}
        
            
        for obstacle in self.get_obstacles():
            obstacle.render(viewer, first_time=first_time, ship_view=True)
            trans = obstacle.ship_view_trans
            trans.set_translation(*self.ship.body.GetLocalPoint(obstacle.body.position))
            angle = obstacle.body.angle
            ship_angle=self.ship.body.angle
            local_angle = np.arctan2(*reversed(self.ship.body.GetLocalVector((math.cos(angle), math.sin(angle)))))
            trans.set_rotation(local_angle)
            if draw_traj and obstacle.type == BodyType.SHIP:
                dot = rendering.make_circle(obstacle.SHIP_WIDTH / 2)
                dot.transform = rendering.Transform()
                dot.position = copy.copy(obstacle.body.position)
                #dot.obstacle = obstacle
                dot.set_color(*obstacle.get_color_ship_view())
                #dot.add_attr(rendering.Transform(translation=obstacle.body.position))
                dot.add_attr(dot.transform)
                self.dots[id(obstacle)].append(dot)
                self.dots[id(obstacle)] = self.dots[id(obstacle)][-MAX_TRAIL_LENGTH:]

        for trail in self.dots.values():
            for dot in trail:
                dot.transform.set_translation(*self.ship.body.GetLocalPoint(dot.position))
                viewer.add_onetime(dot)

        for geom in viewer.geoms + viewer.onetime_geoms: 
            if hasattr(geom, 'userData'):
                geom.set_color(*geom.userData.get_color_ship_view())

        x, y = self.ship.body.position
        left = - radius
        right = radius
        top = radius
        bottom = -radius


        viewer.set_bounds(left, right, bottom, top)


    def render(self, viewer, first_time):
        DEBORDER = 3
        cyan = rgb(126, 150, 233)

        ship = self.ship

        if first_time:

            
            water = rendering.FilledPolygon((
                (-DEBORDER * self.WIDTH, -DEBORDER * self.HEIGHT),
                (-DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER*self.WIDTH, -DEBORDER*self.HEIGHT)))

            water.set_color(*cyan)
            viewer.add_geom(water)

            if self.waypoint_support:
                path = rendering.PolyLine(self.path, False)
                path.set_linewidth(5)
                path.set_color(0, 0, 255)
                viewer.add_geom(path)

        for body in self.get_bodies():
            body.render(viewer, first_time)

        for geom in viewer.geoms + viewer.onetime_geoms: 
            if hasattr(geom, 'userData'):
                geom.set_color(*geom.userData.get_color())
        
        if self.waypoint_support:
            for i, waypoint in enumerate(self.waypoints):
                t = rendering.Transform(translation = waypoint)
                if i == 0:
                    viewer.draw_circle(self.WAYPOINT_RADIUS, color=(0,255,0), filled=False, linewidth=3).add_attr(t)
                else:
                    viewer.draw_circle(self.WAYPOINT_RADIUS, color=(0,0,255), filled=False, linewidth=3).add_attr(t)


                    
        #FIXME Feels pretty hacky, should check on that later
        # Adjusting window
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
    ROCK_SCALE_DEFAULT = 2

    def init_specific(self, rock_scale, n_rocks):
        self.n_rocks = n_rocks
        self.rock_scale = rock_scale

    def __init__(self, n_rocks, rock_scale = ROCK_SCALE_DEFAULT, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorld.init_specific(self, rock_scale, n_rocks)
        super().__init__(ship_kwargs, self.rock_scale, waypoint_support)

    def _add_obstacles(self):
        nb_rocks = np.random.randint(self.n_rocks * 0.5, self.n_rocks * 1.5) if self.n_rocks else 0
        for i in range(nb_rocks):
            pos = self.get_random_pos(scale = self.rock_scale)
            rock = Rock(self.world, pos)
            #print(rock.body.fixtures[0].GetAABB(0))
            self.rocks.append(rock)

class RockOnlyWorldLidar(RockOnlyWorld):
    def init_specific(self, n_lidars,):
        self.n_lidars = n_lidars

    def __init__(self, n_rocks, n_lidars, rock_scale = RockOnlyWorld.ROCK_SCALE_DEFAULT, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        RockOnlyWorld.__init__(self, n_rocks, rock_scale, ship_kwargs, waypoint_support)

    def _build_ship(self, angle, position=(0,0)):
        return ShipLidar(self.world, angle, position, self.n_lidars, 150, **self.ship_kwargs if self.ship_kwargs else dict())


class ShipsOnlyWorld(World):
    SCALE = 1.5

    def init_specific(self, n_ships, ship_scale):
        self.n_ships = n_ships
        self.ship_scale = ship_scale

    def __init__(self, n_ships, scale = SCALE, ship_kwargs=None, waypoint_support=False):
        ShipsOnlyWorld.init_specific(self, n_ships, scale)
        super().__init__(ship_kwargs, scale, waypoint_support=waypoint_support)

    def _add_obstacles(self):
        epsilon_random = 0.05
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
    SCALE = ShipsOnlyWorld.SCALE

    def __init__(self, n_ships, n_lidars, scale = SCALE, ship_kwargs=None, waypoint_support=False):
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        super().__init__(n_ships, scale, ship_kwargs, waypoint_support)
  
    def _build_ship(self, angle, position=(0,0)):
        return ShipLidar(self.world, angle, position, self.n_lidars, 150, **self.ship_kwargs if self.ship_kwargs else dict())

class ShipsAndRocksWorld(ShipsOnlyWorldLidar):
    SCALE_SHIP = ShipsOnlyWorldLidar.SCALE
    SCALE_ROCK = RockOnlyWorld.ROCK_SCALE_DEFAULT

    def __init__(self, n_ships, n_rocks, n_lidars, ship_scale = SCALE_SHIP, rock_scale = SCALE_ROCK, ship_kwargs=None, waypoint_support=True):
        RockOnlyWorld.init_specific(self, rock_scale, n_rocks)
        RockOnlyWorldLidar.init_specific(self, n_lidars)
        ShipsOnlyWorld.init_specific(self, n_ships, ship_scale)
        World.__init__(self, ship_kwargs, scale=max(self.rock_scale, self.ship_scale), waypoint_support=waypoint_support)

    def _add_obstacles(self):
        ShipsOnlyWorld._add_obstacles(self)
        RockOnlyWorld._add_obstacles(self)
        

    def populate(self):
        World.populate(self)
        


class ImpossibleMap(World):
    def __init__(self):
        super().__init__()
