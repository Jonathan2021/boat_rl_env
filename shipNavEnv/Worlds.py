import Box2D
from Box2D import b2PolygonShape, b2FixtureDef
from shipNavEnv.Bodies import Ship, Rock, Target, Body, ShipObstacle, ShipLidar
from shipNavEnv.Callbacks import ContactDetector, PlaceOccupied
from shipNavEnv.utils import rgb
import numpy as np
import math

class World:
    GRAVITY = (0,0)
    HEIGHT = 450
    WIDTH = 800

    def __init__(self, ship_kwargs=None):
        self.ship_kwargs = ship_kwargs
        self.listener = ContactDetector()
        self.world = Box2D.b2World(contactListener = self.listener, gravity = World.GRAVITY)
        self.ships =  []
        self.target = None
        self.rocks = []
        self.ship = None
        self.viewer = None
        self.populate()

    def _build_ship(self, angle, position=(0,0)):
        return Ship(self.world, angle, position, **self.ship_kwargs if self.ship_kwargs else dict())
    
    def populate(self):
        angle = self.get_random_angle()
        self.ship = self._build_ship(angle)

        # Look ahead to avoid bad initialisations (with rock in front)
        mass = self.ship.body.massData
        ship_height = self.ship.SHIP_HEIGHT
        ship_width = self.ship.SHIP_WIDTH
        look_ahead_angle = 45 * np.pi / 180
        look_ahead_distance = 4 * ship_height / np.sin(look_ahead_angle)
        #diagonal = np.sqrt(self.HEIGHT ** 2 + self.WIDTH ** 2)
        look_ahead_def = b2FixtureDef(shape=b2PolygonShape(vertices=(
            (-ship_width / 2,0),
            (+ship_width / 2, 0),
            (ship_width / 2 + np.cos(look_ahead_angle) * look_ahead_distance, np.sin(look_ahead_angle) * look_ahead_distance),
            (-ship_width - np.cos(look_ahead_angle) * look_ahead_distance, np.sin(look_ahead_angle) * look_ahead_distance)
            )))
        look_ahead_fixture = self.ship.body.CreateFixture(look_ahead_def, density = 0)
        self.get_random_free_space(self.ship)
        self.ship.body.DestroyFixture(look_ahead_fixture)
        self.ship.body.massData = mass # Recalculates mass when destroying a fixture but since we calculated our own, put it back (or body won't move)

        self.target = Target(self.world, (0, 0))
        self.get_random_free_space(self.target)

    def reset(self):
        for body in self.get_bodies():
            body.destroy()
        
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []
        self.viewer = None

        self.populate()

    def destroy(self):
        for body in self.get_bodies():
            body.destroy()
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []

    def get_random_pos(self, scale=1):
        extension = (scale - 1) / 2
        return np.random.uniform( [0 - extension * self.WIDTH, 0 + - extension * self.HEIGHT], [self.WIDTH * (1 + extension), self.HEIGHT * (1 + extension)])
    
    def get_random_angle(self):
        return np.random.uniform(0, 2 * math.pi)

    # FIXME According to Box2D doc Caution: Do not create a body at the origin and then move it. If you create several bodies at the origin, then performance will suffer.
    # Fix idea -> use functools partial to create body with position and return it (or destroy it)
    def get_random_free_space(self, body : Body, trial = 0, limit = 100):
        if trial == limit:
            return False # FIXME (maybe take something outside world border
        body_ = body.body
        query = PlaceOccupied(ignore=[body])
        position = self.get_random_pos()
        body_.position = position
        for fixture in body_.fixtures:
            aabb = fixture.GetAABB(0)
            self.world.QueryAABB(query, aabb)
            if query.fixture:
                return self.get_random_free_space(body, trial +1, limit)
        return True        


    def get_bodies(self):
        return ([self.ship] if self.ship else []) + ([self.target] if self.target else []) + self.get_obstacles()

    def get_obstacles(self):
        return self.rocks + self.ships

    def _get_local_ship_pos_dist(self, x):
        COGpos = self.ship.body.GetWorldPoint(self.ship.body.localCenter)
        x_distance = (x.body.position[0] - COGpos[0])
        y_distance = (x.body.position[1] - COGpos[1])
        return self.ship.body.GetLocalVector((x_distance,y_distance))

    def get_ship_dist(self, x):
        return np.linalg.norm(self._get_local_ship_pos_dist(x)) - (x.radius if hasattr(x, 'radius') else 0)

    def get_ship_target_dist(self):
        return self.get_ship_dist(self.target)

    def get_ship_standard_dist(self, x):
        return 2 * self.get_ship_dist(x) / np.maximum(self.WIDTH, self.HEIGHT) - 1

    def get_ship_target_standard_dist(self):
        return self.get_ship_standard_dist(self.target)

    def get_ship_bearing(self, x):
        localPos = self._get_local_ship_pos_dist(x)
        return np.arctan2(localPos[0], localPos[1])

    def get_ship_target_bearing(self):
        return self.get_ship_bearing(self.target)

    def get_ship_standard_bearing(self, x):
        return self.get_ship_bearing(x) / np.pi

    def get_ship_target_standard_bearing(self):
        return self.get_ship_standard_bearing(self.target)

    def update_obstacle_data(self):
        for obstacle in self.get_obstacles():
                distance = self.get_ship_dist(obstacle)
                bearing = self.get_ship_bearing(obstacle)
                obstacle.distance_to_ship = distance
                obstacle.bearing_from_ship = bearing
                obstacle.seen = self.ship.can_see(obstacle)
                if not obstacle.seen:
                    obstacle.unsee()
                    

    def step(self, fps):
        for body in self.get_bodies():
            body.step(fps)
        ### DEBUG ###
        #print('Step: %d \nShip: %s\nLocals: %s' % (self.stepnumber, self.ship, locals()))
        
        # one step forward
        velocityIterations = 8
        positionIterations = 3
        self.world.Step(1.0 / fps, velocityIterations, positionIterations)
        
        self.ship.update()
        self.update_obstacle_data()

    def render(self, mode='human', close=False):
        DEBORDER = 10
        cyan = rgb(126, 150, 233)

        #print([d.userData for d in self.drawlist])
        if close:
            if self.viewer:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        ship = self.ship

        if not self.viewer:

            self.viewer = rendering.Viewer(self.WIDTH, self.HEIGHT)
            
            water = rendering.FilledPolygon((
                (-DEBORDER * self.WIDTH, -DEBORDER * self.HEIGHT),
                (-DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER * self.WIDTH, DEBORDER*self.HEIGHT),
                (DEBORDER*self.WIDTH, -DEBORDER*self.WIDTH)))

            water.set_color(*cyan)
            self.viewer.add_geom(water)

        for body in self.get_bodies():
            body.render(self.viewer)
            
                    
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
        
        self.viewer.set_bounds(width_min,width_max,height_min,height_max)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class RockOnlyWorld(World):
    ROCK_SCALE_DEFAULT = 3
    def __init__(self, n_rocks, rock_scale = ROCK_SCALE_DEFAULT, ship_kwargs=None):
        self.n_rocks = n_rocks
        self.rock_scale = rock_scale
        super().__init__(ship_kwargs)

    def populate(self):
        for i in range(self.n_rocks):
            pos = self.get_random_pos(scale = self.rock_scale)
            rock = Rock(self.world, pos)
            self.rocks.append(rock)

        World.populate(self)

class RockOnlyWorldLidar(RockOnlyWorld):
    def __init__(self, n_rocks, n_lidars, rock_scale = RockOnlyWorld.ROCK_SCALE_DEFAULT, ship_kwargs=None):
        self.n_lidars = n_lidars
        RockOnlyWorld.__init__(self, n_rocks, rock_scale, ship_kwargs)

    def _build_ship(self, angle, position=(0,0)):
        return ShipLidar(self.world, angle, position, self.n_lidars, np.maximum(self.WIDTH, self.HEIGHT), **self.ship_kwargs if self.ship_kwargs else dict())


class ShipsOnlyWorld(World):
    def __init__(self, n_ships, ship_kwargs=None):
        self.n_ships = n_ships
        super().__init__(ship_kwargs)

    def populate(self):
        for i in range(self.n_ships):
            pos = self.get_random_pos()
            angle = self.get_random_angle()
            ship = ShipObstacle(self.world, angle, pos)
            self.ships.append(ship)

        super().populate()

class ShipsAndRocksMap(World):
    def __init__(self):
        super().__init__()

class ImpossibleMap(World):
    def __init__(self):
        super().__init__()
