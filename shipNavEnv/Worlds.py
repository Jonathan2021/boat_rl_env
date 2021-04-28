import Box2D
from shipNavEnv.Bodies import Ship, Rock, Target, Body
import abc
from shipNavEnv.Callbacks import ContactDetector, PlaceOccupied
import numpy as np
import math

class World:
    GRAVITY = (0,0)
    HEIGHT = 900
    WIDTH = 1600

    def __init__(self):
        self.listener = ContactDetector()
        self.world = Box2D.b2World(contactListener = self.listener, gravity = World.GRAVITY)
        self.ships =  []
        self.target = None
        self.rocks = []
        self.ship = None
        self.populate()

    def get_bodies(self):
        return ([self.ship] if self.ship else []) + ([self.target] if self.target else []) + self.get_obstacles()

    def get_obstacles(self):
        return self.rocks + self.ships

    @abc.abstractmethod
    def populate(self):
        pass

    def reset(self):
        self.destroy()
        self.populate()

    def destroy(self):
        for body in self.get_bodies():
            body.destroy()
        self.ship = None
        self.target = None
        self.ships = []
        self.rocks = []

    def get_random_pos(self):
        return np.random.uniform( [0 ,0], [World.WIDTH, World.HEIGHT])
    
    def get_random_angle(self):
        return np.random.uniform(0, 2 * math.pi)

    def get_random_free_space(self, body : Body, trial = 0, limit = 100):
        if trial == limit:
            return False # FIXME (maybe take something outside world border
        body_ = body.body
        query = PlaceOccupied()
        position = self.get_random_pos()
        body_.position = position
        for fixture in body_.fixtures:
            aabb = fixture.GetAABB(0)
            self.world.QueryAABB(query, aabb)
            if query.fixture:
                return self.get_random_free_space(body, trial +1, limit)
        return True        
    
    @abc.abstractmethod
    def step(self):
        pass

    

class EmptyWorld(World):
    def __init__(self, ship_kwargs=None):
        self.ship_kwargs = ship_kwargs
        super().__init__()

    def populate(self):
        angle = self.get_random_angle()
        self.ship = Ship(self.world, angle, 0, 0, **self.ship_kwargs if self.ship_kwargs else dict())
        self.get_random_free_space(self.ship)

        self.target = Target(self.world, 0, 0)
        self.get_random_free_space(self.target)

    def _get_local_ship_pos_dist(self, x):
        COGpos = self.ship.body.GetWorldPoint(self.ship.body.localCenter)
        x_distance = (x.body.position[0] - COGpos[0])
        y_distance = (x.body.position[1] - COGpos[1])
        return self.ship.body.GetLocalVector((x_distance,y_distance))

    def get_ship_dist(self, x):
        return np.linalg.norm(self._get_local_ship_pos_dist(x))

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
                    obstacle.reset()
                    

    def step(self, fps):
        COGpos = self.ship.body.GetWorldPoint(self.ship.body.localCenter)

        force_thruster = (-np.sin(self.ship.body.angle + self.ship.thruster_angle) * self.ship.THRUSTER_MAX_FORCE,
                  np.cos(self.ship.body.angle + self.ship.thruster_angle) * self.ship.THRUSTER_MAX_FORCE )
        
        localVelocity = self.ship.body.GetLocalVector(self.ship.body.linearVelocity)

        force_damping_in_ship_frame = (-localVelocity[0] * Ship.K_Yv,-localVelocity[1] *Ship.K_Xu)
        
        force_damping = self.ship.body.GetWorldVector(force_damping_in_ship_frame)
        force_damping = (np.cos(self.ship.body.angle)* force_damping_in_ship_frame[0] -np.sin(self.ship.body.angle) * force_damping_in_ship_frame[1],
                  np.sin(self.ship.body.angle)* force_damping_in_ship_frame[0] + np.cos(self.ship.body.angle) * force_damping_in_ship_frame[1] )
        
        torque_damping = -self.ship.body.angularVelocity *Ship.K_Nr

        self.ship.body.ApplyTorque(torque=torque_damping,wake=False)
        self.ship.body.ApplyForce(force=force_thruster, point=self.ship.body.position, wake=False)
        self.ship.body.ApplyForce(force=force_damping, point=COGpos, wake=False)

        ### DEBUG ###
        #print('Step: %d \nShip: %s\nLocals: %s' % (self.stepnumber, self.ship, locals()))
        
        # one step forward
        velocityIterations = 8
        positionIterations = 3
        self.world.Step(1.0 / fps, velocityIterations, positionIterations)
        
        self.update_obstacle_data()


class RockOnlyWorld(EmptyWorld):
    def __init__(self, n_rocks, ship_kwargs):
        self.n_rocks = n_rocks
        super().__init__(ship_kwargs)

    def populate(self, nb_rocks = 20):
        for i in range(self.n_rocks):
            x, y = self.get_random_pos()
            rock = Rock(self.world, x, y)
            self.rocks.append(rock)

        super().populate()

class ShipsOnlyMap(EmptyWorld):
    def __init__(self):
        super().__init__()

class ShipsAndRocksMap(EmptyWorld):
    def __init__(self):
        super().__init__()

class ImpossibleMap(EmptyWorld):
    def __init__(self):
        super().__init__()
