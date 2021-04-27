import Box2D
from shipNavEnv.Bodies import Ship, Rock, Target, Body
import abc
from shipNavEnv.Callbacks import ContactDetector, PlaceOccupied
import numpy as np
import math

class World:
    GRAVITY = (0,0)
    WIDTH = 900
    HEIGHT = 1600

    def __init__(self):
        self.world = Box2D.b2World(gravity = World.GRAVITY)
        self.listener = ContactDetector()
        self.bodies = []
        self.ships =  []
        self.target = None
        self.rocks = []
        self.ship = None

    @abc.abstractmethod
    def populate(self):
        pass

    def reset(self):
        self.destroy()
        self.world.contactListener = self.listener
        self.populate()

    def destroy(self):
        for body in self.bodies:
            body.destroy()
        self.bodies = []

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


class EmptyWorld(World):
    def __init__(self):
        super().__init__()

    def populate(self):
        angle = self.get_random_angle()
        self.ship = Ship(self.world, angle, 0, 0)
        self.get_random_free_space(self.ship)

        self.target = Target(self.world, 0, 0)
        self.get_random_free_space(self.target)

        self.bodies.append(self.ship)
        self.bodies.append(self.target)


class RockOnlyWorld(EmptyWorld):
    def __init__(self, n_rocks=20):
        super().__init__()
        self.n_rocks = n_rocks

    def populate(self, nb_rocks = 20):
        for i in range(self.n_rocks):
            x, y = self.get_random_pos()
            rock = Rock(self.world, x, y)
            self.rocks.append(rock)
            self.bodies.append(rock)

        super().populate()


class ShipsOnlyMap(World):
    def __init__(self):
        super().__init__()

class ShipsAndRocksMap(World):
    def __init__(self):
        super().__init__()

class ImpossibleMap(World):
    def __init__(self):
        super().__init__()
