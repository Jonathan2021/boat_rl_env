from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from shipNavEnv.utils import int_round_iter
import math

class GridAdapter:
    """
    Class to make a compressed grid from rocks and find a path to a target in it using a chosen algorithm.
    """
    def __init__(self, left, bottom, right, top, down_scale=10, margin=1):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

        self.down_scale = down_scale # How compressed to we want our grid to be ?
        self.dim1 = int(round((top - bottom) / self.down_scale)) # Height
        self.dim2 = int(round((right - left) / self.down_scale)) # Width
        self.map = [[1 for _ in range(self.dim2)] for _ in range(self.dim1)] # Actual grid (matrix)
        self.algos = {'Astar': AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)} # Can add different algos from the pathfinding library
        self.start = None
        self.end = None
        self.grid = None
        self.path = []
        self.margin = margin # How safe we want to be (the higher the margin the larger the rock appears to be)

    def _add_circle(self, x0, y0, radius):
        """
        Draw a circle in the grid at a given point and a given radius
        """
        f = 1 - radius
        ddf_x = 1
        ddf_y = -2 * radius
        x = 0
        y = radius
        self.map[self.safe_y(y0 + radius)][self.safe_x(x0)] = 0
        self.map[self.safe_y(y0 - radius)][self.safe_x(x0)] = 0
        self.map[self.safe_y(y0)][self.safe_x(x0 + radius)] = 0
        self.map[self.safe_y(y0)][self.safe_x(x0 - radius)] = 0

        while x < y:
            if f >= 0:
                y -= 1
                ddf_y +=2
                f += ddf_y
            x += 1
            ddf_x += 2
            f += ddf_x    
            self.map[self.safe_y(y0 + y)][self.safe_x(x0 + x)] = 0
            self.map[self.safe_y(y0 + y)][self.safe_x(x0 - x)] = 0
            self.map[self.safe_y(y0 - y)][self.safe_x(x0 + x)] = 0
            self.map[self.safe_y(y0 - y)][self.safe_x(x0 - x)] = 0
            self.map[self.safe_y(y0 + x)][self.safe_x(x0 + y)] = 0
            self.map[self.safe_y(y0 + x)][self.safe_x(x0 - y)] = 0
            self.map[self.safe_y(y0 - x)][self.safe_x(x0 + y)] = 0
            self.map[self.safe_y(y0 - x)][self.safe_x(x0 - y)] = 0
    
    def add_rock(self, rock):
        """ Add a Rock object to the grid """
        x, y = self.transform_coords_to_grid(*rock.body.position) # Get grid coordinates

        # Get transformed radius
        radius = (rock.radius + self.margin) / self.down_scale
        dround = radius - round(radius)
        if dround > 0 and dround < self.margin / (2.5*self.down_scale):
            radius = int(round(radius))
        else:
            radius = math.ceil(radius)

        self._add_circle(x, y, radius) # Add the circle

    def add_rocks(self, rocks):
        """ Add a list of rocks """
        for rock in rocks:
           self.add_rock(rock) 

    def safe_x(self, x):
        """ Get x in the map bounds """
        return min(max(0, x), self.dim2 - 1)

    def safe_y(self, y):
        """ Get y in map bounds """
        return min(max(y, 0), self.dim1 - 1)

    def transform_coords_to_grid(self, x, y):
        """ Downscale coordinates from the original world to the grid """
        return self.safe_x(int(round((x - self.left) / self.down_scale))), self.safe_y(int(round((y - self.bottom) / self.down_scale)))

    def transform_coords_to_world(self, x, y):
        """ Grid coordinates to original world coordinates """
        return self.down_scale*x + self.left, y*self.down_scale + self.bottom

    def find_path(self, start, end, algo='Astar'):
        """
        Find a path from start to end using a specified algorithm
        """
        if not self.grid:
            self.grid = Grid(matrix=self.map)
        self.start = self.transform_coords_to_grid(*start)
        self.end = self.transform_coords_to_grid(*end)
        start = self.grid.node(*self.start)
        end = self.grid.node(*self.end)
        finder = self.algos[algo]
        self.path, runs = finder.find_path(start, end, self.grid)
        return list(map(lambda x : self.transform_coords_to_world(*x), self.path)) # Return path in original world coordinates
