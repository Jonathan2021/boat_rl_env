from shipNavEnv.envs import ShipNavMultiShipsLidarRadar
from shipNavEnv.Worlds import ShipsAndRocksWorld
from functools import partial

class ShipNavComplete(ShipNavMultiShipsLidarRadar):
    """
    Complete environment with rocks, ship obstacles and an agent with lidars.
    Inherits from env with ship obstacles and agent with lidar but no rocks.
    """
    possible_kwargs = ShipNavMultiShipsLidarRadar.possible_kwargs.copy()
    possible_kwargs.update({'waypoints': False})
    possible_kwargs.pop('scale', None)
    possible_kwargs.update({'rock_scale': ShipsAndRocksWorld.SCALE_SHIP, 'ship_scale': ShipsAndRocksWorld.SCALE_SHIP})

    def _build_world(self):
        """
        World with ships and rocks and other chosen parameters
        """
        world = ShipsAndRocksWorld(self.n_ships, self.n_rocks, self.n_lidars, self.ship_scale, self.rock_scale, {'obs_radius': self.obs_radius,'display_traj':self.display_traj}, self.waypoints)
        world.update_obstacle_data = partial(world.update_obstacle_data, rocks=False)
        return world
