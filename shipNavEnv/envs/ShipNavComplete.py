from shipNavEnv.envs import ShipNavMultiShipsLidarRadar
from shipNavEnv.Worlds import ShipsAndRocksWorld
from functools import partial

class ShipNavComplete(ShipNavMultiShipsLidarRadar):
    possible_kwargs = ShipNavMultiShipsLidarRadar.possible_kwargs.copy()
    possible_kwargs.update({'waypoints':True})
    possible_kwargs.pop('scale', None)
    possible_kwargs.update({'rock_scale': ShipsAndRocksWorld.SCALE_SHIP, 'ship_scale': ShipsAndRocksWorld.SCALE_SHIP})

    def _build_world(self):
        world = ShipsAndRocksWorld(self.n_ships, self.n_rocks, self.n_lidars, self.ship_scale, self.rock_scale, {'obs_radius': self.obs_radius}, self.waypoints)
        world.update_obstacle_data = partial(world.update_obstacle_data, rocks=False)
        return world
