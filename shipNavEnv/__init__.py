from gym.envs.registration import register

register(
    id='ShipNav-v0',
    entry_point='shipNavEnv.envs:ShipNavRocks',
)

register(
    id='ShipNav-v1',
    entry_point='shipNavEnv.envs:ShipNavRocksLidar',
)

register(
    id='ShipNav-v2',
    entry_point='shipNavEnv.envs:ShipNavMultiShipsRadius',
)

register(
    id='ShipNav-v3',
    entry_point='shipNavEnv.envs:ShipNavRocksContinuousSteer',
)

register(
    id='ShipNav-v4',
    entry_point='shipNavEnv.envs:ShipNavRocksSteerAndThrustContinuous',
)

register(
    id='ShipNav-v5',
    entry_point='shipNavEnv.envs:ShipNavMultiShipsLidar',
)

register(
    id='ShipNav-v6',
    entry_point='shipNavEnv.envs:ShipNavMultiShipsLidarRadar',
)

register(
    id='ShipNav-v7',
    entry_point='shipNavEnv.envs:ShipNavComplete',
)
