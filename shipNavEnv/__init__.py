import gym


def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
    )

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
    entry_point='shipNavEnv.envs:ShipNavMultiShips',
)

register(
    id='ShipNav-v3',
    entry_point='shipNavEnv.envs:ShipNavRocksContinuousSteer',
)

register(
    id='ShipNav-v4',
    entry_point='shipNavEnv.envs:ShipNavRocksSteerAndThrustContinuous',
)
