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
    entry_point='gym_ShipNavigation.envs:ShipNavigationWithOneObstacleEnv',
)
register(
    id='ShipNav-v1',
    entry_point='gym_ShipNavigation.envs:ShipNavigationWithTrafficEnv',
)
